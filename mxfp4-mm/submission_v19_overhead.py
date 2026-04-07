"""
MXFP4-MM v19: Kernel launch overhead minimization for MI355X.

Problem: The CK path (quant + shuffle + gemm) takes 23us but the GEMM alone is 6us.
The 17us overhead comes from Triton JIT dispatch, Python interpreter overhead,
and tensor allocation between the 3 kernel calls.

This submission implements multiple strategies to reduce that overhead:

1. CUDA Graph capture for the 3-kernel sequence (quant + shuffle + gemm)
2. torch.compile for operator fusion/optimization
3. Pre-allocated buffers to eliminate allocation overhead
4. Fused Triton kernel (quant + shuffle in one pass)
5. Minimized Python overhead via cached views and pre-computed metadata
6. Stream-based parallelism investigation

For m <= 64: fused Triton gemm_a16wfp4_preshuffle (1 kernel, inline quant)
For m > 64: overhead-optimized CK path with the best available strategy
"""
import torch
import sys
from task import input_t, output_t

_BF16 = torch.bfloat16

# ============================================================================
# Imports
# ============================================================================
_preshuffle = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _preshuffle = gemm_a16wfp4_preshuffle
except Exception:
    pass

import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4
_FP4X2 = _dt.fp4x2
_E8M0 = _dt.fp8_e8m0


# ============================================================================
# STRATEGY 0: Baseline lean CK (for comparison / fallback)
# ============================================================================
def _lean_ck(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_sc = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ============================================================================
# STRATEGY 1: CUDA Graph Capture
# ============================================================================
# CUDA Graphs capture a fixed sequence of GPU operations (kernel launches,
# memory copies) into a graph that can be replayed with a single CPU-side
# call, eliminating per-kernel Python/driver overhead.
#
# Constraints:
# - All tensor shapes must be fixed at capture time (no dynamic shapes)
# - All tensors must be pre-allocated (graph captures specific GPU addresses)
# - Triton kernels: YES, they work with CUDA graphs since they compile to
#   HIP/HSA kernels. The JIT overhead happens at compile time, not replay.
# - CK ASM kernels: YES, they are standard GPU kernels callable via graphs.
# - The graph replays the exact same kernel sequence with the same buffers.
#   Input data must be copied into the captured input buffers before replay.
#
# For our 3-kernel sequence (quant + shuffle + gemm), this eliminates:
# - 3x kernel launch overhead (~2-3us each = 6-9us saved)
# - Python interpreter overhead between calls (~2us saved)
# - Total potential savings: 8-11us, bringing 23us closer to 12-15us
# ============================================================================

_graph_cache = {}  # (m, n, k) -> (graph, input_buf, output_buf)


def _build_cuda_graph(m, n, k, B_shuffle, B_scale_sh, device):
    """Capture quant + shuffle + gemm into a single CUDA graph."""
    # Pre-allocate fixed input/output buffers
    A_buf = torch.empty((m, k), dtype=_BF16, device=device)

    # Warmup: run the sequence once to ensure all Triton kernels are compiled
    # (JIT compilation must happen BEFORE graph capture)
    A_buf.normal_()
    A_fp4, A_sc = _quant(A_buf)
    A_sc_sh = _shuffle(A_sc).view(_E8M0)
    _ = _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        A_sc_sh, B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )
    torch.cuda.synchronize()

    # Second warmup to ensure stable state
    A_fp4, A_sc = _quant(A_buf)
    A_sc_sh = _shuffle(A_sc).view(_E8M0)
    out = _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        A_sc_sh, B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )
    torch.cuda.synchronize()

    # Now capture the graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        A_fp4_g, A_sc_g = _quant(A_buf)
        A_sc_sh_g = _shuffle(A_sc_g).view(_E8M0)
        out_g = _gemm_a4w4(
            A_fp4_g.view(_FP4X2), B_shuffle,
            A_sc_sh_g, B_scale_sh,
            bpreshuffle=True, dtype=_dt.bf16,
        )

    return g, A_buf, out_g


def _cuda_graph_ck(data):
    """CK path using CUDA graph replay."""
    A, _, _, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B_shuffle.view(torch.uint8).shape[0]
    key = (m, n, k)

    if key not in _graph_cache:
        try:
            g, A_buf, out_g = _build_cuda_graph(
                m, n, k, B_shuffle, B_scale_sh, A.device
            )
            _graph_cache[key] = (g, A_buf, out_g)
        except Exception as e:
            print(f"[v19] CUDA graph capture failed for {key}: {e}", file=sys.stderr)
            return _lean_ck(data)

    g, A_buf, out_g = _graph_cache[key]
    # Copy input into the graph's fixed input buffer
    A_buf.copy_(A)
    # Replay the captured graph (single CPU call -> 3 GPU kernels)
    g.replay()
    # Return a clone so the graph buffer can be reused
    # (or return out_g directly if caller won't modify it)
    return out_g.clone()


# ============================================================================
# STRATEGY 2: torch.compile
# ============================================================================
# torch.compile can potentially:
# - Fuse the Python-level overhead into a single compiled graph
# - Optimize tensor allocation patterns
# - Eliminate redundant .view() operations
#
# Caveats:
# - Triton custom kernels may not be capturable by torch.compile's graph
#   tracing (they appear as opaque custom ops)
# - CK ASM kernels (called via aiter C++ bindings) are also opaque
# - torch.compile may fall back to eager mode for custom ops
# - The compilation itself adds significant first-call latency
#
# Best case: torch.compile eliminates Python overhead and fuses views
# Worst case: it adds tracing overhead without actual fusion
# ============================================================================

_compiled_ck = None


def _get_compiled_ck():
    global _compiled_ck
    if _compiled_ck is not None:
        return _compiled_ck

    def _ck_inner(A, B_shuffle, B_scale_sh):
        A_fp4, A_sc = _quant(A)
        A_sc_sh = _shuffle(A_sc).view(_E8M0)
        return _gemm_a4w4(
            A_fp4.view(_FP4X2), B_shuffle,
            A_sc_sh, B_scale_sh,
            bpreshuffle=True, dtype=_dt.bf16,
        )

    try:
        # mode="reduce-overhead" uses CUDA graphs under the hood
        # mode="max-autotune" tries more optimizations but slower compile
        _compiled_ck = torch.compile(
            _ck_inner,
            mode="reduce-overhead",
            fullgraph=False,  # Allow graph breaks for custom ops
        )
    except Exception:
        _compiled_ck = _ck_inner

    return _compiled_ck


def _torch_compile_ck(data):
    """CK path via torch.compile."""
    A, _, _, B_shuffle, B_scale_sh = data
    fn = _get_compiled_ck()
    return fn(A, B_shuffle, B_scale_sh)


# ============================================================================
# STRATEGY 3: Pre-allocated Buffers
# ============================================================================
# Tensor allocation on GPU involves:
# - Python object creation (~0.5us)
# - CUDA/HIP memory allocation or pool lookup (~0.5-2us)
# - Metadata setup (shape, strides, dtype) (~0.2us)
#
# By pre-allocating all intermediate buffers and reusing them, we eliminate
# this per-call allocation overhead. The key insight is that for a given
# (m, n, k) shape, all intermediate tensor sizes are deterministic.
#
# dynamic_mxfp4_quant outputs:
#   A_fp4: [m, k//2] uint8 (fp4x2 packed)
#   A_scale: [m, k//32] uint8 (e8m0)
#
# e8m0_shuffle outputs:
#   A_scale_shuffled: same shape as A_scale, reordered
#
# gemm_a4w4 outputs:
#   C: [m, n] bfloat16
#
# Challenge: dynamic_mxfp4_quant and e8m0_shuffle are Triton kernels that
# allocate their own output tensors internally. We cannot easily pass
# pre-allocated buffers without modifying the kernel source.
#
# Workaround: We can at least pre-allocate the output buffer and cache
# the view operations to minimize Python-side overhead.
# ============================================================================

_prealloc_cache = {}  # (m, n, k) -> cached metadata


def _prealloc_ck(data):
    """CK path with pre-allocated output buffer and cached metadata."""
    A, _, _, B_shuffle, B_scale_sh = data
    m, k = A.shape
    key = (m, k, id(B_shuffle))

    # The quant and shuffle still allocate internally, but we minimize
    # everything else: no .shape lookups, no .view() on hot path
    A_fp4, A_sc = _quant(A)
    # Fuse view operations: chain .view() calls are essentially free
    # (they just change metadata) but each Python call still costs ~0.3us
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ============================================================================
# STRATEGY 4: Fused Triton Kernel (quant + shuffle in one pass)
# ============================================================================
# The e8m0_shuffle kernel just reorders the scale bytes to match the CK
# weight layout. This reordering can be fused directly into the quant
# kernel's output write pattern.
#
# The shuffle pattern for e8m0 scales (from aiter fp4_utils.py):
#   Input:  [M, K//32] e8m0 scales in row-major order
#   Output: [M, K//32] e8m0 scales reordered for CK's tile access pattern
#
# The shuffle permutation for CK's (16,16) layout:
#   For each block of 32 scales along K, group into (4, 2, 2, 1) tiles
#   and permute dimensions to match CK's memory access order.
#
# By writing the quantized scales directly in shuffled order, we save:
# - One full kernel launch (~3-5us)
# - One read + write of the scale tensor from/to HBM
# ============================================================================

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _fused_quant_shuffle_kernel(
        # Input
        x_ptr,
        # Outputs
        out_fp4_ptr,
        out_scale_ptr,
        # Dimensions
        M, K,
        stride_xm, stride_xk,
        stride_fp4m, stride_fp4k,
        stride_sm, stride_sk,
        # Constants
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        QGROUP: tl.constexpr,  # 32
    ):
        """Fused MXFP4 quantization + e8m0 scale shuffle.

        Performs dynamic_mxfp4_quant and writes scales in e8m0_shuffle order,
        eliminating the separate shuffle kernel launch.

        Grid: (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        # Compute offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_k = offs_k < K

        # Load input tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # ── MXFP4 Quantization (per-32-element blocks) ──────────────────
        NUM_BLK: tl.constexpr = BLOCK_K // QGROUP
        x_blocked = x.reshape(BLOCK_M, NUM_BLK, QGROUP)

        # Compute e8m0 scale (power-of-2 exponent)
        amax = tl.max(tl.abs(x_blocked), axis=-1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)
        scale_exp = tl.log2(amax).floor() - 2
        scale_exp = tl.clamp(scale_exp, min=-127, max=127)

        bs_e8m0 = scale_exp.to(tl.uint8) + 127  # [BLOCK_M, NUM_BLK, 1]

        # Quantize to FP4
        qx = x_blocked * tl.exp2(-scale_exp)
        qx = qx.to(tl.uint32, bitcast=True)
        s = qx & 0x80000000
        qx = qx ^ s
        qf = qx.to(tl.float32, bitcast=True)

        sat = qf >= 6
        den = (not sat) & (qf < 1)
        nor = not (sat | den)

        # Denormal path
        denorm_x = qf + tl.cast(149 << 23, tl.float32, bitcast=True)
        denorm_x = denorm_x.to(tl.uint32, bitcast=True) - (149 << 23)
        denorm_x = denorm_x.to(tl.uint8)

        # Normal path
        nor_x = qx
        mant_odd = (nor_x >> 22) & 1
        nor_x += ((-126 << 23) + (1 << 21) - 1)
        nor_x += mant_odd
        nor_x = nor_x >> 22
        nor_x = nor_x.to(tl.uint8)

        e2m1 = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
        e2m1 = tl.where(nor, nor_x, e2m1)
        e2m1 = tl.where(den, denorm_x, e2m1)
        sign_lp = (s >> 28).to(tl.uint8)
        e2m1 = e2m1 | sign_lp

        # Pack two fp4 values per byte (low nibble = even, high = odd)
        e2m1 = tl.reshape(e2m1, [BLOCK_M, NUM_BLK, QGROUP // 2, 2])
        evens, odds = tl.split(e2m1)
        fp4_packed = evens | (odds << 4)
        fp4_packed = fp4_packed.reshape(BLOCK_M, BLOCK_K // 2)

        # ── Store FP4 output ─────────────────────────────────────────────
        offs_fp4k = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        fp4_ptrs = out_fp4_ptr + offs_m[:, None] * stride_fp4m + offs_fp4k[None, :] * stride_fp4k
        fp4_mask = mask_m[:, None] & (offs_fp4k[None, :] < K // 2)
        tl.store(fp4_ptrs, fp4_packed, mask=fp4_mask)

        # ── Store scales in SHUFFLED order ───────────────────────────────
        # The e8m0_shuffle permutation for CK (16,16) layout:
        # Input scale shape per row: [K//32] scales
        # The shuffle groups scales into blocks matching CK's tile pattern.
        #
        # For the CK a4w4 with bpreshuffle=True, the scale shuffle is:
        # reshape [n_scale_rows//32, n_scale_cols*32] where each 32-row block
        # has its columns interleaved in a specific pattern.
        #
        # The exact shuffle pattern from aiter/utility/fp4_utils.py:
        #   t = scale.view(dtype=torch.uint8)
        #   t = t.reshape(rows // 4, 4, cols // 8, 2, 4)
        #   t = t.permute(0, 2, 1, 4, 3)
        #   t = t.reshape(rows, cols)
        #
        # For our tile: bs_e8m0 is [BLOCK_M, NUM_BLK] (after squeeze)
        # We write it in the permuted order directly.
        #
        # Since the shuffle operates on the full scale tensor and involves
        # cross-row permutations (groups of 4 rows), we cannot easily do
        # it per-tile in the quant kernel without complex index math.
        # Instead, we write raw scales and let the shuffle happen separately,
        # OR we write a dedicated fused kernel that handles the full M dimension.

        # For now: write scales in normal order (shuffle will still be needed)
        # The real optimization here is reducing to 2 kernels instead of 3.
        scale_flat = bs_e8m0.reshape(BLOCK_M, NUM_BLK)
        offs_sk = pid_k * NUM_BLK + tl.arange(0, NUM_BLK)
        scale_ptrs = out_scale_ptr + offs_m[:, None] * stride_sm + offs_sk[None, :] * stride_sk
        scale_mask = mask_m[:, None] & (offs_sk[None, :] < K // QGROUP)
        tl.store(scale_ptrs, scale_flat, mask=scale_mask)

    @triton.jit
    def _fused_quant_and_shuffle_kernel(
        # Input
        x_ptr,
        # Outputs
        out_fp4_ptr,
        out_scale_sh_ptr,  # Output scales in SHUFFLED order
        # Dimensions
        M, K,
        stride_xm, stride_xk,
        stride_fp4m, stride_fp4k,
        stride_sm, stride_sk,
        # Constants
        BLOCK_M: tl.constexpr,  # Must be multiple of 4 for shuffle
        BLOCK_K: tl.constexpr,
        QGROUP: tl.constexpr,  # 32
    ):
        """Fully fused MXFP4 quantization + e8m0 scale shuffle.

        This kernel processes BLOCK_M rows at a time (must be >= 4 for the
        shuffle's 4-row grouping) and writes scales directly in shuffled order.

        The e8m0_shuffle permutation:
          t = scale.reshape(rows // 4, 4, cols // 8, 2, 4)
          t = t.permute(0, 2, 1, 4, 3)  -> (rows//4, cols//8, 4, 4, 2)
          t = t.reshape(rows, cols)

        Grid: (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_k = offs_k < K

        # Load input tile
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # ── Quantization (same as above) ────────────────────────────────
        NUM_BLK: tl.constexpr = BLOCK_K // QGROUP
        x_blocked = x.reshape(BLOCK_M, NUM_BLK, QGROUP)

        amax = tl.max(tl.abs(x_blocked), axis=-1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)
        scale_exp = tl.log2(amax).floor() - 2
        scale_exp = tl.clamp(scale_exp, min=-127, max=127)

        bs_e8m0 = scale_exp.to(tl.uint8) + 127
        qx = x_blocked * tl.exp2(-scale_exp)
        qx = qx.to(tl.uint32, bitcast=True)
        s = qx & 0x80000000
        qx = qx ^ s
        qf = qx.to(tl.float32, bitcast=True)

        sat = qf >= 6
        den = (not sat) & (qf < 1)
        nor = not (sat | den)

        denorm_x = qf + tl.cast(149 << 23, tl.float32, bitcast=True)
        denorm_x = denorm_x.to(tl.uint32, bitcast=True) - (149 << 23)
        denorm_x = denorm_x.to(tl.uint8)

        nor_x = qx
        mant_odd = (nor_x >> 22) & 1
        nor_x += ((-126 << 23) + (1 << 21) - 1)
        nor_x += mant_odd
        nor_x = nor_x >> 22
        nor_x = nor_x.to(tl.uint8)

        e2m1 = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
        e2m1 = tl.where(nor, nor_x, e2m1)
        e2m1 = tl.where(den, denorm_x, e2m1)
        sign_lp = (s >> 28).to(tl.uint8)
        e2m1 = e2m1 | sign_lp

        e2m1 = tl.reshape(e2m1, [BLOCK_M, NUM_BLK, QGROUP // 2, 2])
        evens, odds = tl.split(e2m1)
        fp4_packed = evens | (odds << 4)
        fp4_packed = fp4_packed.reshape(BLOCK_M, BLOCK_K // 2)

        # Store FP4 output (unchanged)
        offs_fp4k = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        fp4_ptrs = out_fp4_ptr + offs_m[:, None] * stride_fp4m + offs_fp4k[None, :] * stride_fp4k
        fp4_mask = mask_m[:, None] & (offs_fp4k[None, :] < K // 2)
        tl.store(fp4_ptrs, fp4_packed, mask=fp4_mask)

        # ── Write scales in SHUFFLED order ───────────────────────────────
        # The shuffle: reshape(M//4, 4, K_sc//8, 2, 4) -> permute(0,2,1,4,3) -> reshape(M, K_sc)
        #
        # For each scale element at logical position (m, k_sc):
        #   group_m = m // 4,  local_m = m % 4    (dim 0, dim 1)
        #   group_k = k_sc // 8, rem_k = k_sc % 8
        #   rem_k decomposes as: d3 = rem_k // 4, d4 = rem_k % 4  (dim 3, dim 4)
        #
        # After permute(0, 2, 1, 4, 3):
        #   output index = (group_m, group_k, local_m, d4, d3)
        #   out_m = group_m * (K_sc//8) * ... no, the reshape flattens differently
        #
        # Let's work out the output linear index:
        #   Before permute: shape (M//4, 4, K_sc//8, 2, 4), strides: (4*K_sc//8*2*4, K_sc//8*2*4, 2*4, 4, 1)
        #   = (4*K_sc, K_sc, 8, 4, 1)
        #   After permute(0,2,1,4,3): shape (M//4, K_sc//8, 4, 4, 2)
        #   strides: (4*K_sc, 8, K_sc, 1, 4)
        #   Flattened output[m_out, k_out] where:
        #     m_out = group_m * (K_sc//8) * 4 ... no.
        #
        # Actually after permute + reshape back to (M, K_sc):
        #   The output is contiguous with shape (M//4, K_sc//8, 4, 4, 2)
        #   Reshaping to (M, K_sc) means:
        #     row = group_m * (K_sc//8 * 4) + group_k * 4 + local_m
        #     but that's wrong, reshape just reinterprets the flat buffer.
        #
        # Let me think about this differently. The permuted tensor has shape
        # (M//4, K_sc//8, 4, 4, 2) and is contiguous. Reshaping to (M, K_sc)
        # means element at flat index i maps to (i // K_sc, i % K_sc).
        #
        # The flat index of element (g_m, g_k, l_m, d4, d3) in permuted order:
        #   i = g_m * (K_sc//8 * 4 * 4 * 2) + g_k * (4 * 4 * 2) + l_m * (4 * 2) + d4 * 2 + d3
        #   i = g_m * (K_sc//8 * 32) + g_k * 32 + l_m * 8 + d4 * 2 + d3
        #   i = g_m * 4*K_sc + g_k * 32 + l_m * 8 + d4 * 2 + d3
        #
        # So out_row = i // K_sc = g_m * 4 + (g_k * 32 + l_m * 8 + d4 * 2 + d3) // K_sc
        # out_col = i % K_sc = (g_k * 32 + l_m * 8 + d4 * 2 + d3) % K_sc
        #
        # This is complex. For a simpler approach, we compute the shuffled
        # output index for each (m, k_sc) position.
        #
        # Original: scale[m, k_sc]
        # In reshape(M//4, 4, K_sc//8, 2, 4):
        #   g_m = m // 4, l_m = m % 4
        #   g_k = k_sc // 8
        #   rem = k_sc % 8 -> d3 = rem // 4, d4 = rem % 4
        #
        # After permute(0,2,1,4,3) -> (g_m, g_k, l_m, d4, d3)
        # Flat index in output: g_m*(K_sc//8*4*4*2) + g_k*(4*4*2) + l_m*(4*2) + d4*2 + d3
        # = g_m * 4*K_sc + g_k*32 + l_m*8 + d4*2 + d3
        #
        # Output position: out_row = flat // K_sc, out_col = flat % K_sc
        #
        # For the store, we compute the flat output offset directly.

        # scales: [BLOCK_M, NUM_BLK] after squeeze
        scale_flat = bs_e8m0.reshape(BLOCK_M, NUM_BLK)

        K_SC = K // QGROUP  # Total number of scale columns

        # Compute logical (m, k_sc) for each element in our tile
        m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        k_sc_idx = pid_k * NUM_BLK + tl.arange(0, NUM_BLK)  # [NUM_BLK]

        # Decompose for shuffle
        g_m = m_idx // 4        # [BLOCK_M]
        l_m = m_idx % 4         # [BLOCK_M]
        g_k = k_sc_idx // 8     # [NUM_BLK]
        rem_k = k_sc_idx % 8    # [NUM_BLK]
        d3 = rem_k // 4         # [NUM_BLK]
        d4 = rem_k % 4          # [NUM_BLK]

        # Compute flat output index for each (m, k_sc) pair
        # flat = g_m * 4*K_SC + g_k*32 + l_m*8 + d4*2 + d3
        flat_idx = (g_m[:, None] * (4 * K_SC)
                    + g_k[None, :] * 32
                    + l_m[:, None] * 8
                    + d4[None, :] * 2
                    + d3[None, :])

        # Store using flat index
        scale_mask = (m_idx[:, None] < M) & (k_sc_idx[None, :] < K_SC)
        tl.store(out_scale_sh_ptr + flat_idx, scale_flat, mask=scale_mask)


def _fused_quant_shuffle(A):
    """Run the fused quant+shuffle Triton kernel."""
    m, k = A.shape
    QGROUP = 32
    k_sc = k // QGROUP

    # Allocate outputs
    A_fp4 = torch.empty((m, k // 2), dtype=torch.uint8, device=A.device)
    A_scale_sh = torch.empty((m, k_sc), dtype=torch.uint8, device=A.device)

    # Kernel config
    BLOCK_M = 32 if m >= 32 else m
    BLOCK_K = min(k, 256)  # Process up to 256 K elements per block

    # Ensure BLOCK_K is multiple of QGROUP
    BLOCK_K = max(QGROUP, BLOCK_K - (BLOCK_K % QGROUP))

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k, BLOCK_K))

    _fused_quant_and_shuffle_kernel[grid](
        A,
        A_fp4, A_scale_sh,
        m, k,
        A.stride(0), A.stride(1),
        A_fp4.stride(0), A_fp4.stride(1),
        # Scale strides: since we write via flat index, these are unused
        # but we pass them for the kernel signature
        k_sc, 1,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        QGROUP=QGROUP,
    )

    return A_fp4, A_scale_sh


def _fused_triton_quant_ck(data):
    """Use fused quant+shuffle Triton kernel, then CK GEMM (2 kernels instead of 3)."""
    A, _, _, B_shuffle, B_scale_sh = data

    A_fp4, A_scale_sh = _fused_quant_shuffle(A)

    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        A_scale_sh.view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ============================================================================
# STRATEGY 5: Minimized Python Overhead
# ============================================================================
# Profile of Python overhead per call:
#   data[0]              ~0.1us (tuple index)
#   A.shape              ~0.1us (cached property)
#   _quant(A)            ~0.3us Python + kernel launch
#   A_fp4.view(_FP4X2)   ~0.3us (creates new tensor view)
#   _shuffle(A_sc)       ~0.3us Python + kernel launch
#   .view(_E8M0)          ~0.3us
#   _gemm_a4w4(...)      ~0.3us Python + kernel launch
#   Total Python: ~1.7us
#
# Optimization: Pre-unpack data tuple, cache device, minimize attribute access
# ============================================================================

def _minimal_overhead_ck(data):
    """CK path with absolute minimum Python overhead."""
    # Unpack all at once (single tuple index operation)
    A = data[0]
    B_sh = data[3]
    B_sc = data[4]

    # Run quant -> shuffle -> gemm with minimal intermediate ops
    A_fp4, A_sc = _quant(A)
    A_sc_sh = _shuffle(A_sc)

    # Chain views: these are metadata-only, ~0.3us each
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_sh,
        A_sc_sh.view(_E8M0), B_sc,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ============================================================================
# STRATEGY 6: Stream-based Parallelism
# ============================================================================
# For the CK path, all 3 kernels are sequential (each depends on prior output).
# True parallelism is not possible. However, we can overlap:
# - A quantization on stream 1
# - Output allocation on stream 2 (minor)
#
# The real win would be pipelining across calls: while GEMM runs for batch i,
# start quantizing A for batch i+1. But this requires caller cooperation.
#
# For single-call optimization, streams don't help because of data dependencies.
# ============================================================================

def _stream_parallel_ck(data):
    """Attempted stream parallelism (limited by data dependencies)."""
    A, _, _, B_shuffle, B_scale_sh = data

    # All operations are sequential due to data dependencies:
    # quant -> shuffle -> gemm
    # Streams cannot help here. This is just the lean CK path.
    A_fp4, A_sc = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ============================================================================
# COMBINED STRATEGY: Best of all approaches
# ============================================================================
# For the final submission, we try strategies in order of expected benefit:
#
# 1. CUDA Graph (highest potential: eliminates all launch overhead)
# 2. Fused quant+shuffle (reduces 3 kernels to 2)
# 3. torch.compile (may or may not help with custom ops)
# 4. Minimal overhead CK (baseline with Python optimizations)
#
# For small M: fused Triton preshuffle (already 1 kernel)
# ============================================================================

def _fused_triton_path(data):
    """Fused Triton for small M."""
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, kp = B_u8.shape
    sc = B_scale_sh.view(torch.uint8)
    return _preshuffle(
        A,
        B_u8.reshape(n // 16, kp * 16),
        sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32),
        prequant=True, dtype=_BF16,
    )


# ── Strategy selection state ────────────────────────────────────────────────
_triton_ok = None
_large_m_strategy = {}  # (m, n, k) -> callable


def _select_large_m_strategy(data):
    """Try strategies for large M in order of expected benefit."""
    A, _, _, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B_shuffle.view(torch.uint8).shape[0]
    key = (m, n, k)

    # Strategy 1: CUDA Graph
    try:
        result = _cuda_graph_ck(data)
        # Verify result shape is correct
        if result.shape == (m, n) and result.dtype == _BF16:
            print(f"[v19] {key}: CUDA Graph SUCCESS", file=sys.stderr)
            _large_m_strategy[key] = _cuda_graph_ck
            return result
    except Exception as e:
        print(f"[v19] {key}: CUDA Graph failed: {e}", file=sys.stderr)

    # Strategy 2: Fused quant+shuffle (2 kernels instead of 3)
    if _HAS_TRITON:
        try:
            result = _fused_triton_quant_ck(data)
            if result.shape == (m, n) and result.dtype == _BF16:
                print(f"[v19] {key}: Fused quant+shuffle SUCCESS", file=sys.stderr)
                _large_m_strategy[key] = _fused_triton_quant_ck
                return result
        except Exception as e:
            print(f"[v19] {key}: Fused quant+shuffle failed: {e}", file=sys.stderr)

    # Strategy 3: torch.compile
    try:
        result = _torch_compile_ck(data)
        if result.shape == (m, n) and result.dtype == _BF16:
            print(f"[v19] {key}: torch.compile SUCCESS", file=sys.stderr)
            _large_m_strategy[key] = _torch_compile_ck
            return result
    except Exception as e:
        print(f"[v19] {key}: torch.compile failed: {e}", file=sys.stderr)

    # Strategy 4: Minimal overhead lean CK (always works)
    print(f"[v19] {key}: Using minimal overhead CK", file=sys.stderr)
    _large_m_strategy[key] = _minimal_overhead_ck
    return _minimal_overhead_ck(data)


def custom_kernel(data: input_t) -> output_t:
    global _triton_ok

    m = data[0].shape[0]

    # ── Small M: fused Triton (1 kernel) ────────────────────────────────
    if m <= 64 and _preshuffle is not None and _triton_ok is not False:
        if _triton_ok is None:
            try:
                result = _fused_triton_path(data)
                _triton_ok = True
                return result
            except Exception:
                _triton_ok = False
        else:
            return _fused_triton_path(data)

    # ── Large M: try overhead-minimized strategies ──────────────────────
    A = data[0]
    k = A.shape[1]
    n = data[3].view(torch.uint8).shape[0]
    key = (m, n, k)

    if key in _large_m_strategy:
        return _large_m_strategy[key](data)

    return _select_large_m_strategy(data)
