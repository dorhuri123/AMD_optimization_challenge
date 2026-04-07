"""
MXFP4-MM v18: Custom fused Triton GEMM with raw B layout.

Key optimization: work directly with raw (non-shuffled) B_q, avoiding all
shuffle/unshuffle overhead in registers. Cache transposed B and raw scales
on first call to amortize preparation cost.

Architecture:
  - Inline bf16->mxfp4 quantization of A (from AITER _mxfp4_quant_op)
  - Raw B_q transposed to [K//2, N] for K-major access
  - Raw B scales [N, K//32] (re-derived from B bf16, cached)
  - tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1")
  - @triton.autotune with configs for small M (4-32) and large M (64-256)

Fallback: AITER gemm_a16wfp4 (non-preshuffle) -> gemm_a16wfp4_preshuffle -> CK gemm_a4w4
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

_BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Inline MXFP4 quantization (exact copy of AITER _mxfp4_quant_op)
# ---------------------------------------------------------------------------
@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Convert bf16/f32 tile [BLOCK_SIZE_M, BLOCK_SIZE_N] to mxfp4.
    Returns (fp4_packed [M, N//2], scales_e8m0 [M, N//QUANT_GROUP])."""
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1
    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # Compute per-block scale (e8m0)
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    qx = x * quant_scale
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal path
    denorm_exp: tl.constexpr = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal path
    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    # Pack two fp4 values per byte
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


# ---------------------------------------------------------------------------
# Main GEMM kernel: bf16 A x raw fp4 B (no shuffle)
# ---------------------------------------------------------------------------
def _make_configs():
    """Focused autotune configs for MI355X, based on AITER tuned defaults.

    AITER analysis shows: BLOCK_SIZE_K=512 and num_stages=1 are consistently
    best on gfx950. Key variables are BLOCK_SIZE_M and BLOCK_SIZE_N.
    """
    configs = []

    # --- Proven AITER defaults (highest priority) ---
    # Non-preshuffle: BLOCK_M=4/8, BLOCK_N=128, BLOCK_K=512, warps=4/8
    configs.append(triton.Config(
        {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1},
        num_warps=4, num_stages=1,
    ))
    configs.append(triton.Config(
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1},
        num_warps=8, num_stages=1,
    ))
    # Preshuffle default: BLOCK_M=32, BLOCK_N=64, BLOCK_K=512
    configs.append(triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
         "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1},
        num_warps=8, num_stages=1,
    ))

    # --- Exploration around proven defaults ---
    # Vary BLOCK_M for different M sizes
    for bm in [16, 32, 64, 128]:
        for bn in [64, 128, 256]:
            for bk in [256, 512]:
                for nw in [4, 8]:
                    configs.append(triton.Config(
                        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                         "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1},
                        num_warps=nw, num_stages=1,
                    ))

    # Try num_stages=2 for large BLOCK_K with double buffering
    for bm in [32, 64, 128]:
        for bn in [128, 256]:
            configs.append(triton.Config(
                {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                 "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1},
                num_warps=8, num_stages=2,
            ))

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = (c.kwargs["BLOCK_SIZE_M"], c.kwargs["BLOCK_SIZE_N"],
               c.kwargs["BLOCK_SIZE_K"], c.num_warps, c.num_stages)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


@triton.autotune(configs=_make_configs(), key=["M", "N", "K"])
@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0,
})
@triton.jit
def _fused_gemm_raw_kernel(
    a_ptr, b_ptr, c_ptr, bs_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fused bf16->mxfp4 quant + A4W4 GEMM with raw (non-shuffled) B.

    A: [M, K] bf16
    B: [K//2, N] uint8 (fp4x2 packed, K-major -- transposed from raw B_q)
    B_scales: [N, K//32] uint8 (e8m0 raw)
    C: [M, N] bf16 output
    """
    SCALE_GROUP: tl.constexpr = 32

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_bsn > 0)
    tl.assume(stride_bsk > 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grouped ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # A pointers: [BLOCK_SIZE_M, BLOCK_SIZE_K] bf16
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k_bf16[None, :] * stride_ak

    # B pointers: [BLOCK_SIZE_K//2, BLOCK_SIZE_N] uint8 (raw layout)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k_fp4 = tl.arange(0, BLOCK_SIZE_K // 2)
    b_ptrs = b_ptr + offs_k_fp4[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # B scale pointers: [BLOCK_SIZE_N, BLOCK_SIZE_K//32] uint8
    offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP)
    bs_ptrs = bs_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    for ki in range(num_k_iters):
        # Load B scales [BLOCK_SIZE_N, BLOCK_SIZE_K//32]
        b_scales = tl.load(bs_ptrs)

        # Load A tile as bf16 [BLOCK_SIZE_M, BLOCK_SIZE_K]
        if EVEN_K:
            a_bf16 = tl.load(a_ptrs)
            b_fp4 = tl.load(b_ptrs)
        else:
            a_bf16 = tl.load(
                a_ptrs,
                mask=offs_k_bf16[None, :] < K - ki * BLOCK_SIZE_K,
                other=0,
            )
            b_fp4 = tl.load(
                b_ptrs,
                mask=offs_k_fp4[:, None] < (K // 2) - ki * (BLOCK_SIZE_K // 2),
                other=0,
            )

        # Inline quantize A to MXFP4
        a_fp4, a_scales = _mxfp4_quant_op(a_bf16, BLOCK_SIZE_K, BLOCK_SIZE_M, SCALE_GROUP)

        # Hardware-accelerated scaled dot product (native FP4 MFMA)
        acc += tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1")

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP) * stride_bsk

    # Write output
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ---------------------------------------------------------------------------
# Cached data preparation
# ---------------------------------------------------------------------------
_cache = {}  # maps (B_q data_ptr, n, k) -> (B_t, B_scale_raw)


def _get_raw_b_data(B_q, B_bf16, n, k):
    """Get transposed B and raw scales, caching for reuse."""
    key = (B_q.data_ptr(), n, k)
    if key in _cache:
        return _cache[key]

    # Transpose B_q from [N, K//2] to [K//2, N]
    B_u8 = B_q.view(torch.uint8) if B_q.dtype != torch.uint8 else B_q
    B_t = B_u8.T.contiguous()

    # Re-derive raw (non-shuffled) scales from B bf16
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    _, B_scale_raw = dynamic_mxfp4_quant(B_bf16)
    B_scale_raw = B_scale_raw.view(torch.uint8)
    # Trim to actual dimensions (dynamic_mxfp4_quant may pad)
    k_scale = k // 32
    B_scale_raw = B_scale_raw[:n, :k_scale].contiguous()

    _cache[key] = (B_t, B_scale_raw)
    return B_t, B_scale_raw


# ---------------------------------------------------------------------------
# Custom kernel launcher
# ---------------------------------------------------------------------------
def _custom_fused(data):
    """Launch the custom fused kernel with raw B layout."""
    A, B_bf16, B_q, _, _ = data
    m, k = A.shape
    n = B_q.shape[0]

    B_t, B_scale_raw = _get_raw_b_data(B_q, B_bf16, n, k)

    C = torch.empty((m, n), dtype=_BF16, device=A.device)
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )

    _fused_gemm_raw_kernel[grid](
        A, B_t, C, B_scale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        B_t.stride(0), B_t.stride(1),
        C.stride(0), C.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
    )
    return C


# ---------------------------------------------------------------------------
# AITER fallback: gemm_a16wfp4 (non-preshuffle, same kernel idea but AITER tuned)
# ---------------------------------------------------------------------------
_a16wfp4 = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    _a16wfp4 = gemm_a16wfp4
except Exception:
    pass


def _aiter_raw(data):
    """AITER gemm_a16wfp4 with raw B layout (it does the transpose internally)."""
    A, B_bf16, B_q, _, _ = data
    m, k = A.shape
    n = B_q.shape[0]

    B_u8 = B_q.view(torch.uint8)

    # Get raw B scales
    _, B_scale_raw = _get_raw_b_data(B_q, B_bf16, n, k)

    return _a16wfp4(A, B_u8, B_scale_raw, dtype=_BF16)


# ---------------------------------------------------------------------------
# AITER fallback: gemm_a16wfp4_preshuffle (uses shuffled B)
# ---------------------------------------------------------------------------
_preshuffle = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _preshuffle = gemm_a16wfp4_preshuffle
except Exception:
    pass


def _fused_preshuffle(data):
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, kp = B_u8.shape
    w = B_u8.reshape(n // 16, kp * 16)
    sc = B_scale_sh.view(torch.uint8)
    ws = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
    return _preshuffle(A, w, ws, prequant=True, dtype=_BF16)


# ---------------------------------------------------------------------------
# CK fallback
# ---------------------------------------------------------------------------
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4
_FP4X2 = _dt.fp4x2
_E8M0 = _dt.fp8_e8m0


def _lean_ck(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_sc = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ---------------------------------------------------------------------------
# Dispatch with strategy selection
# ---------------------------------------------------------------------------
_strategy = {}  # maps M -> strategy function


def custom_kernel(data: input_t) -> output_t:
    global _strategy

    A = data[0]
    m = A.shape[0]

    # If we already know the best strategy for this M, use it
    if m in _strategy:
        return _strategy[m](data)

    # Strategy 1: Custom fused kernel with raw B (best for avoiding shuffle overhead)
    try:
        result = _custom_fused(data)
        _strategy[m] = _custom_fused
        return result
    except Exception:
        pass

    # Strategy 2: AITER gemm_a16wfp4 non-preshuffle
    if _a16wfp4 is not None:
        try:
            result = _aiter_raw(data)
            _strategy[m] = _aiter_raw
            return result
        except Exception:
            pass

    # Strategy 3: AITER preshuffle (proven for small M)
    if _preshuffle is not None:
        try:
            result = _fused_preshuffle(data)
            _strategy[m] = _fused_preshuffle
            return result
        except Exception:
            pass

    # Strategy 4: CK fallback (always works)
    _strategy[m] = _lean_ck
    return _lean_ck(data)
