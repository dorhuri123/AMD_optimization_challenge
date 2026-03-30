"""
Custom Triton MLA decode kernel with MXFP4 KV cache — v2.0

AITER does NOT support MXFP4 KV in mla_decode_fwd — this is our custom kernel.

Design:
  - Decode only (q_seq_len=1): Q is one vector per head
  - MQA: 1 KV head, 16 Q heads — load KV once, reuse for all heads
  - MXFP4 KV: ~2x less HBM than fp8 (4-bit values + amortized E8M0 scales)
  - Online softmax: streaming over KV tiles
  - Two-pass approach: first compute scores, then accumulate values
    (simpler than single-pass for Triton, and KV tiles are loaded twice
     but the 2x savings from MXFP4 still nets ~1x vs fp8 single-pass)

Approach: "dequant-and-dot" — manually unpack fp4x2, decode E2M1, apply
E8M0 block scales, then standard dot products in fp32.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
MXFP4_BLOCK_SIZE = 32
NUM_SCALE_BLOCKS_QK = QK_HEAD_DIM // MXFP4_BLOCK_SIZE   # 18
NUM_SCALE_BLOCKS_V = V_HEAD_DIM // MXFP4_BLOCK_SIZE     # 16
PACKED_QK = QK_HEAD_DIM // 2   # 288
PACKED_V = V_HEAD_DIM // 2     # 256


@triton.jit
def _e2m1_decode(nibble):
    """Decode 4-bit E2M1 → float32. Input is uint8 with value 0-15."""
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    mant = nibble & 1
    exp_f = exp.to(tl.float32)
    mant_f = mant.to(tl.float32)
    normal_val = tl.math.exp2(exp_f - 1.0) * (1.0 + 0.5 * mant_f)
    denorm_val = 0.5 * mant_f
    val = tl.where(exp == 0, denorm_val, normal_val)
    val = tl.where(sign != 0, -val, val)
    return val


@triton.jit
def _dequant_fp4_block(
    kv_fp4_ptr, kv_scale_ptr,
    kv_idx,         # (BLOCK_KV,) global KV token indices
    mask_kv,        # (BLOCK_KV,) bool
    d_start,        # starting dim (must be multiple of 32)
    stride_kv_tok,
    stride_sc_tok,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Load and dequant a (BLOCK_KV, BLOCK_D) tile of KV as two halves: even and odd dims.

    Returns (lo_scaled, hi_scaled) each of shape (BLOCK_KV, BLOCK_D // 2) float32.
    lo_scaled[k, j] = dequantized value at KV token k, dim (d_start + 2*j)
    hi_scaled[k, j] = dequantized value at KV token k, dim (d_start + 2*j + 1)
    """
    packed_offsets = d_start // 2 + tl.arange(0, BLOCK_D // 2)

    # Load packed fp4x2: (BLOCK_KV, BLOCK_D // 2) uint8
    kv_packed = tl.load(
        kv_fp4_ptr + kv_idx[:, None] * stride_kv_tok + packed_offsets[None, :],
        mask=mask_kv[:, None], other=0,
    )

    # Unpack and decode E2M1
    lo_f32 = _e2m1_decode(kv_packed & 0x0F)
    hi_f32 = _e2m1_decode((kv_packed >> 4) & 0x0F)

    # Load E8M0 scales: (BLOCK_KV, BLOCK_D // 32)
    num_scales = BLOCK_D // MXFP4_BLOCK_SIZE
    scale_start = d_start // MXFP4_BLOCK_SIZE
    scale_offsets = scale_start + tl.arange(0, num_scales)
    scales = tl.load(
        kv_scale_ptr + kv_idx[:, None] * stride_sc_tok + scale_offsets[None, :],
        mask=mask_kv[:, None], other=127,
    )
    scale_f32 = tl.math.exp2(scales.to(tl.float32) - 127.0)

    # Apply scales: each scale covers 16 packed bytes (32 elements = 16 even + 16 odd)
    # Reshape: (BLOCK_KV, num_scales, 16) and broadcast
    lo_blocked = tl.reshape(lo_f32, [BLOCK_KV, num_scales, MXFP4_BLOCK_SIZE // 2])
    hi_blocked = tl.reshape(hi_f32, [BLOCK_KV, num_scales, MXFP4_BLOCK_SIZE // 2])
    scale_exp = scale_f32[:, :, None]

    lo_scaled = tl.reshape(lo_blocked * scale_exp, [BLOCK_KV, BLOCK_D // 2])
    hi_scaled = tl.reshape(hi_blocked * scale_exp, [BLOCK_KV, BLOCK_D // 2])

    return lo_scaled, hi_scaled


@triton.jit
def mla_decode_mxfp4_kernel(
    Q_ptr,          # (batch, NUM_HEADS, QK_HEAD_DIM) bf16
    KV_fp4_ptr,     # (total_kv, PACKED_QK) uint8
    KV_scale_ptr,   # (total_kv, NUM_SCALE_BLOCKS_QK) uint8
    O_ptr,          # (batch, NUM_HEADS, V_HEAD_DIM) bf16
    kv_indptr_ptr,  # (batch + 1,) int32
    stride_q_tok, stride_q_head,
    stride_o_tok, stride_o_head,
    stride_kv_tok,
    stride_sc_tok,
    sm_scale,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """MLA decode with MXFP4 KV.

    Grid: (batch_size, NUM_HEADS).
    Each program handles one query × one head.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    # Load full Q vector for this head into registers
    q_base = Q_ptr + pid_b * stride_q_tok + pid_h * stride_q_head
    # Split Q into even/odd for dot product with dequanted fp4
    # Q shape: (QK_HEAD_DIM,) → we load in BLOCK_D chunks during the score loop

    # ═══════ Online softmax + value accumulation ═══════
    m_prev = tl.full([], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([], 0.0, dtype=tl.float32)
    # Value accumulators — separate even and odd to match dequant layout
    # V has 512 dims = 256 packed pairs
    acc_even = tl.zeros([PACKED_V], dtype=tl.float32)
    acc_odd = tl.zeros([PACKED_V], dtype=tl.float32)

    num_tiles = tl.cdiv(kv_len, BLOCK_KV)
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * BLOCK_KV
        kv_offsets = tile_start + tl.arange(0, BLOCK_KV)
        mask_kv = kv_offsets < kv_len
        kv_idx = kv_start + kv_offsets

        # ── Compute scores: Q · K^T ──
        scores = tl.zeros([BLOCK_KV], dtype=tl.float32)

        # Loop over QK dimension in BLOCK_D chunks
        for d_tile in tl.static_range(QK_HEAD_DIM // BLOCK_D):
            d_start = d_tile * BLOCK_D
            d_offs = d_start + tl.arange(0, BLOCK_D)

            # Load Q chunk
            q_chunk = tl.load(q_base + d_offs).to(tl.float32)
            q_even = q_chunk[0::2]   # (BLOCK_D // 2,)
            q_odd = q_chunk[1::2]    # (BLOCK_D // 2,)

            # Dequant KV tile
            lo, hi = _dequant_fp4_block(
                KV_fp4_ptr, KV_scale_ptr,
                kv_idx, mask_kv, d_start,
                stride_kv_tok, stride_sc_tok,
                BLOCK_KV, BLOCK_D,
            )

            # Dot product: scores += Q_even · lo^T + Q_odd · hi^T
            scores += tl.sum(lo * q_even[None, :], axis=1)
            scores += tl.sum(hi * q_odd[None, :], axis=1)

        scores = scores * sm_scale
        scores = tl.where(mask_kv, scores, float("-inf"))

        # ── Online softmax ──
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        # Use exp2 with log2(e) factor for numerical precision
        log2e: tl.constexpr = 1.4426950408889634
        alpha = tl.math.exp2((m_prev - m_new) * log2e)
        p = tl.math.exp2((scores - m_new) * log2e)
        p = tl.where(mask_kv, p, 0.0)
        l_new = l_prev * alpha + tl.sum(p, axis=0)

        # Rescale previous accumulator
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ── Value accumulation (first 512 dims) ──
        for v_tile in tl.static_range(V_HEAD_DIM // BLOCK_D):
            vd_start = v_tile * BLOCK_D
            v_lo, v_hi = _dequant_fp4_block(
                KV_fp4_ptr, KV_scale_ptr,
                kv_idx, mask_kv, vd_start,
                stride_kv_tok, stride_sc_tok,
                BLOCK_KV, BLOCK_D,
            )
            # Weighted sum: p (BLOCK_KV,) × v (BLOCK_KV, BLOCK_D//2) → (BLOCK_D//2,)
            v_start_packed = vd_start // 2
            v_end_packed = v_start_packed + BLOCK_D // 2
            acc_even[v_start_packed:v_end_packed] += tl.sum(p[:, None] * v_lo, axis=0)
            acc_odd[v_start_packed:v_end_packed] += tl.sum(p[:, None] * v_hi, axis=0)

        m_prev = m_new
        l_prev = l_new

    # ═══════ Final normalize ═══════
    acc_even = acc_even / l_prev
    acc_odd = acc_odd / l_prev

    # ═══════ Interleave even/odd and store ═══════
    # Output layout: (batch, NUM_HEADS, V_HEAD_DIM) with V_HEAD_DIM contiguous
    o_base = O_ptr + pid_b * stride_o_tok + pid_h * stride_o_head

    # Store interleaved: out[2i] = acc_even[i], out[2i+1] = acc_odd[i]
    even_offsets = tl.arange(0, PACKED_V) * 2       # 0, 2, 4, ...
    odd_offsets = tl.arange(0, PACKED_V) * 2 + 1    # 1, 3, 5, ...
    tl.store(o_base + even_offsets, acc_even.to(tl.bfloat16), mask=even_offsets < V_HEAD_DIM)
    tl.store(o_base + odd_offsets, acc_odd.to(tl.bfloat16), mask=odd_offsets < V_HEAD_DIM)


def triton_mla_decode_mxfp4(
    q: torch.Tensor,
    kv_fp4: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Python wrapper for Triton MLA decode with MXFP4 KV."""
    batch_size = config["batch_size"]

    # Squeeze KV head dim: (total_kv, 1, 288) → (total_kv, 288)
    kv_fp4_2d = kv_fp4.squeeze(1) if kv_fp4.dim() == 3 else kv_fp4
    # Scale: (total_kv, 18) — already 2D from generate_input

    o = torch.empty((batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    # BLOCK_D must divide both QK_HEAD_DIM (576) and V_HEAD_DIM (512)
    # 576 = 64 × 9, 512 = 64 × 8 → BLOCK_D=64 works
    BLOCK_KV = 64
    BLOCK_D = 64

    grid = (batch_size, NUM_HEADS)
    mla_decode_mxfp4_kernel[grid](
        q, kv_fp4_2d, kv_scale, o, kv_indptr,
        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        kv_fp4_2d.stride(0),
        kv_scale.stride(0),
        SM_SCALE,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
    )
    return o


def custom_kernel_mxfp4(data: input_t) -> output_t:
    """Drop-in replacement for custom_kernel using MXFP4 KV."""
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp4, kv_scale = kv_data["mxfp4"]
    return triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config)
