"""
Custom Triton MLA decode kernel with MXFP4 KV cache — v2.1

Approach: manually dequant fp4x2 → f32 in registers, standard dot products.
Even/odd accumulator pattern avoids interleaving in inner loop.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Python-side constants (not accessed from JIT kernels)
NUM_HEADS = 16
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)


@triton.jit
def _e2m1_decode(nibble):
    """Decode 4-bit E2M1 → float32."""
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
def _dequant_block(
    kv_fp4_ptr, kv_scale_ptr,
    kv_idx, mask_kv, d_start,
    stride_kv_tok, stride_sc_tok,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Load and dequant a (BLOCK_KV, BLOCK_D) KV tile.

    Returns (lo_scaled, hi_scaled) each (BLOCK_KV, BLOCK_D // 2) f32.
    """
    HALF_D: tl.constexpr = BLOCK_D // 2
    NUM_SCALES: tl.constexpr = BLOCK_D // 32

    packed_offsets = d_start // 2 + tl.arange(0, HALF_D)
    kv_packed = tl.load(
        kv_fp4_ptr + kv_idx[:, None] * stride_kv_tok + packed_offsets[None, :],
        mask=mask_kv[:, None], other=0,
    )

    lo_f32 = _e2m1_decode(kv_packed & 0x0F)
    hi_f32 = _e2m1_decode((kv_packed >> 4) & 0x0F)

    scale_start = d_start // 32
    scale_offsets = scale_start + tl.arange(0, NUM_SCALES)
    scales = tl.load(
        kv_scale_ptr + kv_idx[:, None] * stride_sc_tok + scale_offsets[None, :],
        mask=mask_kv[:, None], other=127,
    )
    scale_f32 = tl.math.exp2(scales.to(tl.float32) - 127.0)

    lo_blocked = tl.reshape(lo_f32, [BLOCK_KV, NUM_SCALES, 16])
    hi_blocked = tl.reshape(hi_f32, [BLOCK_KV, NUM_SCALES, 16])
    scale_exp = scale_f32[:, :, None]

    lo_scaled = tl.reshape(lo_blocked * scale_exp, [BLOCK_KV, HALF_D])
    hi_scaled = tl.reshape(hi_blocked * scale_exp, [BLOCK_KV, HALF_D])
    return lo_scaled, hi_scaled


@triton.jit
def mla_decode_mxfp4_kernel(
    Q_ptr, KV_fp4_ptr, KV_scale_ptr, O_ptr, kv_indptr_ptr,
    stride_q_tok, stride_q_head,
    stride_o_tok, stride_o_head,
    stride_kv_tok, stride_sc_tok,
    sm_scale,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QK_DIM: tl.constexpr,     # 576
    V_DIM: tl.constexpr,      # 512
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    q_base = Q_ptr + pid_b * stride_q_tok + pid_h * stride_q_head

    HALF_D: tl.constexpr = BLOCK_D // 2
    PACKED_V: tl.constexpr = V_DIM // 2

    m_prev = tl.full([], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([], 0.0, dtype=tl.float32)
    acc_even = tl.zeros([PACKED_V], dtype=tl.float32)
    acc_odd = tl.zeros([PACKED_V], dtype=tl.float32)

    num_tiles = tl.cdiv(kv_len, BLOCK_KV)
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * BLOCK_KV
        kv_offsets = tile_start + tl.arange(0, BLOCK_KV)
        mask_kv = kv_offsets < kv_len
        kv_idx = kv_start + kv_offsets

        # ── Score: Q · K^T ──
        scores = tl.zeros([BLOCK_KV], dtype=tl.float32)

        for d_tile in tl.static_range(QK_DIM // BLOCK_D):
            d_start = d_tile * BLOCK_D
            d_offs = d_start + tl.arange(0, BLOCK_D)
            q_chunk = tl.load(q_base + d_offs).to(tl.float32)
            q_even = q_chunk[0::2]
            q_odd = q_chunk[1::2]

            lo, hi = _dequant_block(
                KV_fp4_ptr, KV_scale_ptr,
                kv_idx, mask_kv, d_start,
                stride_kv_tok, stride_sc_tok,
                BLOCK_KV, BLOCK_D,
            )
            scores += tl.sum(lo * q_even[None, :], axis=1)
            scores += tl.sum(hi * q_odd[None, :], axis=1)

        scores = scores * sm_scale
        scores = tl.where(mask_kv, scores, float("-inf"))

        # ── Online softmax ──
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        log2e: tl.constexpr = 1.4426950408889634
        alpha = tl.math.exp2((m_prev - m_new) * log2e)
        p = tl.math.exp2((scores - m_new) * log2e)
        p = tl.where(mask_kv, p, 0.0)
        l_new = l_prev * alpha + tl.sum(p, axis=0)

        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ── Value accumulation (first V_DIM dims) ──
        for v_tile in tl.static_range(V_DIM // BLOCK_D):
            vd_start = v_tile * BLOCK_D
            v_lo, v_hi = _dequant_block(
                KV_fp4_ptr, KV_scale_ptr,
                kv_idx, mask_kv, vd_start,
                stride_kv_tok, stride_sc_tok,
                BLOCK_KV, BLOCK_D,
            )
            v_base = vd_start // 2
            # p: (BLOCK_KV,), v_lo/v_hi: (BLOCK_KV, HALF_D)
            weighted_lo = tl.sum(p[:, None] * v_lo, axis=0)  # (HALF_D,)
            weighted_hi = tl.sum(p[:, None] * v_hi, axis=0)
            acc_even[v_base:v_base + HALF_D] += weighted_lo
            acc_odd[v_base:v_base + HALF_D] += weighted_hi

        m_prev = m_new
        l_prev = l_new

    # ── Normalize ──
    acc_even = acc_even / l_prev
    acc_odd = acc_odd / l_prev

    # ── Store interleaved output ──
    o_base = O_ptr + pid_b * stride_o_tok + pid_h * stride_o_head
    even_offsets = tl.arange(0, PACKED_V) * 2
    odd_offsets = tl.arange(0, PACKED_V) * 2 + 1
    tl.store(o_base + even_offsets, acc_even.to(tl.bfloat16), mask=even_offsets < V_DIM)
    tl.store(o_base + odd_offsets, acc_odd.to(tl.bfloat16), mask=odd_offsets < V_DIM)


def triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config):
    batch_size = config["batch_size"]

    kv_fp4_2d = kv_fp4.squeeze(1) if kv_fp4.dim() == 3 else kv_fp4
    kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_scale_u8 = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale

    o = torch.empty((batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    BLOCK_KV = 64
    BLOCK_D = 64  # must divide both 576 and 512

    grid = (batch_size, NUM_HEADS)
    mla_decode_mxfp4_kernel[grid](
        q, kv_fp4_2d, kv_scale_u8, o, kv_indptr,
        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        kv_fp4_2d.stride(0), kv_scale_u8.stride(0),
        SM_SCALE,
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
    )
    return o


def custom_kernel_mxfp4(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp4, kv_scale = kv_data["mxfp4"]
    return triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config)
