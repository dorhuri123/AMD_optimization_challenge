"""
Triton MLA decode with MXFP4 KV cache — v3.0 (single-pass, redundant softmax)

Each program handles one (batch, head, v_chunk). The QK scores are recomputed
redundantly by all 8 V-chunk programs (V_DIM=512/BLOCK_D=64 = 8 chunks), but
this eliminates the HBM score scratch buffer entirely.

This is the same pattern used by sglang's production decode attention kernel.
The 8x QK recomputation is cheap (compute-bound dot products) vs the HBM
bandwidth saved by avoiding a (batch, heads, kv_seq_len) scratch buffer.

Grid: (batch_size, NUM_HEADS, V_DIM // BLOCK_D)
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

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
    """Dequant (BLOCK_KV, BLOCK_D) tile → (lo, hi) each (BLOCK_KV, BLOCK_D//2)."""
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
def mla_decode_fused_kernel(
    Q_ptr, KV_fp4_ptr, KV_scale_ptr, O_ptr,
    kv_indptr_ptr,
    stride_q_tok, stride_q_head,
    stride_kv_tok, stride_sc_tok,
    stride_o_tok, stride_o_head,
    sm_scale,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    """Single-pass fused MLA decode with redundant QK scoring.

    Grid: (batch, NUM_HEADS, V_DIM // BLOCK_D)
    Each program: one V chunk, recomputes full QK scores independently.
    """
    HALF_D: tl.constexpr = BLOCK_D // 2
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    q_base = Q_ptr + pid_b * stride_q_tok + pid_h * stride_q_head
    o_base = O_ptr + pid_b * stride_o_tok + pid_h * stride_o_head
    vd_start = pid_v * BLOCK_D

    # Online softmax state
    m_prev = tl.full([], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([], 0.0, dtype=tl.float32)
    acc_even = tl.zeros([HALF_D], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_D], dtype=tl.float32)

    num_tiles = tl.cdiv(kv_len, BLOCK_KV)
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * BLOCK_KV
        kv_offsets = tile_start + tl.arange(0, BLOCK_KV)
        mask_kv = kv_offsets < kv_len
        kv_idx = kv_start + kv_offsets

        # ── Step 1: Compute QK scores (redundant across V chunks) ──
        scores = tl.zeros([BLOCK_KV], dtype=tl.float32)
        for d_tile in tl.static_range(QK_DIM // BLOCK_D):
            d_start = d_tile * BLOCK_D
            q_even_offs = d_start + tl.arange(0, HALF_D) * 2
            q_odd_offs = d_start + tl.arange(0, HALF_D) * 2 + 1
            q_even = tl.load(q_base + q_even_offs).to(tl.float32)
            q_odd = tl.load(q_base + q_odd_offs).to(tl.float32)

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

        # ── Step 2: Online softmax update ──
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)
        p = tl.math.exp2((scores - m_new) * LOG2E)
        p = tl.where(mask_kv, p, 0.0)

        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha
        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        m_prev = m_new

        # ── Step 3: Accumulate V for this chunk ──
        v_lo, v_hi = _dequant_block(
            KV_fp4_ptr, KV_scale_ptr,
            kv_idx, mask_kv, vd_start,
            stride_kv_tok, stride_sc_tok,
            BLOCK_KV, BLOCK_D,
        )
        acc_even += tl.sum(p[:, None] * v_lo, axis=0)
        acc_odd += tl.sum(p[:, None] * v_hi, axis=0)

    # ── Final normalization ──
    acc_even = acc_even / l_prev
    acc_odd = acc_odd / l_prev

    # ── Store interleaved output ──
    even_offsets = vd_start + tl.arange(0, HALF_D) * 2
    odd_offsets = vd_start + tl.arange(0, HALF_D) * 2 + 1
    tl.store(o_base + even_offsets, acc_even.to(tl.bfloat16), mask=even_offsets < V_DIM)
    tl.store(o_base + odd_offsets, acc_odd.to(tl.bfloat16), mask=odd_offsets < V_DIM)


def triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config):
    batch_size = config["batch_size"]

    kv_fp4_2d = kv_fp4.squeeze(1) if kv_fp4.dim() == 3 else kv_fp4
    kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_scale_u8 = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale

    o = torch.empty((batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    BLOCK_KV = 64
    BLOCK_D = 64
    num_v_chunks = V_HEAD_DIM // BLOCK_D  # 8

    grid = (batch_size, NUM_HEADS, num_v_chunks)
    mla_decode_fused_kernel[grid](
        q, kv_fp4_2d, kv_scale_u8, o, kv_indptr,
        q.stride(0), q.stride(1),
        kv_fp4_2d.stride(0), kv_scale_u8.stride(0),
        o.stride(0), o.stride(1),
        SM_SCALE,
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
    )
    return o


def custom_kernel_mxfp4(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp4, kv_scale = kv_data["mxfp4"]
    return triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config)
