"""
Triton MLA decode with MXFP4 KV cache — v3.1 (single-pass + split-K)

Two-stage flash-decoding:
  Stage 1: Each program handles one (batch, head, kv_split, v_chunk).
           Computes partial online-softmax output + log-sum-exp.
  Stage 2: Reduce across kv_splits for each (batch, head, v_chunk).

Grid stage 1: (batch * NUM_KV_SPLITS, NUM_HEADS, V_DIM // BLOCK_D)
Grid stage 2: (batch, NUM_HEADS, V_DIM // BLOCK_D)
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
def mla_stage1_kernel(
    Q_ptr, KV_fp4_ptr, KV_scale_ptr,
    Partial_O_ptr,    # (batch, NUM_KV_SPLITS, NUM_HEADS, num_v_chunks, BLOCK_D) f32
    Partial_lse_ptr,  # (batch, NUM_KV_SPLITS, NUM_HEADS) f32  — shared across V chunks
    kv_indptr_ptr,
    stride_q_tok, stride_q_head,
    stride_kv_tok, stride_sc_tok,
    stride_po_b, stride_po_s, stride_po_h, stride_po_v,
    stride_lse_b, stride_lse_s, stride_lse_h,
    sm_scale,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    """Stage 1: partial attention for one KV split.

    Grid: (batch * NUM_KV_SPLITS, NUM_HEADS, V_DIM // BLOCK_D)
    """
    HALF_D: tl.constexpr = BLOCK_D // 2
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)

    pid_b = pid_bs // NUM_KV_SPLITS
    pid_s = pid_bs % NUM_KV_SPLITS

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    # This split's KV range
    split_size = tl.cdiv(kv_len, NUM_KV_SPLITS)
    split_kv_start = pid_s * split_size
    split_kv_end = tl.minimum(split_kv_start + split_size, kv_len)
    split_len = split_kv_end - split_kv_start

    q_base = Q_ptr + pid_b * stride_q_tok + pid_h * stride_q_head
    vd_start = pid_v * BLOCK_D

    m_prev = tl.full([], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([], 0.0, dtype=tl.float32)
    acc_even = tl.zeros([HALF_D], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_D], dtype=tl.float32)

    num_tiles = tl.cdiv(split_len, BLOCK_KV)
    for tile_idx in range(num_tiles):
        tile_start = split_kv_start + tile_idx * BLOCK_KV
        kv_offsets = tile_start + tl.arange(0, BLOCK_KV)
        mask_kv = kv_offsets < split_kv_end
        kv_idx = kv_start + kv_offsets

        # QK scores (redundant across V chunks)
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

        # Online softmax
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)
        p = tl.math.exp2((scores - m_new) * LOG2E)
        p = tl.where(mask_kv, p, 0.0)

        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha
        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        m_prev = m_new

        # V accumulation
        v_lo, v_hi = _dequant_block(
            KV_fp4_ptr, KV_scale_ptr,
            kv_idx, mask_kv, vd_start,
            stride_kv_tok, stride_sc_tok,
            BLOCK_KV, BLOCK_D,
        )
        acc_even += tl.sum(p[:, None] * v_lo, axis=0)
        acc_odd += tl.sum(p[:, None] * v_hi, axis=0)

    # Store partial output (interleaved even/odd)
    po_base = (Partial_O_ptr
               + pid_b * stride_po_b
               + pid_s * stride_po_s
               + pid_h * stride_po_h
               + pid_v * stride_po_v)
    even_offsets = tl.arange(0, HALF_D) * 2
    odd_offsets = tl.arange(0, HALF_D) * 2 + 1
    tl.store(po_base + even_offsets, acc_even)
    tl.store(po_base + odd_offsets, acc_odd)

    # Store LSE (only from v_chunk 0 to avoid redundant writes)
    if pid_v == 0:
        # lse = m + log(l) in natural log
        lse_val = m_prev + tl.log(l_prev + 1e-10)
        lse_base = Partial_lse_ptr + pid_b * stride_lse_b + pid_s * stride_lse_s + pid_h * stride_lse_h
        tl.store(lse_base, lse_val)


@triton.jit
def mla_reduce_kernel(
    Partial_O_ptr, Partial_lse_ptr, O_ptr,
    stride_po_b, stride_po_s, stride_po_h, stride_po_v,
    stride_lse_b, stride_lse_s, stride_lse_h,
    stride_o_tok, stride_o_head,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    V_DIM: tl.constexpr,
):
    """Stage 2: Reduce partial outputs across KV splits.

    Grid: (batch, NUM_HEADS, V_DIM // BLOCK_D)
    """
    HALF_D: tl.constexpr = BLOCK_D // 2

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)
    vd_start = pid_v * BLOCK_D

    # Find global max LSE across splits
    m_global = tl.full([], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NUM_KV_SPLITS):
        lse_s = tl.load(Partial_lse_ptr + pid_b * stride_lse_b + s * stride_lse_s + pid_h * stride_lse_h)
        m_global = tl.maximum(m_global, lse_s)

    # Weighted reduction
    acc_even = tl.zeros([HALF_D], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_D], dtype=tl.float32)
    l_global = tl.full([], 0.0, dtype=tl.float32)

    for s in tl.static_range(NUM_KV_SPLITS):
        lse_s = tl.load(Partial_lse_ptr + pid_b * stride_lse_b + s * stride_lse_s + pid_h * stride_lse_h)
        weight = tl.math.exp(lse_s - m_global)
        l_global += weight

        po_base = (Partial_O_ptr
                   + pid_b * stride_po_b
                   + s * stride_po_s
                   + pid_h * stride_po_h
                   + pid_v * stride_po_v)
        even_offsets = tl.arange(0, HALF_D) * 2
        odd_offsets = tl.arange(0, HALF_D) * 2 + 1
        p_even = tl.load(po_base + even_offsets)
        p_odd = tl.load(po_base + odd_offsets)
        acc_even += weight * p_even
        acc_odd += weight * p_odd

    acc_even = acc_even / l_global
    acc_odd = acc_odd / l_global

    # Store final output
    o_base = O_ptr + pid_b * stride_o_tok + pid_h * stride_o_head
    even_offsets = vd_start + tl.arange(0, HALF_D) * 2
    odd_offsets = vd_start + tl.arange(0, HALF_D) * 2 + 1
    tl.store(o_base + even_offsets, acc_even.to(tl.bfloat16), mask=even_offsets < V_DIM)
    tl.store(o_base + odd_offsets, acc_odd.to(tl.bfloat16), mask=odd_offsets < V_DIM)


def triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config):
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    kv_fp4_2d = kv_fp4.squeeze(1) if kv_fp4.dim() == 3 else kv_fp4
    kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_scale_u8 = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale

    BLOCK_KV = 64
    BLOCK_D = 64
    num_v_chunks = V_HEAD_DIM // BLOCK_D  # 8

    # Choose NUM_KV_SPLITS based on sequence length
    if kv_seq_len <= 1024:
        NUM_KV_SPLITS = 4
    elif kv_seq_len <= 4096:
        NUM_KV_SPLITS = 8
    else:
        NUM_KV_SPLITS = 16

    # Partial buffers
    partial_o = torch.empty(
        (batch_size, NUM_KV_SPLITS, NUM_HEADS, num_v_chunks, BLOCK_D),
        dtype=torch.float32, device=q.device,
    )
    partial_lse = torch.empty(
        (batch_size, NUM_KV_SPLITS, NUM_HEADS),
        dtype=torch.float32, device=q.device,
    )
    o = torch.empty((batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    # Stage 1
    grid1 = (batch_size * NUM_KV_SPLITS, NUM_HEADS, num_v_chunks)
    mla_stage1_kernel[grid1](
        q, kv_fp4_2d, kv_scale_u8,
        partial_o, partial_lse, kv_indptr,
        q.stride(0), q.stride(1),
        kv_fp4_2d.stride(0), kv_scale_u8.stride(0),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        SM_SCALE,
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
    )

    # Stage 2: reduce
    grid2 = (batch_size, NUM_HEADS, num_v_chunks)
    mla_reduce_kernel[grid2](
        partial_o, partial_lse, o,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        o.stride(0), o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_D=BLOCK_D, V_DIM=V_HEAD_DIM,
    )
    return o


def custom_kernel_mxfp4(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp4, kv_scale = kv_data["mxfp4"]
    return triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_indptr, config)
