"""
Triton FP8 MLA decode kernel — pure Triton, zero AITER dependency.

Three-stage approach:
  Stage 1: 3D grid (batch*split, head, v_chunk) — each program computes scores
           over full 576 QK dims and accumulates one V_BLOCK_D chunk of V.
  Stage 2: Reduce across splits using log-sum-exp.

FP8 KV cache: (total_kv, 1, 576) fp8_e4m3 with a single scalar scale.
Q: (batch_size, 16, 576) bfloat16.
Output: (batch_size, 16, 512) bfloat16.

Note: Each v_chunk recomputes QK scores independently. This is redundant compute
but keeps register pressure low (critical for occupancy on CDNA3).
For decode (memory-bound), the extra compute is largely hidden by memory latency.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS: int = 16
QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512
SM_SCALE: float = 1.0 / (QK_HEAD_DIM ** 0.5)

# Split-K config per (batch_size, kv_seq_len)
KV_SPLITS_MAP = {
    (4, 1024): 4,
    (4, 8192): 16,
    (32, 1024): 4,
    (32, 8192): 16,
    (64, 1024): 4,
    (64, 8192): 16,
    (256, 1024): 4,
    (256, 8192): 8,
}
DEFAULT_KV_SPLITS = 8

_buf_cache: dict = {}


@triton.jit
def _mla_stage1_fp8(
    Q_ptr,
    KV_ptr,
    kv_scale_ptr,
    Partial_O_ptr,    # (batch, splits, heads, V_DIM) f32
    Partial_m_ptr,    # (batch, splits, heads) f32
    Partial_l_ptr,    # (batch, splits, heads) f32
    kv_indptr_ptr,
    stride_q_batch, stride_q_head,
    stride_kv_tok,
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    sm_scale,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    V_BLOCK_D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    """
    Stage 1: For each (batch, split, head, v_chunk), compute partial attention.

    QK scores computed over all 576 dims (redundantly per v_chunk).
    V accumulated over V_BLOCK_D dims only.
    Only v_chunk 0 stores m and l.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    QK_DIM: tl.constexpr = 576

    pid_bs = tl.program_id(0)  # batch * splits + split
    pid_h = tl.program_id(1)   # head
    pid_v = tl.program_id(2)   # v_chunk

    pid_b = pid_bs // NUM_KV_SPLITS
    pid_s = pid_bs % NUM_KV_SPLITS

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, NUM_KV_SPLITS)
    split_kv_start = pid_s * split_size
    split_kv_end = tl.minimum(split_kv_start + split_size, kv_len)

    kv_scale = tl.load(kv_scale_ptr)
    q_base = Q_ptr + pid_b * stride_q_batch + pid_h * stride_q_head
    vd_start = pid_v * V_BLOCK_D

    m_prev = tl.full([], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([], 0.0, dtype=tl.float32)
    acc = tl.zeros([V_BLOCK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(split_kv_end - split_kv_start, BLOCK_KV)

    for tile_idx in range(num_tiles):
        tile_start = split_kv_start + tile_idx * BLOCK_KV
        kv_offsets = tile_start + tl.arange(0, BLOCK_KV)
        mask_kv = kv_offsets < split_kv_end
        kv_idx = kv_start + kv_offsets

        # --- Compute scores: Q[576] @ K[BLOCK_KV, 576]^T ---
        scores = tl.zeros([BLOCK_KV], dtype=tl.float32)

        for d_tile in tl.static_range(QK_DIM // BLOCK_D):  # 576/64 = 9
            d_start = d_tile * BLOCK_D
            d_offsets = d_start + tl.arange(0, BLOCK_D)

            q_tile = tl.load(q_base + d_offsets).to(tl.float32)
            k_tile = tl.load(
                KV_ptr + kv_idx[:, None] * stride_kv_tok + d_offsets[None, :],
                mask=mask_kv[:, None],
                other=0.0,
            ).to(tl.float32)

            scores += tl.sum(k_tile * q_tile[None, :], axis=1)

        # FP8 dequant for K + attention scale
        scores = scores * (kv_scale * sm_scale)
        scores = tl.where(mask_kv, scores, float("-inf"))

        # --- Online softmax ---
        m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)
        p = tl.math.exp2((scores - m_new) * LOG2E)
        p = tl.where(mask_kv, p, 0.0)

        acc = acc * alpha
        l_prev = l_prev * alpha + tl.sum(p, axis=0)
        m_prev = m_new

        # --- Accumulate V chunk: p @ V[BLOCK_KV, V_BLOCK_D] ---
        vd_offsets = vd_start + tl.arange(0, V_BLOCK_D)
        v_tile = tl.load(
            KV_ptr + kv_idx[:, None] * stride_kv_tok + vd_offsets[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(p[:, None] * v_tile, axis=0)

    # Apply kv_scale to V (deferred dequant — valid since kv_scale is scalar)
    acc = acc * kv_scale

    # Store partial V output
    po_base = (Partial_O_ptr + pid_b * stride_po_b + pid_s * stride_po_s
               + pid_h * stride_po_h + vd_start)
    tl.store(po_base + tl.arange(0, V_BLOCK_D), acc)

    # Store m and l (only from v_chunk 0 — same across all v_chunks)
    if pid_v == 0:
        ml_off = pid_b * stride_ml_b + pid_s * stride_ml_s + pid_h * stride_ml_h
        tl.store(Partial_m_ptr + ml_off, m_prev)
        tl.store(Partial_l_ptr + ml_off, l_prev)


@triton.jit
def _mla_reduce_fp8(
    Partial_O_ptr,
    Partial_m_ptr,
    Partial_l_ptr,
    O_ptr,
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    stride_o_batch, stride_o_head,
    NUM_KV_SPLITS: tl.constexpr,
    V_BLOCK_D: tl.constexpr,
):
    """
    Stage 2: Reduce partial results across splits for one (batch, head, v_chunk).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)
    vd_start = pid_v * V_BLOCK_D

    # Global max across splits
    m_global = tl.full([], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NUM_KV_SPLITS):
        m_s = tl.load(Partial_m_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)
        m_global = tl.maximum(m_global, m_s)

    # Weighted reduction
    l_global = tl.full([], 0.0, dtype=tl.float32)
    acc = tl.zeros([V_BLOCK_D], dtype=tl.float32)
    v_offsets = tl.arange(0, V_BLOCK_D)

    for s in tl.static_range(NUM_KV_SPLITS):
        m_s = tl.load(Partial_m_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)
        l_s = tl.load(Partial_l_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)

        rescale = tl.math.exp(m_s - m_global)
        l_global += l_s * rescale

        po_base = (Partial_O_ptr + pid_b * stride_po_b + s * stride_po_s
                   + pid_h * stride_po_h + vd_start)
        partial = tl.load(po_base + v_offsets)
        acc += rescale * partial

    acc = acc / (l_global + 1e-10)

    o_base = O_ptr + pid_b * stride_o_batch + pid_h * stride_o_head + vd_start
    tl.store(o_base + v_offsets, acc.to(tl.bfloat16))


def _get_buffers(batch_size, num_kv_splits, device):
    """Get or allocate cached buffers."""
    key = (batch_size, num_kv_splits)
    if key not in _buf_cache:
        _buf_cache[key] = {
            "partial_o": torch.empty(
                (batch_size, num_kv_splits, NUM_HEADS, V_HEAD_DIM),
                dtype=torch.float32, device=device,
            ),
            "partial_m": torch.empty(
                (batch_size, num_kv_splits, NUM_HEADS),
                dtype=torch.float32, device=device,
            ),
            "partial_l": torch.empty(
                (batch_size, num_kv_splits, NUM_HEADS),
                dtype=torch.float32, device=device,
            ),
            "output": torch.empty(
                (batch_size, NUM_HEADS, V_HEAD_DIM),
                dtype=torch.bfloat16, device=device,
            ),
        }
    return _buf_cache[key]


def triton_mla_decode_fp8(q, kv_fp8, kv_scale, kv_indptr, config):
    """Pure Triton FP8 MLA decode — flash-decoding with split-K."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    num_kv_splits = KV_SPLITS_MAP.get((batch_size, kv_seq_len), DEFAULT_KV_SPLITS)

    kv_2d = kv_fp8.view(-1, QK_HEAD_DIM)

    BLOCK_KV = 64
    BLOCK_D = 64
    V_BLOCK_D = 64
    num_v_chunks = V_HEAD_DIM // V_BLOCK_D  # 8

    bufs = _get_buffers(batch_size, num_kv_splits, q.device)
    partial_o = bufs["partial_o"]
    partial_m = bufs["partial_m"]
    partial_l = bufs["partial_l"]
    output = bufs["output"]

    # Stage 1: 3D grid — each program handles one (batch*split, head, v_chunk)
    grid1 = (batch_size * num_kv_splits, NUM_HEADS, num_v_chunks)
    _mla_stage1_fp8[grid1](
        q, kv_2d, kv_scale,
        partial_o, partial_m, partial_l,
        kv_indptr,
        q.stride(0), q.stride(1),
        kv_2d.stride(0),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        SM_SCALE,
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
        V_BLOCK_D=V_BLOCK_D,
        NUM_KV_SPLITS=num_kv_splits,
    )

    # Stage 2: reduce across splits
    grid2 = (batch_size, NUM_HEADS, num_v_chunks)
    _mla_reduce_fp8[grid2](
        partial_o, partial_m, partial_l, output,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        output.stride(0), output.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        V_BLOCK_D=V_BLOCK_D,
    )

    return output


def custom_kernel(data: input_t) -> output_t:
    """Drop-in replacement for submission.py custom_kernel."""
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp8, kv_scale = kv_data["fp8"]
    return triton_mla_decode_fp8(q, kv_fp8, kv_scale, kv_indptr, config)
