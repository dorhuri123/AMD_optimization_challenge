"""
Triton MXFP4 MLA decode kernel for MI355X (gfx950).

Uses hardware tl.dot_scaled for 2x throughput vs FP8 on MI355X.
Q quantized to MXFP4 via aiter's dynamic_mxfp4_quant.
K from kv_data["mxfp4"], V from kv_data["bf16"].

Key design:
  - BLOCK_M=16 (16 Q heads treated as 16 "queries" for MQA)
  - Tiles K dimension in chunks of 128 (MFMA native size for e2m1)
  - 3D grid: (batch*splits, 1, v_chunks) — all 16 heads processed together
  - Split-K for parallelism across KV tokens
  - V accumulated with regular tl.dot in bf16
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.utility.fp4_utils import dynamic_mxfp4_quant

NUM_HEADS: int = 16
QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512

# K dim tiles: 576 = 4 * 128 + 64 → 5 tiles of 128 (last padded)
NUM_K_TILES: int = 5  # ceil(576 / 128)
K_TILE_DIM: int = 128  # unpacked dims per tile
K_TILE_PACKED: int = 64  # packed bytes per tile (128 / 2)
K_TILE_SCALES: int = 4  # scale blocks per tile (128 / 32)
PACKED_QK: int = 288  # actual packed dim (576 / 2)
NUM_SCALES: int = 18  # actual scale blocks (576 / 32)

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
_q_mxfp4_cache: dict = {}


@triton.jit
def _mla_mxfp4_stage1(
    Q_packed_ptr,     # [batch*16, 288] uint8 (packed e2m1)
    Q_scale_ptr,      # [batch*16, 18] uint8 (e8m0 scales)
    K_packed_ptr,     # [total_kv, 288] uint8 (packed e2m1)
    K_scale_ptr,      # [total_kv, 18] uint8 (e8m0 scales)
    V_bf16_ptr,       # [total_kv, 576] bf16 (first V_CHUNK_D dims used per v_chunk)
    Partial_O_ptr,    # [batch, splits, 16, V_DIM] f32
    Partial_m_ptr,    # [batch, splits, 16] f32
    Partial_l_ptr,    # [batch, splits, 16] f32
    kv_indptr_ptr,    # [batch+1] i32
    stride_q_packed,  # stride per Q row in packed data
    stride_q_scale,   # stride per Q row in scale data
    stride_kv_packed, # stride per KV token in packed data
    stride_kv_scale,  # stride per KV token in scale data
    stride_v_tok,     # stride per V token (576 for bf16)
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    BLOCK_N: tl.constexpr,       # KV tokens per tile
    V_CHUNK_D: tl.constexpr,     # V dims per chunk
    NUM_KV_SPLITS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """
    Stage 1: For each (batch, split, v_chunk), compute partial attention.

    All 16 Q heads are processed together (BLOCK_M=16).
    K dimension tiled in 5 chunks of 128 dims each via dot_scaled.
    V accumulated using regular tl.dot in bf16.
    """
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_bs = tl.program_id(0)  # batch * splits + split
    pid_v = tl.program_id(2)   # v_chunk index

    pid_b = pid_bs // NUM_KV_SPLITS
    pid_s = pid_bs % NUM_KV_SPLITS

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, NUM_KV_SPLITS)
    split_kv_start = pid_s * split_size
    split_kv_end = tl.minimum(split_kv_start + split_size, kv_len)

    # Q pointers for this batch element (all 16 heads)
    q_row_base = pid_b * NUM_HEADS  # first Q row for this batch
    offs_m = tl.arange(0, NUM_HEADS)  # 0..15 (head indices)

    vd_start = pid_v * V_CHUNK_D

    # Online softmax state per head
    m_prev = tl.full([NUM_HEADS], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(split_kv_end - split_kv_start, BLOCK_N)

    for tile_idx in range(num_tiles):
        tile_start = split_kv_start + tile_idx * BLOCK_N
        kv_offsets = tile_start + tl.arange(0, BLOCK_N)
        mask_kv = kv_offsets < split_kv_end
        kv_idx = kv_start + kv_offsets  # global KV indices

        # --- Compute scores via dot_scaled: Q[16, 576] @ K[576, BLOCK_N] ---
        qk = tl.zeros([NUM_HEADS, BLOCK_N], dtype=tl.float32)

        # Tile over K dimension: 5 tiles of 128 dims each
        # Tile 0: dims 0-127 (packed 0-63, scales 0-3)
        # Tile 1: dims 128-255 (packed 64-127, scales 4-7)
        # Tile 2: dims 256-383 (packed 128-191, scales 8-11)
        # Tile 3: dims 384-511 (packed 192-255, scales 12-15)
        # Tile 4: dims 512-575 (packed 256-287, scales 16-17) + padding to 128

        for k_tile in tl.static_range(5):
            k_packed_start = k_tile * 64   # K_TILE_PACKED
            k_scale_start = k_tile * 4     # K_TILE_SCALES

            # Load Q chunk: [16, 64] packed uint8
            q_d_offs = k_packed_start + tl.arange(0, 64)  # K_TILE_PACKED
            q_chunk = tl.load(
                Q_packed_ptr + (q_row_base + offs_m[:, None]) * stride_q_packed + q_d_offs[None, :],
                mask=(q_d_offs[None, :] < 288),  # PACKED_QK
                other=0,
            )

            # Load Q scale chunk: [16, 4] uint8
            qs_offs = k_scale_start + tl.arange(0, 4)  # K_TILE_SCALES
            q_scale_chunk = tl.load(
                Q_scale_ptr + (q_row_base + offs_m[:, None]) * stride_q_scale + qs_offs[None, :],
                mask=(qs_offs[None, :] < 18),  # NUM_SCALES
                other=0,
            )

            # Load K chunk transposed: [64, BLOCK_N] packed uint8
            k_d_offs = k_packed_start + tl.arange(0, 64)  # K_TILE_PACKED
            k_chunk = tl.load(
                K_packed_ptr + kv_idx[None, :] * stride_kv_packed + k_d_offs[:, None],
                mask=mask_kv[None, :] & (k_d_offs[:, None] < 288),  # PACKED_QK
                other=0,
            )

            # Load K scale chunk: [BLOCK_N, 4] uint8
            ks_offs = k_scale_start + tl.arange(0, 4)  # K_TILE_SCALES
            k_scale_chunk = tl.load(
                K_scale_ptr + kv_idx[:, None] * stride_kv_scale + ks_offs[None, :],
                mask=mask_kv[:, None] & (ks_offs[None, :] < 18),  # NUM_SCALES
                other=0,
            )

            # Accumulate dot_scaled: [16, 64] x [64, BLOCK_N] -> [16, BLOCK_N]
            qk = tl.dot_scaled(
                q_chunk, q_scale_chunk, "e2m1",
                k_chunk, k_scale_chunk, "e2m1",
                fast_math=True, acc=qk,
            )

        # Apply masking for out-of-bounds KV tokens
        qk = tl.where(mask_kv[None, :], qk, float("-inf"))

        # --- Online softmax ---
        m_new = tl.maximum(m_prev, tl.max(qk, 1))
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)
        p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
        p = tl.where(mask_kv[None, :], p, 0.0)

        acc = acc * alpha[:, None]
        l_prev = l_prev * alpha + tl.sum(p, 1)
        m_prev = m_new

        # --- Accumulate V: p[16, BLOCK_N] @ V[BLOCK_N, V_CHUNK_D] ---
        vd_offsets = vd_start + tl.arange(0, V_CHUNK_D)
        v_tile = tl.load(
            V_bf16_ptr + kv_idx[:, None] * stride_v_tok + vd_offsets[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        # p: [16, BLOCK_N] f32 -> bf16 for tl.dot
        acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

    # Store partial outputs
    for h in tl.static_range(NUM_HEADS):
        po_base = (Partial_O_ptr + pid_b * stride_po_b + pid_s * stride_po_s
                   + h * stride_po_h + vd_start)
        tl.store(po_base + tl.arange(0, V_CHUNK_D), acc[h, :])

    # Store m and l (only from v_chunk 0)
    if pid_v == 0:
        for h in tl.static_range(NUM_HEADS):
            ml_off = pid_b * stride_ml_b + pid_s * stride_ml_s + h * stride_ml_h
            tl.store(Partial_m_ptr + ml_off, m_prev[h])
            tl.store(Partial_l_ptr + ml_off, l_prev[h])


@triton.jit
def _mla_mxfp4_reduce(
    Partial_O_ptr, Partial_m_ptr, Partial_l_ptr, O_ptr,
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    stride_o_batch, stride_o_head,
    NUM_KV_SPLITS: tl.constexpr,
    V_CHUNK_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """Reduce across splits for one (batch, head, v_chunk)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)
    vd_start = pid_v * V_CHUNK_D

    m_global = tl.full([], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NUM_KV_SPLITS):
        m_s = tl.load(Partial_m_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)
        m_global = tl.maximum(m_global, m_s)

    l_global = tl.full([], 0.0, dtype=tl.float32)
    acc = tl.zeros([V_CHUNK_D], dtype=tl.float32)
    v_offsets = tl.arange(0, V_CHUNK_D)

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


def triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_bf16, kv_indptr, config):
    """
    MXFP4 MLA decode using hardware tl.dot_scaled on MI355X.

    Args:
        q: (batch_size, 16, 576) bf16
        kv_fp4: (total_kv, 1, 288) fp4x2 (packed e2m1)
        kv_scale: (total_kv, 18) e8m0 scales
        kv_bf16: (total_kv, 1, 576) bf16 (for V)
        kv_indptr: (batch_size+1,) i32
        config: dict
    """
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    num_kv_splits = KV_SPLITS_MAP.get((batch_size, kv_seq_len), DEFAULT_KV_SPLITS)

    # Quantize Q to MXFP4
    q_2d = q.view(-1, QK_HEAD_DIM)  # (batch*16, 576)
    q_packed, q_scale = dynamic_mxfp4_quant(q_2d)
    # q_packed: (batch*16, 288) uint8
    # q_scale: (batch*16, 18) uint8

    # Flatten KV
    kv_fp4_2d = kv_fp4.view(-1, PACKED_QK)  # (total_kv, 288)
    kv_scale_2d = kv_scale  # already (total_kv, 18)
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)  # (total_kv, 576)

    BLOCK_N = 64
    V_CHUNK_D = 128
    num_v_chunks = V_HEAD_DIM // V_CHUNK_D  # 4

    bufs = _get_buffers(batch_size, num_kv_splits, q.device)

    # Stage 1: one program per (batch*split, v_chunk)
    grid1 = (batch_size * num_kv_splits, 1, num_v_chunks)
    _mla_mxfp4_stage1[grid1](
        q_packed, q_scale,
        kv_fp4_2d, kv_scale_2d,
        v_bf16_2d,
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"],
        kv_indptr,
        q_packed.stride(0), q_scale.stride(0),
        kv_fp4_2d.stride(0), kv_scale_2d.stride(0),
        v_bf16_2d.stride(0),
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        BLOCK_N=BLOCK_N, V_CHUNK_D=V_CHUNK_D,
        NUM_KV_SPLITS=num_kv_splits,
        NUM_HEADS=NUM_HEADS,
    )

    # Stage 2: reduce across splits
    grid2 = (batch_size, NUM_HEADS, num_v_chunks)
    _mla_mxfp4_reduce[grid2](
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"], bufs["output"],
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        bufs["output"].stride(0), bufs["output"].stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        V_CHUNK_D=V_CHUNK_D,
        NUM_HEADS=NUM_HEADS,
    )

    return bufs["output"]
