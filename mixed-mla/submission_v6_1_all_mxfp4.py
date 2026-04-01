"""
Optimized MLA decode submission -- v6.0 Hybrid (Fast MXFP4 + AITER)

KEY FIX from v4/v5: Eliminates 4x redundant QK computation.
v4/v5 used a 3D grid (batch*splits, 1, v_chunks=4) where each v_chunk
program recomputed ALL QK attention scores independently, then only
accumulated 128 of the 512 V dims. This wastes 3/4 of QK compute.

v6 uses a 2D grid (batch*splits, 1) where each program:
  1. Computes QK scores ONCE via dot_scaled (5 tiles of 128 K dims)
  2. Applies online softmax
  3. Tiles V accumulation: 4 iterations of p @ V_chunk[BLOCK_N, 128]
  4. Stores all 512 V dims + softmax state

This eliminates the 4x QK redundancy, making large configs competitive
with AITER instead of 2-10x slower.

Routing:
  MXFP4: (4,1024), (4,8192), (32,1024), (64,1024)
  AITER: (32,8192), (64,8192), (256,1024), (256,8192)
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import dynamic_mxfp4_quant

# ===============================================================
# CONSTANTS
# ===============================================================

NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = 576  # KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM: int = 512   # KV_LORA_RANK
SM_SCALE: float = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE: int = 1
FP8_DTYPE = aiter_dtypes.fp8

# K dim layout: 576 = 4*128 + 64 -> 5 tiles of 128 (last padded)
PACKED_QK: int = 288       # 576 / 2 packed bytes
NUM_SCALES: int = 18       # 576 / 32 scale blocks

# Routing: ALL configs use MXFP4 (testing if fixed kernel beats AITER everywhere)
MXFP4_CONFIGS = {(4, 1024), (4, 8192), (32, 1024), (32, 8192),
                 (64, 1024), (64, 8192), (256, 1024), (256, 8192)}

# MXFP4 split-K tuning
MXFP4_KV_SPLITS_MAP = {
    (4, 1024): 4,
    (4, 8192): 16,
    (32, 1024): 4,
    (32, 8192): 16,
    (64, 1024): 4,
    (64, 8192): 16,
    (256, 1024): 4,
    (256, 8192): 8,
}
MXFP4_DEFAULT_KV_SPLITS = 8

# AITER split-K tuning
AITER_KV_SPLITS_MAP = {
    (32, 8192): 48,
    (64, 8192): 24,
    (256, 1024): 16,
    (256, 8192): 24,
}
AITER_DEFAULT_KV_SPLITS = 16

# Caches
_mxfp4_buf_cache: dict = {}
_meta_cache: dict = {}
_alloc_cache: dict = {}


# ===============================================================
# MXFP4 TRITON KERNEL -- STAGE 1 (FAST: no redundant QK)
# ===============================================================

@triton.jit
def _mla_mxfp4_stage1_fast(
    Q_packed_ptr,     # [batch*16, 288] uint8 (packed e2m1)
    Q_scale_ptr,      # [batch*16, 18] uint8 (e8m0 scales)
    K_packed_ptr,     # [total_kv, 288] uint8 (packed e2m1)
    K_scale_ptr,      # [total_kv, 18] uint8 (e8m0 scales)
    V_bf16_ptr,       # [total_kv, 576] bf16 (first 512 dims used)
    Partial_O_ptr,    # [batch, splits, 16, V_HEAD_DIM] f32
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
    sm_scale,
    BLOCK_N: tl.constexpr,       # KV tokens per tile (64)
    NUM_KV_SPLITS: tl.constexpr,
    NUM_HEADS: tl.constexpr,     # 16
    V_CHUNK_D: tl.constexpr,     # 128 — V tile width
    NUM_V_CHUNKS: tl.constexpr,  # 4 — V_HEAD_DIM / V_CHUNK_D
):
    """
    Stage 1: For each (batch, split), compute partial attention over ALL 512 V dims.
    QK scores computed ONCE per KV tile, then V accumulated in 4 chunks of 128.

    2D grid: (batch_size * num_kv_splits, 1)
    Each program handles all 16 heads and all 512 V dims.
    """
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_bs = tl.program_id(0)  # batch * splits + split

    pid_b = pid_bs // NUM_KV_SPLITS
    pid_s = pid_bs % NUM_KV_SPLITS

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, NUM_KV_SPLITS)
    split_kv_start = pid_s * split_size
    split_kv_end = tl.minimum(split_kv_start + split_size, kv_len)

    # Q pointers for this batch element (all 16 heads)
    q_row_base = pid_b * NUM_HEADS
    offs_m = tl.arange(0, NUM_HEADS)  # 0..15

    # Online softmax state per head
    m_prev = tl.full([NUM_HEADS], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([NUM_HEADS], dtype=tl.float32)

    # 4 separate V accumulators: each [NUM_HEADS, V_CHUNK_D] = [16, 128]
    acc0 = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)
    acc1 = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)
    acc2 = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)
    acc3 = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(split_kv_end - split_kv_start, BLOCK_N)

    for tile_idx in range(num_tiles):
        tile_start = split_kv_start + tile_idx * BLOCK_N
        kv_offsets = tile_start + tl.arange(0, BLOCK_N)
        mask_kv = kv_offsets < split_kv_end
        kv_idx = kv_start + kv_offsets  # global KV indices

        # ---- Compute QK scores via dot_scaled (ONCE per tile) ----
        qk = tl.zeros([NUM_HEADS, BLOCK_N], dtype=tl.float32)

        for k_tile in tl.static_range(5):
            k_packed_start = k_tile * 64   # 128/2 packed bytes per tile
            k_scale_start = k_tile * 4     # 128/32 scale blocks per tile

            # Q chunk: [16, 64] packed uint8
            q_d_offs = k_packed_start + tl.arange(0, 64)
            q_chunk = tl.load(
                Q_packed_ptr + (q_row_base + offs_m[:, None]) * stride_q_packed + q_d_offs[None, :],
                mask=(q_d_offs[None, :] < 288),
                other=0,
            )

            # Q scale chunk: [16, 4] uint8 (e8m0)
            qs_offs = k_scale_start + tl.arange(0, 4)
            q_scale_chunk = tl.load(
                Q_scale_ptr + (q_row_base + offs_m[:, None]) * stride_q_scale + qs_offs[None, :],
                mask=(qs_offs[None, :] < 18),
                other=0,
            )

            # K chunk TRANSPOSED: [64, BLOCK_N] packed uint8
            k_d_offs = k_packed_start + tl.arange(0, 64)
            k_chunk = tl.load(
                K_packed_ptr + kv_idx[None, :] * stride_kv_packed + k_d_offs[:, None],
                mask=mask_kv[None, :] & (k_d_offs[:, None] < 288),
                other=0,
            )

            # K scale chunk: [BLOCK_N, 4] uint8 (e8m0)
            ks_offs = k_scale_start + tl.arange(0, 4)
            k_scale_chunk = tl.load(
                K_scale_ptr + kv_idx[:, None] * stride_kv_scale + ks_offs[None, :],
                mask=mask_kv[:, None] & (ks_offs[None, :] < 18),
                other=0,
            )

            # dot_scaled: [16, 64] x [64, BLOCK_N] -> accumulate [16, BLOCK_N]
            qk = tl.dot_scaled(
                q_chunk, q_scale_chunk, "e2m1",
                k_chunk, k_scale_chunk, "e2m1",
                fast_math=True, acc=qk,
            )

        # Scale and mask
        qk *= sm_scale
        qk = tl.where(mask_kv[None, :], qk, float("-inf"))

        # ---- Online softmax ----
        m_new = tl.maximum(m_prev, tl.max(qk, 1))
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)
        p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
        p = tl.where(mask_kv[None, :], p, 0.0)

        # Rescale all 4 accumulators
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        acc2 = acc2 * alpha[:, None]
        acc3 = acc3 * alpha[:, None]
        l_prev = l_prev * alpha + tl.sum(p, 1)
        m_prev = m_new

        # Cast p to bf16 for V matmul
        p_bf16 = p.to(tl.bfloat16)

        # ---- Accumulate V in 4 chunks of 128 dims ----
        # V chunk 0: dims 0-127
        v_offs_0 = tl.arange(0, V_CHUNK_D)
        v0 = tl.load(
            V_bf16_ptr + kv_idx[:, None] * stride_v_tok + v_offs_0[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        acc0 += tl.dot(p_bf16, v0, out_dtype=tl.float32)

        # V chunk 1: dims 128-255
        v_offs_1 = V_CHUNK_D + tl.arange(0, V_CHUNK_D)
        v1 = tl.load(
            V_bf16_ptr + kv_idx[:, None] * stride_v_tok + v_offs_1[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        acc1 += tl.dot(p_bf16, v1, out_dtype=tl.float32)

        # V chunk 2: dims 256-383
        v_offs_2 = 2 * V_CHUNK_D + tl.arange(0, V_CHUNK_D)
        v2 = tl.load(
            V_bf16_ptr + kv_idx[:, None] * stride_v_tok + v_offs_2[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        acc2 += tl.dot(p_bf16, v2, out_dtype=tl.float32)

        # V chunk 3: dims 384-511
        v_offs_3 = 3 * V_CHUNK_D + tl.arange(0, V_CHUNK_D)
        v3 = tl.load(
            V_bf16_ptr + kv_idx[:, None] * stride_v_tok + v_offs_3[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        acc3 += tl.dot(p_bf16, v3, out_dtype=tl.float32)

    # ---- Store partial outputs: all 512 V dims ----
    head_offs = tl.arange(0, NUM_HEADS)
    v_offs = tl.arange(0, V_CHUNK_D)

    po_base = Partial_O_ptr + pid_b * stride_po_b + pid_s * stride_po_s

    # Store acc0: V dims 0-127
    tl.store(
        po_base + head_offs[:, None] * stride_po_h + v_offs[None, :],
        acc0,
    )
    # Store acc1: V dims 128-255
    tl.store(
        po_base + head_offs[:, None] * stride_po_h + (V_CHUNK_D + v_offs[None, :]),
        acc1,
    )
    # Store acc2: V dims 256-383
    tl.store(
        po_base + head_offs[:, None] * stride_po_h + (2 * V_CHUNK_D + v_offs[None, :]),
        acc2,
    )
    # Store acc3: V dims 384-511
    tl.store(
        po_base + head_offs[:, None] * stride_po_h + (3 * V_CHUNK_D + v_offs[None, :]),
        acc3,
    )

    # Store m and l
    ml_base = pid_b * stride_ml_b + pid_s * stride_ml_s
    tl.store(
        Partial_m_ptr + ml_base + head_offs * stride_ml_h,
        m_prev,
    )
    tl.store(
        Partial_l_ptr + ml_base + head_offs * stride_ml_h,
        l_prev,
    )


# ===============================================================
# MXFP4 TRITON KERNEL -- STAGE 2: REDUCE (all 512 V dims at once)
# ===============================================================

@triton.jit
def _mla_mxfp4_reduce_fast(
    Partial_O_ptr, Partial_m_ptr, Partial_l_ptr, O_ptr,
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    stride_o_batch, stride_o_head,
    NUM_KV_SPLITS: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """Reduce across splits for one (batch, head). Processes all 512 V dims."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Find global max across splits
    m_global = tl.full([], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NUM_KV_SPLITS):
        m_s = tl.load(Partial_m_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)
        m_global = tl.maximum(m_global, m_s)

    # Rescale and accumulate all 512 V dims
    l_global = tl.full([], 0.0, dtype=tl.float32)
    v_offsets = tl.arange(0, V_HEAD_DIM)
    acc = tl.zeros([V_HEAD_DIM], dtype=tl.float32)

    for s in tl.static_range(NUM_KV_SPLITS):
        m_s = tl.load(Partial_m_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)
        l_s = tl.load(Partial_l_ptr + pid_b * stride_ml_b + s * stride_ml_s + pid_h * stride_ml_h)
        rescale = tl.math.exp(m_s - m_global)
        l_global += l_s * rescale

        po_base = (Partial_O_ptr + pid_b * stride_po_b + s * stride_po_s
                   + pid_h * stride_po_h)
        partial = tl.load(po_base + v_offsets)
        acc += rescale * partial

    acc = acc / (l_global + 1e-10)
    o_base = O_ptr + pid_b * stride_o_batch + pid_h * stride_o_head
    tl.store(o_base + v_offsets, acc.to(tl.bfloat16))


# ===============================================================
# MXFP4 BUFFER CACHE
# ===============================================================

def _mxfp4_get_buffers(batch_size, num_kv_splits, device):
    key = (batch_size, num_kv_splits)
    if key not in _mxfp4_buf_cache:
        _mxfp4_buf_cache[key] = {
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
    return _mxfp4_buf_cache[key]


# ===============================================================
# MXFP4 DECODE PATH (FAST)
# ===============================================================

def _mxfp4_path(q, kv_data, kv_indptr, config):
    """
    Fast MXFP4 MLA decode: single QK computation per program.
    Q quantized to MXFP4, K via dot_scaled, V via bf16 tl.dot.
    2D grid eliminates 4x redundant QK computation from v4/v5.
    """
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    num_kv_splits = MXFP4_KV_SPLITS_MAP.get(
        (batch_size, kv_seq_len), MXFP4_DEFAULT_KV_SPLITS
    )

    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    # Quantize Q to MXFP4
    q_2d = q.view(-1, QK_HEAD_DIM)  # (batch*16, 576)
    q_packed_raw, q_scale_raw = dynamic_mxfp4_quant(q_2d)
    q_packed = q_packed_raw.view(torch.uint8)
    q_scale = q_scale_raw.view(torch.uint8)

    # Flatten KV tensors, cast to uint8 for Triton
    kv_fp4_2d = kv_fp4.reshape(-1, PACKED_QK).view(torch.uint8)  # (total_kv, 288)
    kv_scale_2d = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)  # (total_kv, 576)

    BLOCK_N = 64
    V_CHUNK_D = 128
    NUM_V_CHUNKS = V_HEAD_DIM // V_CHUNK_D  # 4

    bufs = _mxfp4_get_buffers(batch_size, num_kv_splits, q.device)

    # Stage 1: 2D grid — one program per (batch*split)
    # Each program computes QK ONCE and accumulates ALL 512 V dims
    grid1 = (batch_size * num_kv_splits, 1)
    _mla_mxfp4_stage1_fast[grid1](
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
        SM_SCALE,
        BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=num_kv_splits,
        NUM_HEADS=NUM_HEADS,
        V_CHUNK_D=V_CHUNK_D,
        NUM_V_CHUNKS=NUM_V_CHUNKS,
    )

    # Stage 2: reduce across splits — 2D grid (batch, heads)
    # Each program reduces all 512 V dims for one (batch, head)
    grid2 = (batch_size, NUM_HEADS)
    _mla_mxfp4_reduce_fast[grid2](
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"], bufs["output"],
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        bufs["output"].stride(0), bufs["output"].stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        V_HEAD_DIM=V_HEAD_DIM,
        NUM_HEADS=NUM_HEADS,
    )

    return bufs["output"]


# ===============================================================
# FP8 QUANTIZATION (for AITER path)
# ===============================================================

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ===============================================================
# AITER CACHED METADATA
# ===============================================================

def _get_cached_meta(bs, nq, nkv, q_dtype, kv_dtype, qo_indptr, kv_indptr, num_kv_splits):
    key = (bs, num_kv_splits, q_dtype, kv_dtype)
    if key not in _meta_cache:
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        total_kv = int(kv_indptr[-1].item())
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")

        info = get_mla_metadata_info_v1(
            bs, 1, nq, q_dtype, kv_dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=q_dtype, dtype_kv=kv_dtype,
        )
        _meta_cache[key] = {
            "work_meta_data": wm, "work_indptr": wi, "work_info_set": wis,
            "reduce_indptr": ri, "reduce_final_map": rfm, "reduce_partial_map": rpm,
            "kv_indices": kv_indices, "kv_last_page_len": kv_last_page_len,
        }
    return _meta_cache[key]


def _get_cached_allocs(bs, nq, device):
    key = (bs, nq)
    if key not in _alloc_cache:
        _alloc_cache[key] = {
            "output": torch.empty((bs, nq, V_HEAD_DIM), dtype=torch.bfloat16, device=device),
        }
    return _alloc_cache[key]


# ===============================================================
# AITER DECODE PATH
# ===============================================================

def _aiter_path(q, kv_data, qo_indptr, kv_indptr, config):
    """AITER cached a8w8 path for large configs."""
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]
    num_kv_splits = AITER_KV_SPLITS_MAP.get((bs, kvlen), AITER_DEFAULT_KV_SPLITS)

    q_fp8, q_scale = quantize_fp8(q)

    kv_fp8, kv_scale = kv_data["fp8"]

    meta = _get_cached_meta(
        bs, NUM_HEADS, NUM_KV_HEADS,
        q_fp8.dtype, kv_fp8.dtype,
        qo_indptr, kv_indptr, num_kv_splits,
    )
    allocs = _get_cached_allocs(bs, NUM_HEADS, q.device)
    o = allocs["output"]

    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])

    mla_decode_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d, o,
        qo_indptr, kv_indptr,
        meta["kv_indices"], meta["kv_last_page_len"],
        1,
        page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE, logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=meta["work_meta_data"],
        work_indptr=meta["work_indptr"],
        work_info_set=meta["work_info_set"],
        reduce_indptr=meta["reduce_indptr"],
        reduce_final_map=meta["reduce_final_map"],
        reduce_partial_map=meta["reduce_partial_map"],
    )
    return o


# ===============================================================
# ENTRY POINT
# ===============================================================

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]

    if (bs, kvlen) in MXFP4_CONFIGS:
        return _mxfp4_path(q, kv_data, kv_indptr, config)
    else:
        return _aiter_path(q, kv_data, qo_indptr, kv_indptr, config)
