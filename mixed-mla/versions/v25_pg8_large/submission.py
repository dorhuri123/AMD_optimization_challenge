"""
v24: Optimal hybrid — 3 paths routed by (batch_size, kv_seq_len).

Path A: MXFP4 Triton (small configs)
  {(4,1024), (4,8192), (32,1024)}
  dot_scaled QK with e2m1, bf16 V accumulation

Path B: a16w8 + PAGE_SIZE=2 (medium/large configs)
  {(32,8192), (64,1024), (64,8192), (256,1024)}
  bf16 Q (no quant), fp8 KV, paged kv_indptr

Path C: a8w8 + PAGE_SIZE=1 (bs=256, kv=8192 only)
  fp8 Q + fp8 KV, persistent cached metadata

Expected geomean: ~52 μs
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
FP8_DTYPE = aiter_dtypes.fp8

# MXFP4 layout constants
PACKED_QK: int = 288       # 576 / 2 packed bytes
NUM_SCALES: int = 18       # 576 / 32 scale blocks

# ===============================================================
# ROUTING
# ===============================================================

# Path A: MXFP4 Triton
MXFP4_CONFIGS = {(4, 1024), (4, 8192), (32, 1024)}
MXFP4_KV_SPLITS = {
    (4, 1024): 4,
    (4, 8192): 16,
    (32, 1024): 4,
}

# Path B: a16w8 + PAGE_SIZE=2
A16W8_PG2_CONFIGS = {(32, 8192), (64, 1024), (64, 8192), (256, 1024)}
# splits: 8 if total_kv<=4096 else 16

# Path C: a8w8 + PAGE_SIZE=8 — for (256, 8192) and other kv=8192 large configs
# Competitor uses pg8 for kv>=8192: fewer page transactions, better coalescing
A8W8_PAGE_SIZE = 8
A8W8_KV_GRAN = max(1, 16 // 8)  # = 2
A8W8_SPLITS = 16

# ===============================================================
# CACHES
# ===============================================================

_mxfp4_buf_cache: dict = {}
_a16w8_meta_cache: dict = {}
_a8w8_meta_cache: dict = {}
_output_cache: dict = {}


# ===============================================================
# MXFP4 TRITON KERNEL — STAGE 1
# ===============================================================

@triton.jit
def _mla_mxfp4_stage1(
    Q_packed_ptr,     # [batch*16, 288] uint8 (packed e2m1)
    Q_scale_ptr,      # [batch*16, 18] uint8 (e8m0 scales)
    K_packed_ptr,     # [total_kv, 288] uint8 (packed e2m1)
    K_scale_ptr,      # [total_kv, 18] uint8 (e8m0 scales)
    V_bf16_ptr,       # [total_kv, 576] bf16 (first 512 dims used)
    Partial_O_ptr,    # [batch, splits, 16, V_DIM] f32
    Partial_m_ptr,    # [batch, splits, 16] f32
    Partial_l_ptr,    # [batch, splits, 16] f32
    kv_indptr_ptr,    # [batch+1] i32
    stride_q_packed,
    stride_q_scale,
    stride_kv_packed,
    stride_kv_scale,
    stride_v_tok,
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    sm_scale,
    BLOCK_N: tl.constexpr,
    V_CHUNK_D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_bs = tl.program_id(0)
    pid_v = tl.program_id(2)

    pid_b = pid_bs // NUM_KV_SPLITS
    pid_s = pid_bs % NUM_KV_SPLITS

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, NUM_KV_SPLITS)
    split_kv_start = pid_s * split_size
    split_kv_end = tl.minimum(split_kv_start + split_size, kv_len)

    q_row_base = pid_b * NUM_HEADS
    offs_m = tl.arange(0, NUM_HEADS)

    vd_start = pid_v * V_CHUNK_D

    m_prev = tl.full([NUM_HEADS], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(split_kv_end - split_kv_start, BLOCK_N)

    for tile_idx in range(num_tiles):
        tile_start = split_kv_start + tile_idx * BLOCK_N
        kv_offsets = tile_start + tl.arange(0, BLOCK_N)
        mask_kv = kv_offsets < split_kv_end
        kv_idx = kv_start + kv_offsets

        qk = tl.zeros([NUM_HEADS, BLOCK_N], dtype=tl.float32)

        for k_tile in tl.static_range(5):
            k_packed_start = k_tile * 64
            k_scale_start = k_tile * 4

            q_d_offs = k_packed_start + tl.arange(0, 64)
            q_chunk = tl.load(
                Q_packed_ptr + (q_row_base + offs_m[:, None]) * stride_q_packed + q_d_offs[None, :],
                mask=(q_d_offs[None, :] < 288),
                other=0,
            )

            qs_offs = k_scale_start + tl.arange(0, 4)
            q_scale_chunk = tl.load(
                Q_scale_ptr + (q_row_base + offs_m[:, None]) * stride_q_scale + qs_offs[None, :],
                mask=(qs_offs[None, :] < 18),
                other=0,
            )

            k_d_offs = k_packed_start + tl.arange(0, 64)
            k_chunk = tl.load(
                K_packed_ptr + kv_idx[None, :] * stride_kv_packed + k_d_offs[:, None],
                mask=mask_kv[None, :] & (k_d_offs[:, None] < 288),
                other=0,
            )

            ks_offs = k_scale_start + tl.arange(0, 4)
            k_scale_chunk = tl.load(
                K_scale_ptr + kv_idx[:, None] * stride_kv_scale + ks_offs[None, :],
                mask=mask_kv[:, None] & (ks_offs[None, :] < 18),
                other=0,
            )

            qk = tl.dot_scaled(
                q_chunk, q_scale_chunk, "e2m1",
                k_chunk, k_scale_chunk, "e2m1",
                fast_math=True, acc=qk,
            )

        qk *= sm_scale
        qk = tl.where(mask_kv[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_prev, tl.max(qk, 1))
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)
        p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
        p = tl.where(mask_kv[None, :], p, 0.0)

        acc = acc * alpha[:, None]
        l_prev = l_prev * alpha + tl.sum(p, 1)
        m_prev = m_new

        vd_offsets = vd_start + tl.arange(0, V_CHUNK_D)
        v_tile = tl.load(
            V_bf16_ptr + kv_idx[:, None] * stride_v_tok + vd_offsets[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

    po_base = (Partial_O_ptr + pid_b * stride_po_b + pid_s * stride_po_s + vd_start)
    head_offs = tl.arange(0, NUM_HEADS)
    v_offs = tl.arange(0, V_CHUNK_D)
    tl.store(
        po_base + head_offs[:, None] * stride_po_h + v_offs[None, :],
        acc,
    )

    if pid_v == 0:
        ml_base = pid_b * stride_ml_b + pid_s * stride_ml_s
        tl.store(Partial_m_ptr + ml_base + head_offs * stride_ml_h, m_prev)
        tl.store(Partial_l_ptr + ml_base + head_offs * stride_ml_h, l_prev)


# ===============================================================
# MXFP4 TRITON KERNEL — STAGE 2: REDUCE
# ===============================================================

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


# ===============================================================
# PATH A: MXFP4 TRITON
# ===============================================================

def _get_mxfp4_buffers(batch_size, num_kv_splits, device):
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
        }
    return _mxfp4_buf_cache[key]


def _mxfp4_path(q, kv_data, kv_indptr, config):
    """Path A: MXFP4 dot_scaled for small configs."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    num_kv_splits = MXFP4_KV_SPLITS[batch_size, kv_seq_len]

    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    # Quantize Q to MXFP4
    q_2d = q.view(-1, QK_HEAD_DIM)
    q_packed_raw, q_scale_raw = dynamic_mxfp4_quant(q_2d)
    q_packed = q_packed_raw.view(torch.uint8)
    q_scale = q_scale_raw.view(torch.uint8)

    kv_fp4_2d = kv_fp4.reshape(-1, PACKED_QK).view(torch.uint8)
    kv_scale_2d = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)

    BLOCK_N = 64
    V_CHUNK_D = 128
    num_v_chunks = V_HEAD_DIM // V_CHUNK_D  # 4

    bufs = _get_mxfp4_buffers(batch_size, num_kv_splits, q.device)

    # Get or cache output buffer
    out_key = (batch_size, NUM_HEADS)
    if out_key not in _output_cache:
        _output_cache[out_key] = torch.empty(
            (batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device,
        )
    o = _output_cache[out_key]

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
        SM_SCALE,
        BLOCK_N=BLOCK_N, V_CHUNK_D=V_CHUNK_D,
        NUM_KV_SPLITS=num_kv_splits,
        NUM_HEADS=NUM_HEADS,
    )

    grid2 = (batch_size, NUM_HEADS, num_v_chunks)
    _mla_mxfp4_reduce[grid2](
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"], o,
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        o.stride(0), o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        V_CHUNK_D=V_CHUNK_D,
        NUM_HEADS=NUM_HEADS,
    )

    return o


# ===============================================================
# PATH B: a16w8 + PAGE_SIZE=2
# ===============================================================

PAGE_SIZE_2 = 2

def _get_a16w8_pg2_meta(batch_size, kv_seq_len, num_kv_splits, qo_indptr, kv_indptr):
    key = (batch_size, kv_seq_len, num_kv_splits)
    if key not in _a16w8_meta_cache:
        seq_lens_kv = kv_indptr[1:] - kv_indptr[:-1]
        num_pages_per_req = (seq_lens_kv + PAGE_SIZE_2 - 1) // PAGE_SIZE_2
        kv_indptr_paged = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        kv_indptr_paged[1:] = torch.cumsum(num_pages_per_req, dim=0)

        kv_last_page_lens = (seq_lens_kv % PAGE_SIZE_2).to(torch.int32)
        kv_last_page_lens = torch.where(kv_last_page_lens == 0, PAGE_SIZE_2, kv_last_page_lens)

        total_pages = int(kv_indptr_paged[-1].item())
        kv_granularity = max(1, 16 // PAGE_SIZE_2)  # 8

        info = get_mla_metadata_info_v1(
            batch_size, 1, NUM_HEADS, torch.bfloat16, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_paged, kv_last_page_lens,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE_2,
            kv_granularity=kv_granularity,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=torch.bfloat16,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
        _a16w8_meta_cache[key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged)

    return _a16w8_meta_cache[key]


def _a16w8_pg2_path(q, kv_data, qo_indptr, kv_indptr, config):
    """Path B: bf16 Q + fp8 KV, page_size=2."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    total_kv = batch_size * kv_seq_len
    num_kv_splits = 8 if total_kv <= 4096 else 16

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged) = \
        _get_a16w8_pg2_meta(batch_size, kv_seq_len, num_kv_splits, qo_indptr, kv_indptr)

    out_key = (batch_size, NUM_HEADS)
    if out_key not in _output_cache:
        _output_cache[out_key] = torch.empty(
            (batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device,
        )
    o = _output_cache[out_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    num_pages = total_kv // PAGE_SIZE_2
    kv_buffer_4d = kv_buffer_fp8.view(num_pages, PAGE_SIZE_2, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr_paged, kv_indices, kv_last_page_lens,
        1,
        page_size=PAGE_SIZE_2,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=None,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o


# ===============================================================
# PATH C: a8w8 + PAGE_SIZE=1
# ===============================================================

PAGE_SIZE_1 = 1

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _get_a8w8_meta(batch_size, num_kv_splits, q_dtype, kv_dtype, qo_indptr, kv_indptr):
    key = (batch_size, num_kv_splits, q_dtype, kv_dtype)
    if key not in _a8w8_meta_cache:
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        total_kv = int(kv_indptr[-1].item())
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")

        info = get_mla_metadata_info_v1(
            batch_size, 1, NUM_HEADS, q_dtype, kv_dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE_1,
            kv_granularity=max(PAGE_SIZE_1, 16),
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        )

        _a8w8_meta_cache[key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len)

    return _a8w8_meta_cache[key]


def _a8w8_path(q, kv_data, qo_indptr, kv_indptr, config):
    """Path C: fp8 Q + fp8 KV, page_size=8 for bs=256,kv=8192."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    num_kv_splits = A8W8_SPLITS  # 16
    ps = A8W8_PAGE_SIZE  # 8

    q_fp8, q_scale = quantize_fp8(q)
    kv_fp8, kv_scale = kv_data["fp8"]

    total_kv = batch_size * kv_seq_len
    cache_key = (batch_size, kv_seq_len, num_kv_splits, "pg8")
    if cache_key not in _a8w8_meta_cache:
        # Compute paged kv_indptr for page_size=8
        seq_lens_kv = kv_indptr[1:] - kv_indptr[:-1]
        num_pages_per_req = (seq_lens_kv + ps - 1) // ps
        kv_indptr_paged = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        kv_indptr_paged[1:] = torch.cumsum(num_pages_per_req, dim=0)
        kv_last_page_lens = (seq_lens_kv % ps).to(torch.int32)
        kv_last_page_lens = torch.where(kv_last_page_lens == 0, ps, kv_last_page_lens)
        total_pages = int(kv_indptr_paged[-1].item())
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")

        info = get_mla_metadata_info_v1(
            batch_size, 1, NUM_HEADS, q_fp8.dtype, kv_fp8.dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(
            qo_indptr, kv_indptr_paged, kv_last_page_lens,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=ps, kv_granularity=A8W8_KV_GRAN,
            max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=q_fp8.dtype, dtype_kv=kv_fp8.dtype,
        )
        _a8w8_meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged) = _a8w8_meta_cache[cache_key]

    out_key = (batch_size, NUM_HEADS)
    if out_key not in _output_cache:
        _output_cache[out_key] = torch.empty(
            (batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device,
        )
    o = _output_cache[out_key]

    num_pages = total_kv // ps
    kv_4d = kv_fp8.view(num_pages, ps, NUM_KV_HEADS, kv_fp8.shape[-1])

    mla_decode_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d, o,
        qo_indptr, kv_indptr_paged,
        kv_indices, kv_last_page_lens,
        1,
        page_size=ps,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
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
    elif (bs, kvlen) in A16W8_PG2_CONFIGS:
        return _a16w8_pg2_path(q, kv_data, qo_indptr, kv_indptr, config)
    else:
        # (256, 8192) — a8w8 path
        return _a8w8_path(q, kv_data, qo_indptr, kv_indptr, config)
