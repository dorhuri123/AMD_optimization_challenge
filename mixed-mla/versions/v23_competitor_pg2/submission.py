"""
v23: Competitor-inspired PAGE_SIZE=2 approach with a16w8 / a8w8 fallback

Based on THINNGO2511's proven 33us approach:
1. PAGE_SIZE=2 -- KV reshaped as (num_pages, 2, 1, 576)
2. Option A (primary): a16w8 -- bf16 Q + fp8 KV, q_scale=None
3. Option B (fallback): a8w8 -- fp8 Q + fp8 KV with PAGE_SIZE=2
4. kv_granularity = max(1, 16 // PAGE_SIZE) = 8
5. num_kv_splits = 8 if total_kv <= 4096, else 16
6. kv_last_page_len = all PAGE_SIZE (=2)
7. kv_indices = arange(num_pages)
8. Persistent mode with cached metadata

MXFP4 Triton path kept for small-batch configs.

Toggle USE_A16W8 to switch between a16w8 and a8w8 for the AITER path.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
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
PAGE_SIZE: int = 2       # KEY CHANGE: page_size=2 like competitor
FP8_DTYPE = aiter_dtypes.fp8

# a16w8 vs a8w8 toggle -- set False to fall back to fp8 Q + fp8 KV
USE_A16W8: bool = True

# K dim layout for MXFP4
PACKED_QK: int = 288       # 576 / 2 packed bytes
NUM_SCALES: int = 18       # 576 / 32 scale blocks

# ===============================================================
# ROUTING CONFIGS
# ===============================================================

MXFP4_CONFIGS = {(4, 1024), (4, 8192), (32, 1024), (64, 1024)}

# MXFP4 split-K tuning
MXFP4_KV_SPLITS_MAP = {
    (4, 1024): 4,
    (4, 8192): 16,
    (32, 1024): 4,
    (64, 1024): 4,
}
MXFP4_DEFAULT_KV_SPLITS = 8

# AITER split tuning (competitor formula)
# num_kv_splits = 8 if total_kv <= 4096, else 16
# total_kv = bs * kv_seq_len
AITER_KV_SPLITS_MAP = {
    (32, 8192):  16,   # total_kv=262144 > 4096 -> 16
    (64, 8192):  16,   # total_kv=524288 > 4096 -> 16
    (256, 1024): 8,    # total_kv=262144 > 4096 -> 16, but competitor uses 8 for <=4096
    (256, 8192): 16,   # total_kv=2097152 > 4096 -> 16
}
AITER_DEFAULT_KV_SPLITS = 16

# Caches
_mxfp4_buf_cache: dict = {}
_meta_cache: dict = {}
_alloc_cache: dict = {}
_fp8_buf_cache: dict = {}


# ===============================================================
# FUSED FP8 QUANTIZATION TRITON KERNELS
# ===============================================================

@triton.jit
def _amax_kernel(
    input_ptr,       # [N] flat input (any float type)
    amax_ptr,        # [1] output: global amax (float32)
    N,               # total number of elements
    BLOCK: tl.constexpr,
):
    """Each block computes local amax, then atomically updates global max."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_amax = tl.max(tl.abs(x))

    tl.atomic_max(amax_ptr, local_amax)


@triton.jit
def _quantize_fp8_kernel(
    input_ptr,       # [N] flat input (any float type)
    output_ptr,      # [N] flat output (fp8)
    amax_ptr,        # [1] global amax (float32), read-only
    scale_ptr,       # [1] output: scale (float32)
    fp8_max,         # scalar constexpr: max representable FP8 value
    fp8_min,         # scalar constexpr: min representable FP8 value (negative)
    N,               # total number of elements
    BLOCK: tl.constexpr,
):
    """Read global amax, compute scale = amax / fp8_max, quantize elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    amax = tl.load(amax_ptr)
    amax = tl.maximum(amax, 1e-12)
    scale = amax / fp8_max

    if pid == 0:
        tl.store(scale_ptr, scale)

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_scaled = x / scale
    x_clamped = tl.minimum(tl.maximum(x_scaled, fp8_min), fp8_max)

    tl.store(output_ptr + offs, x_clamped, mask=mask)


# ===============================================================
# FUSED FP8 QUANTIZATION HOST WRAPPER
# ===============================================================

def _get_fp8_buffers(num_elements, shape, device):
    key = (num_elements, device)
    if key not in _fp8_buf_cache:
        _fp8_buf_cache[key] = {
            "fp8_out": torch.empty(num_elements, dtype=FP8_DTYPE, device=device),
            "scale_out": torch.empty(1, dtype=torch.float32, device=device),
            "amax_buf": torch.zeros(1, dtype=torch.float32, device=device),
        }
    return _fp8_buf_cache[key]


def fused_quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    fp8_max_val = finfo.max
    fp8_min_val = finfo.min

    flat = tensor.reshape(-1)
    N = flat.numel()

    bufs = _get_fp8_buffers(N, tensor.shape, tensor.device)
    fp8_flat = bufs["fp8_out"]
    scale_out = bufs["scale_out"]
    amax_buf = bufs["amax_buf"]

    amax_buf.zero_()

    BLOCK = 1024
    grid_size = (N + BLOCK - 1) // BLOCK

    _amax_kernel[(grid_size,)](
        flat, amax_buf, N,
        BLOCK=BLOCK,
    )

    _quantize_fp8_kernel[(grid_size,)](
        flat, fp8_flat, amax_buf, scale_out,
        fp8_max_val, fp8_min_val, N,
        BLOCK=BLOCK,
    )

    return fp8_flat.view(tensor.shape), scale_out


# ===============================================================
# MXFP4 TRITON KERNEL -- STAGE 1
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
        tl.store(
            Partial_m_ptr + ml_base + head_offs * stride_ml_h,
            m_prev,
        )
        tl.store(
            Partial_l_ptr + ml_base + head_offs * stride_ml_h,
            l_prev,
        )


# ===============================================================
# MXFP4 TRITON KERNEL -- STAGE 2: REDUCE
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
# MXFP4 DECODE PATH
# ===============================================================

def _mxfp4_path(q, kv_data, kv_indptr, config):
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    num_kv_splits = MXFP4_KV_SPLITS_MAP.get(
        (batch_size, kv_seq_len), MXFP4_DEFAULT_KV_SPLITS
    )

    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    q_2d = q.view(-1, QK_HEAD_DIM)
    q_packed_raw, q_scale_raw = dynamic_mxfp4_quant(q_2d)
    q_packed = q_packed_raw.view(torch.uint8)
    q_scale = q_scale_raw.view(torch.uint8)

    kv_fp4_2d = kv_fp4.reshape(-1, PACKED_QK).view(torch.uint8)
    kv_scale_2d = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)

    BLOCK_N = 64
    V_CHUNK_D = 128
    num_v_chunks = V_HEAD_DIM // V_CHUNK_D

    bufs = _mxfp4_get_buffers(batch_size, num_kv_splits, q.device)

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
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"], bufs["output"],
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        bufs["output"].stride(0), bufs["output"].stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        V_CHUNK_D=V_CHUNK_D,
        NUM_HEADS=NUM_HEADS,
    )

    return bufs["output"]


# ===============================================================
# AITER PERSISTENT PATH with PAGE_SIZE=2 (competitor approach)
# ===============================================================

def _get_num_kv_splits(bs, kv_seq_len):
    """Competitor formula: 8 if total_kv <= 4096, else 16."""
    total_kv = bs * kv_seq_len
    if total_kv <= 4096:
        return 8
    return 16


def _get_cached_meta_pg2(bs, total_kv, q_dtype, kv_dtype, qo_indptr, kv_indptr, num_kv_splits):
    """
    Cached persistent-mode metadata for PAGE_SIZE=2 approach.

    Key differences from PAGE_SIZE=1:
    - kv_indices = arange(num_pages) where num_pages = total_kv // PAGE_SIZE
    - kv_last_page_len = all PAGE_SIZE (=2) since all pages are full
    - kv_granularity = max(1, 16 // PAGE_SIZE) = 8
    - kv_indptr stays token-based (original), NOT page-based
    """
    key = (bs, num_kv_splits, q_dtype, kv_dtype)
    if key not in _meta_cache:
        num_pages = total_kv // PAGE_SIZE
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.full((bs,), PAGE_SIZE, dtype=torch.int32, device="cuda")

        kv_granularity = max(1, 16 // PAGE_SIZE)  # = 8

        nq = NUM_HEADS
        nkv = NUM_KV_HEADS

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
            page_size=PAGE_SIZE, kv_granularity=kv_granularity,
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


def _aiter_path_pg2(q, kv_data, qo_indptr, kv_indptr, config):
    """
    AITER persistent decode with PAGE_SIZE=2 (competitor approach).

    Two modes:
    - a16w8: bf16 Q directly, q_scale=None, kv_scale from fp8 KV
    - a8w8:  fp8 Q with fused quantization, fp8 KV
    """
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]
    total_kv = bs * kvlen  # Avoid .item() GPU-CPU sync
    device = q.device

    num_kv_splits = _get_num_kv_splits(bs, kvlen)

    kv_fp8, kv_scale = kv_data["fp8"]

    # Reshape KV for PAGE_SIZE=2: (total_kv,576) -> (num_pages, 2, 1, 576)
    num_pages = total_kv // PAGE_SIZE
    kv_4d = kv_fp8.view(num_pages, PAGE_SIZE, NUM_KV_HEADS, QK_HEAD_DIM)

    if USE_A16W8:
        # a16w8: pass bf16 Q directly, no Q quantization
        q_input = q.view(-1, NUM_HEADS, QK_HEAD_DIM)  # bf16
        q_dtype = q_input.dtype
        q_scale_input = None
    else:
        # a8w8: quantize Q to fp8
        q_fp8, q_scale_val = fused_quantize_fp8(q)
        q_input = q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM)
        q_dtype = q_input.dtype
        q_scale_input = q_scale_val

    kv_dtype = kv_fp8.dtype

    meta = _get_cached_meta_pg2(
        bs, total_kv,
        q_dtype, kv_dtype,
        qo_indptr, kv_indptr, num_kv_splits,
    )
    allocs = _get_cached_allocs(bs, NUM_HEADS, device)
    o = allocs["output"]

    # Use mla_decode_fwd wrapper (handles arg count correctly on MI355X)
    mla_decode_fwd(
        q_input,                               # bf16 (a16w8) or fp8 (a8w8)
        kv_4d, o,
        qo_indptr, kv_indptr,
        meta["kv_indices"],
        meta["kv_last_page_len"],
        1,                                     # max_seqlen_q
        page_size=PAGE_SIZE,                   # page_size=2
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale_input,                 # None for a16w8, fp8 scale for a8w8
        kv_scale=kv_scale,
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

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]

    if (bs, kvlen) in MXFP4_CONFIGS:
        return _mxfp4_path(q, kv_data, kv_indptr, config)
    else:
        return _aiter_path_pg2(q, kv_data, qo_indptr, kv_indptr, config)
