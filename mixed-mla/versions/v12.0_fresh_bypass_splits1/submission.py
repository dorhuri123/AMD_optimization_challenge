"""
Optimized MLA decode submission -- v12.0 Fresh Bypass + Reduced Splits

Key changes from v11:
1. FRESH-ALLOC BYPASS for ALL AITER configs: calls C++ stage1/reduce directly
   (bypasses Python wrapper ~5-8us savings) but allocates logits/attn_lse fresh
   each call instead of caching them. This avoids the +12us cache penalty seen
   at bs=256 while still getting the bypass Python overhead savings.
2. REDUCED num_kv_splits for bs=256: fewer splits means less reduce work.
   - (256,1024): 16->8, (256,8192): 24->16
   - (32,8192): 48->24, (64,8192): 24->16

Per-config routing:
- MXFP4: (4,1024), (4,8192), (32,1024), (64,1024) -- Triton dot_scaled
- AITER (fresh bypass): all other configs -- direct C++ calls, fresh logits/attn_lse

Carried forward from v11/v9/v8:
1. Fused FP8 Q quantization via two Triton kernels (replaces 3-op PyTorch path)
2. Pre-allocated Q FP8 output buffer, scale buffer, and amax scratch buffer
3. MXFP4 V dequant via bf16 tl.dot
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
PAGE_SIZE: int = 1
FP8_DTYPE = aiter_dtypes.fp8

# K dim layout: 576 = 4*128 + 64 -> 5 tiles of 128 (last padded)
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

# AITER split-K tuning (fresh bypass for all)
AITER_KV_SPLITS_MAP = {
    (32, 8192): 24,    # was 48 in v11, reduced
    (64, 8192): 16,    # was 24, reduced
    (256, 1024): 8,    # was 16, reduced
    (256, 8192): 16,   # was 24, reduced
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
# FUSED FP8 QUANTIZATION HOST WRAPPER (with pre-allocated buffers)
# ===============================================================

def _get_fp8_buffers(num_elements, shape, device):
    """Get or allocate pre-allocated FP8 quantization buffers."""
    key = (num_elements, device)
    if key not in _fp8_buf_cache:
        _fp8_buf_cache[key] = {
            "fp8_out": torch.empty(num_elements, dtype=FP8_DTYPE, device=device),
            "scale_out": torch.empty(1, dtype=torch.float32, device=device),
            "amax_buf": torch.zeros(1, dtype=torch.float32, device=device),
        }
    return _fp8_buf_cache[key]


def fused_quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 using two fused Triton kernels with pre-allocated buffers.

    Returns:
        (fp8_tensor, scale) where:
            fp8_tensor: same shape as input, dtype=FP8_DTYPE
            scale: shape (1,), dtype=float32
    """
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
    stride_q_packed,  # stride per Q row in packed data
    stride_q_scale,   # stride per Q row in scale data
    stride_kv_packed, # stride per KV token in packed data
    stride_kv_scale,  # stride per KV token in scale data
    stride_v_tok,     # stride per V token (576 for bf16)
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    sm_scale,
    BLOCK_N: tl.constexpr,       # KV tokens per tile
    V_CHUNK_D: tl.constexpr,     # V dims per chunk (128)
    NUM_KV_SPLITS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """
    Stage 1: For each (batch, split, v_chunk), compute partial attention.
    All 16 Q heads processed together (BLOCK_M=16).
    K dimension tiled in 5 chunks of 128 dims via dot_scaled.
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

    q_row_base = pid_b * NUM_HEADS
    offs_m = tl.arange(0, NUM_HEADS)  # 0..15

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
    """
    MXFP4 MLA decode using hardware tl.dot_scaled on MI355X.
    Q quantized to MXFP4, K via dot_scaled, V via bf16 tl.dot.
    """
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    num_kv_splits = MXFP4_KV_SPLITS_MAP.get(
        (batch_size, kv_seq_len), MXFP4_DEFAULT_KV_SPLITS
    )

    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    q_2d = q.view(-1, QK_HEAD_DIM)  # (batch*16, 576)
    q_packed_raw, q_scale_raw = dynamic_mxfp4_quant(q_2d)
    q_packed = q_packed_raw.view(torch.uint8)
    q_scale = q_scale_raw.view(torch.uint8)

    kv_fp4_2d = kv_fp4.reshape(-1, PACKED_QK).view(torch.uint8)  # (total_kv, 288)
    kv_scale_2d = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)  # (total_kv, 576)

    BLOCK_N = 64
    V_CHUNK_D = 128
    num_v_chunks = V_HEAD_DIM // V_CHUNK_D  # 512 / 128 = 4

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
# AITER CACHED METADATA (no logits/attn_lse -- those are fresh)
# ===============================================================

def _get_cached_meta(bs, nq, nkv, q_dtype, kv_dtype, qo_indptr, kv_indptr, num_kv_splits):
    """Cached metadata for the AITER fresh-bypass path.

    Caches: work_meta_data, work_indptr, work_info_set, reduce_indptr,
    reduce_final_map, reduce_partial_map, kv_indices, kv_last_page_len.
    Does NOT cache logits/attn_lse -- those are allocated fresh each call.
    """
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
# AITER FRESH-BYPASS PATH -- direct C++ calls, fresh logits/attn_lse
# ===============================================================

def _aiter_path(q, kv_data, qo_indptr, kv_indptr, config):
    """
    Unified AITER path for ALL non-MXFP4 configs.

    Calls C++ stage1/reduce directly (bypasses Python wrapper, saves ~5-8us)
    but allocates logits and attn_lse FRESH each call instead of caching them.
    This avoids the +12us L2 cache pollution penalty seen at bs=256 when
    logits/attn_lse were cached, while still getting the bypass savings.

    Used for: (32, 8192), (64, 8192), (256, 1024), (256, 8192)
    """
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]
    num_kv_splits = AITER_KV_SPLITS_MAP.get((bs, kvlen), AITER_DEFAULT_KV_SPLITS)

    # Fused FP8 quantization
    q_fp8, q_scale = fused_quantize_fp8(q)

    kv_fp8, kv_scale = kv_data["fp8"]

    meta = _get_cached_meta(
        bs, NUM_HEADS, NUM_KV_HEADS,
        q_fp8.dtype, kv_fp8.dtype,
        qo_indptr, kv_indptr, num_kv_splits,
    )
    allocs = _get_cached_allocs(bs, NUM_HEADS, q.device)
    o = allocs["output"]

    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])

    # Fresh allocation each call -- avoids L2 cache pollution at bs=256
    # while still bypassing the Python wrapper overhead
    num_partials = meta["reduce_partial_map"].size(0)
    logits = torch.empty(
        (num_partials, 1, NUM_HEADS, V_HEAD_DIM),
        dtype=torch.float32, device="cuda",
    )
    attn_lse = torch.empty(
        (num_partials, 1, NUM_HEADS, 1),
        dtype=torch.float32, device="cuda",
    )

    # Direct call to AITER C++ stage1 -- 19 args (no final_lse parameter)
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d,
        qo_indptr,
        kv_indptr,
        meta["kv_indices"],
        meta["kv_last_page_len"],
        None,                              # num_kv_splits_indptr (unused in persistent mode)
        meta["work_meta_data"],
        meta["work_indptr"],
        meta["work_info_set"],
        1,                                 # max_seqlen_q
        PAGE_SIZE,                         # page_size
        NUM_KV_HEADS,                      # nhead_kv
        SM_SCALE,                          # sm_scale
        logits,                            # splitData (fresh)
        attn_lse,                          # splitLse (fresh)
        o,                                 # output
        q_scale,                           # q_scale
        kv_scale,                          # kv_scale
    )

    # Direct call to AITER C++ reduce
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        meta["reduce_indptr"],
        meta["reduce_final_map"],
        meta["reduce_partial_map"],
        1,                                 # max_seqlen_q
        o,                                 # final_output
        None,                              # final_lse
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
        return _aiter_path(q, kv_data, qo_indptr, kv_indptr, config)
