"""
v34: Direct ASM stage1+reduce — bypass mla_decode_fwd Python wrapper.

Key optimizations over v24:
1. Direct aiter.mla_decode_stage1_asm_fwd + aiter.mla_reduce_v1 calls
   (skips mla_decode_fwd wrapper overhead, ~7.6% speedup)
2. CU-aware optimal splits per config
3. kvg=32 everywhere (not 64) — ~3% improvement
4. Environment variables for ROCm optimization
5. Keep MXFP4 Triton for (4,1024) and (32,1024)

Routing:
  (4, 1024):   MXFP4 Triton (splits=4)
  (4, 8192):   a8w8  direct ASM pg1 (splits=64)
  (32, 1024):  MXFP4 Triton (splits=4)
  (32, 8192):  a16w8 direct ASM pg2 (splits=16)
  (64, 1024):  a16w8 direct ASM pg2 (splits=8)
  (64, 8192):  a16w8 direct ASM pg2 (splits=8)
  (256, 1024): a16w8 direct ASM pg2 (splits=8)
  (256, 8192): a8w8  direct ASM pg1 (splits=4)
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["GPU_MAX_HW_QUEUES"] = "4"
os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"

import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
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
# ROUTING TABLE
# ===============================================================

# (batch_size, kv_seq_len) -> (path, splits)
# path: "mxfp4", "a16w8_pg2", "a8w8_pg1"
ROUTING = {
    (4, 1024):   ("mxfp4",     4),
    (4, 8192):   ("a8w8_pg1",  64),
    (32, 1024):  ("mxfp4",     4),
    (32, 8192):  ("a16w8_pg2", 16),
    (64, 1024):  ("a16w8_pg2", 8),
    (64, 8192):  ("a16w8_pg2", 8),
    (256, 1024): ("a16w8_pg2", 8),
    (256, 8192): ("a8w8_pg1",  4),
}

# ===============================================================
# CACHES
# ===============================================================

_mxfp4_buf_cache: dict = {}
_direct_meta_cache: dict = {}
_output_cache: dict = {}
_logits_cache: dict = {}
_lse_cache: dict = {}


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
        tl.store(Partial_m_ptr + ml_base + head_offs * stride_ml_h, m_prev)
        tl.store(Partial_l_ptr + ml_base + head_offs * stride_ml_h, l_prev)


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
# PATH: MXFP4 TRITON (small configs)
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


def _mxfp4_path(q, kv_data, kv_indptr, num_kv_splits, config):
    """MXFP4 dot_scaled Triton path for small configs."""
    batch_size = config["batch_size"]

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
# DIRECT ASM HELPERS
# ===============================================================

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _get_direct_meta(batch_size, kv_seq_len, num_kv_splits, page_size, q_dtype, kv_dtype, qo_indptr, kv_indptr):
    """Compute persistent-mode metadata for direct stage1+reduce calls."""
    key = (batch_size, kv_seq_len, num_kv_splits, page_size, q_dtype, kv_dtype)
    if key not in _direct_meta_cache:
        # Compute paged kv_indptr
        seq_lens_kv = kv_indptr[1:] - kv_indptr[:-1]
        num_pages_per_req = (seq_lens_kv + page_size - 1) // page_size
        kv_indptr_paged = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        kv_indptr_paged[1:] = torch.cumsum(num_pages_per_req, dim=0)

        kv_last_page_lens = (seq_lens_kv % page_size).to(torch.int32)
        kv_last_page_lens = torch.where(kv_last_page_lens == 0, page_size, kv_last_page_lens)

        total_pages = int(kv_indptr_paged[-1].item())
        kv_granularity = max(1, 32 // page_size)  # kvg=32 everywhere

        info = get_mla_metadata_info_v1(
            batch_size, 1, NUM_HEADS, q_dtype, kv_dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_paged, kv_last_page_lens,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=page_size,
            kv_granularity=kv_granularity,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        )

        kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
        _direct_meta_cache[key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged)

    return _direct_meta_cache[key]


def _get_logits_lse(batch_size, num_kv_splits, rpm, device):
    """Get or allocate logits and attn_lse buffers for persistent mode."""
    key = (batch_size, num_kv_splits)
    if key not in _logits_cache:
        _logits_cache[key] = torch.empty(
            (rpm.size(0), 1, NUM_HEADS, V_HEAD_DIM),
            dtype=torch.float32, device=device,
        )
        _lse_cache[key] = torch.empty(
            (rpm.size(0), 1, NUM_HEADS, 1),
            dtype=torch.float32, device=device,
        )
    return _logits_cache[key], _lse_cache[key]


# ===============================================================
# PATH: a16w8 direct ASM (bf16 Q, fp8 KV) with page_size=2
# ===============================================================

PAGE_SIZE_2 = 2

def _a16w8_pg2_direct_path(q, kv_data, qo_indptr, kv_indptr, num_kv_splits, config):
    """Direct ASM: bf16 Q + fp8 KV, page_size=2."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged) = \
        _get_direct_meta(
            batch_size, kv_seq_len, num_kv_splits, PAGE_SIZE_2,
            torch.bfloat16, FP8_DTYPE,
            qo_indptr, kv_indptr,
        )

    out_key = (batch_size, NUM_HEADS)
    if out_key not in _output_cache:
        _output_cache[out_key] = torch.empty(
            (batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device,
        )
    o = _output_cache[out_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    num_pages = total_kv // PAGE_SIZE_2
    kv_buffer_4d = kv_buffer_fp8.view(num_pages, PAGE_SIZE_2, NUM_KV_HEADS, kv_buffer_fp8.shape[-1])

    logits, attn_lse = _get_logits_lse(batch_size, num_kv_splits, rpm, q.device)

    # Direct stage1 ASM call
    aiter.mla_decode_stage1_asm_fwd(
        q,                      # q: [bs, nhead, qk_head_dim] bf16
        kv_buffer_4d,           # kv_buffer: [num_pages, page_size, nhead_kv, qk_head_dim] fp8
        qo_indptr,              # qo_indptr
        kv_indptr_paged,        # kv_indptr (paged)
        kv_indices,             # kv_indices
        kv_last_page_lens,      # kv_last_page_lens
        None,                   # num_kv_splits_indptr (None for persistent)
        wm,                     # work_meta_data
        wi,                     # work_indptr
        wis,                    # work_info_set
        1,                      # max_seqlen_q
        PAGE_SIZE_2,            # page_size
        NUM_KV_HEADS,           # nhead_kv
        SM_SCALE,               # sm_scale
        logits,                 # logits output
        attn_lse,               # attn_lse output
        o,                      # o (output, may be used for final_out shortcut)
        None,                   # final_lse
        None,                   # q_scale (None for bf16 Q)
        kv_scale,               # kv_scale
    )

    # Direct reduce call
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        ri,                     # reduce_indptr
        rfm,                    # reduce_final_map
        rpm,                    # reduce_partial_map
        1,                      # max_seqlen_q
        o,                      # output
        None,                   # final_lse
    )

    return o


# ===============================================================
# PATH: a8w8 direct ASM (fp8 Q, fp8 KV) with page_size=1
# ===============================================================

PAGE_SIZE_1 = 1

def _a8w8_pg1_direct_path(q, kv_data, qo_indptr, kv_indptr, num_kv_splits, config):
    """Direct ASM: fp8 Q + fp8 KV, page_size=1."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    q_fp8, q_scale = quantize_fp8(q)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_lens, kv_indptr_paged) = \
        _get_direct_meta(
            batch_size, kv_seq_len, num_kv_splits, PAGE_SIZE_1,
            q_fp8.dtype, FP8_DTYPE,
            qo_indptr, kv_indptr,
        )

    out_key = (batch_size, NUM_HEADS)
    if out_key not in _output_cache:
        _output_cache[out_key] = torch.empty(
            (batch_size, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device,
        )
    o = _output_cache[out_key]

    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE_1, NUM_KV_HEADS, kv_fp8.shape[-1])

    logits, attn_lse = _get_logits_lse(batch_size, num_kv_splits, rpm, q.device)

    # Direct stage1 ASM call
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),  # q fp8
        kv_4d,                  # kv_buffer fp8
        qo_indptr,
        kv_indptr_paged,        # kv_indptr (paged, same as kv_indptr for pg1)
        kv_indices,
        kv_last_page_lens,
        None,                   # num_kv_splits_indptr (None for persistent)
        wm,                     # work_meta_data
        wi,                     # work_indptr
        wis,                    # work_info_set
        1,                      # max_seqlen_q
        PAGE_SIZE_1,            # page_size
        NUM_KV_HEADS,           # nhead_kv
        SM_SCALE,               # sm_scale
        logits,                 # logits output
        attn_lse,               # attn_lse output
        o,                      # o
        None,                   # final_lse
        q_scale,                # q_scale
        kv_scale,               # kv_scale
    )

    # Direct reduce call
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        ri,
        rfm,
        rpm,
        1,                      # max_seqlen_q
        o,
        None,                   # final_lse
    )

    return o


# ===============================================================
# ENTRY POINT
# ===============================================================

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]

    path, num_kv_splits = ROUTING[(bs, kvlen)]

    if path == "mxfp4":
        return _mxfp4_path(q, kv_data, kv_indptr, num_kv_splits, config)
    elif path == "a16w8_pg2":
        return _a16w8_pg2_direct_path(q, kv_data, qo_indptr, kv_indptr, num_kv_splits, config)
    elif path == "a8w8_pg1":
        return _a8w8_pg1_direct_path(q, kv_data, qo_indptr, kv_indptr, num_kv_splits, config)
    else:
        raise ValueError(f"Unknown path: {path}")
