"""
Optimized MLA decode submission -- v22 Small Config Specialist

Two key optimizations for small configs:

1. TRITON MXFP4 Q QUANTIZATION: Replaces AITER's dynamic_mxfp4_quant
   with a single fused Triton kernel. The AITER version launches a HIP kernel
   that is ~15-25μs overhead. Our Triton version should be ~3-5μs.
   FALLBACK: If the Triton quant produces incorrect results, set
   USE_TRITON_QUANT=False to fall back to AITER's dynamic_mxfp4_quant.

2. FUSED SINGLE-SPLIT KERNEL: For (4,1024), use splits=1 and write
   final bf16 output directly from stage1. Eliminates the reduce kernel
   entirely (~2-5μs savings).

Per-config routing:
  MXFP4 fused (splits=1): (4,1024)
  MXFP4 multi-split: (4,8192), (32,1024), (64,1024)
  AITER fresh bypass: all other configs
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
QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512
SM_SCALE: float = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE: int = 2
FP8_DTYPE = aiter_dtypes.fp8

PACKED_QK: int = 288
NUM_SCALES: int = 18

# Toggle: set to False if Triton quant doesn't match AITER's format
USE_TRITON_QUANT: bool = True

# ===============================================================
# ROUTING CONFIGS
# ===============================================================

MXFP4_CONFIGS = {(4, 1024), (4, 8192), (32, 1024), (64, 1024)}
MXFP4_SINGLE_SPLIT_CONFIGS = {(4, 1024)}

MXFP4_KV_SPLITS_MAP = {
    (4, 1024): 1,    # single split, fused
    (4, 8192): 16,
    (32, 1024): 4,
    (64, 1024): 4,
}
MXFP4_DEFAULT_KV_SPLITS = 8

AITER_KV_SPLITS_MAP = {
    (32, 8192): 24,
    (64, 8192): 16,
    (256, 1024): 16,
    (256, 8192): 24,
}
AITER_DEFAULT_KV_SPLITS = 16

# Caches
_mxfp4_buf_cache: dict = {}
_meta_cache: dict = {}
_alloc_cache: dict = {}
_fp8_buf_cache: dict = {}
_mxfp4_q_cache: dict = {}


# ===============================================================
# TRITON MXFP4 Q QUANTIZATION
# ===============================================================

@triton.jit
def _mxfp4_quant_fused(
    input_ptr,       # [num_rows, 576] bf16
    packed_ptr,      # [num_rows, 288] uint8 output (fp4x2 packed)
    scale_ptr,       # [num_rows, NUM_SCALES] uint8 output (E8M0)
    stride_in_row,
    stride_out_row,
    stride_sc_row,
    NUM_ROWS: tl.constexpr,
    ROW_DIM: tl.constexpr,       # 576
    BLOCK_SIZE: tl.constexpr,    # 32
    NUM_BLOCKS: tl.constexpr,    # 18
    HALF_BLOCK: tl.constexpr,    # 16
):
    """
    Fused MXFP4 quantization for Q tensor.
    One program per row. Processes all 18 blocks of 32 elements.

    E2M1 format (4 bits): 1 sign + 2 exponent + 1 mantissa
    Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
    Packing: fp4x2 = low nibble (even elem) | high nibble (odd elem) << 4
    """
    row_id = tl.program_id(0)
    if row_id >= NUM_ROWS:
        return

    in_base = row_id * stride_in_row
    out_base = row_id * stride_out_row
    sc_base = row_id * stride_sc_row

    for blk in tl.static_range(NUM_BLOCKS):
        blk_elem_start = blk * BLOCK_SIZE
        blk_byte_start = blk * HALF_BLOCK

        # Load 32 elements as 16 pairs
        pair_offs = tl.arange(0, HALF_BLOCK)
        even_offs = blk_elem_start + pair_offs * 2
        odd_offs = blk_elem_start + pair_offs * 2 + 1

        even_mask = even_offs < ROW_DIM
        odd_mask = odd_offs < ROW_DIM

        x_even = tl.load(input_ptr + in_base + even_offs, mask=even_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(input_ptr + in_base + odd_offs, mask=odd_mask, other=0.0).to(tl.float32)

        # Block amax
        amax_even = tl.max(tl.abs(x_even))
        amax_odd = tl.max(tl.abs(x_odd))
        amax = tl.maximum(amax_even, amax_odd)
        amax = tl.maximum(amax, 1e-12)

        # E8M0 scale: 2^e where e = ceil(log2(amax/6.0))
        log2_amax = tl.math.log2(amax)
        e_float = log2_amax - 2.584962500721156  # log2(6)
        e_int = tl.math.ceil(e_float).to(tl.int32)
        e_biased = e_int + 127
        e_biased = tl.maximum(e_biased, 0)
        e_biased = tl.minimum(e_biased, 254)

        tl.store(scale_ptr + sc_base + blk, e_biased.to(tl.uint8))

        scale = tl.math.exp2(e_int.to(tl.float32))

        # Quantize even elements to E2M1
        xs_even = x_even / scale
        xs_even = tl.minimum(tl.maximum(xs_even, -6.0), 6.0)
        sign_e = (xs_even < 0.0).to(tl.int32)
        mag_e = tl.abs(xs_even)

        code_e = tl.zeros_like(mag_e).to(tl.int32)
        code_e = tl.where(mag_e >= 0.25, 1, code_e)
        code_e = tl.where(mag_e >= 0.75, 2, code_e)
        code_e = tl.where(mag_e >= 1.25, 3, code_e)
        code_e = tl.where(mag_e >= 1.75, 4, code_e)
        code_e = tl.where(mag_e >= 2.5, 5, code_e)
        code_e = tl.where(mag_e >= 3.5, 6, code_e)
        code_e = tl.where(mag_e >= 5.0, 7, code_e)
        nibble_e = (sign_e << 3) | code_e

        # Quantize odd elements to E2M1
        xs_odd = x_odd / scale
        xs_odd = tl.minimum(tl.maximum(xs_odd, -6.0), 6.0)
        sign_o = (xs_odd < 0.0).to(tl.int32)
        mag_o = tl.abs(xs_odd)

        code_o = tl.zeros_like(mag_o).to(tl.int32)
        code_o = tl.where(mag_o >= 0.25, 1, code_o)
        code_o = tl.where(mag_o >= 0.75, 2, code_o)
        code_o = tl.where(mag_o >= 1.25, 3, code_o)
        code_o = tl.where(mag_o >= 1.75, 4, code_o)
        code_o = tl.where(mag_o >= 2.5, 5, code_o)
        code_o = tl.where(mag_o >= 3.5, 6, code_o)
        code_o = tl.where(mag_o >= 5.0, 7, code_o)
        nibble_o = (sign_o << 3) | code_o

        # Pack: low nibble = even element, high nibble = odd element
        packed_byte = (nibble_e | (nibble_o << 4)).to(tl.uint8)

        byte_offs = blk_byte_start + pair_offs
        tl.store(packed_ptr + out_base + byte_offs, packed_byte, mask=byte_offs < 288)


def _get_mxfp4_q_buffers(num_rows, device):
    key = num_rows
    if key not in _mxfp4_q_cache:
        _mxfp4_q_cache[key] = {
            "packed": torch.empty((num_rows, PACKED_QK), dtype=torch.uint8, device=device),
            "scale": torch.empty((num_rows, NUM_SCALES), dtype=torch.uint8, device=device),
        }
    return _mxfp4_q_cache[key]


def triton_mxfp4_quant(q_2d: torch.Tensor):
    """Quantize Q to MXFP4 using fused Triton kernel."""
    num_rows = q_2d.shape[0]
    bufs = _get_mxfp4_q_buffers(num_rows, q_2d.device)

    _mxfp4_quant_fused[(num_rows,)](
        q_2d, bufs["packed"], bufs["scale"],
        q_2d.stride(0), bufs["packed"].stride(0), bufs["scale"].stride(0),
        NUM_ROWS=num_rows,
        ROW_DIM=QK_HEAD_DIM,
        BLOCK_SIZE=32,
        NUM_BLOCKS=NUM_SCALES,
        HALF_BLOCK=16,
    )
    return bufs["packed"], bufs["scale"]


def quantize_q_mxfp4(q_2d: torch.Tensor):
    """Dispatch to Triton or AITER MXFP4 quant depending on flag."""
    if USE_TRITON_QUANT:
        return triton_mxfp4_quant(q_2d)
    else:
        q_packed_raw, q_scale_raw = dynamic_mxfp4_quant(q_2d)
        return q_packed_raw.view(torch.uint8), q_scale_raw.view(torch.uint8)


# ===============================================================
# FUSED FP8 QUANTIZATION (for AITER path)
# ===============================================================

@triton.jit
def _amax_kernel(input_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_amax = tl.max(tl.abs(x))
    tl.atomic_max(amax_ptr, local_amax)


@triton.jit
def _quantize_fp8_kernel(input_ptr, output_ptr, amax_ptr, scale_ptr,
                         fp8_max, fp8_min, N, BLOCK: tl.constexpr):
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


def _get_fp8_buffers(num_elements, shape, device):
    key = (num_elements, device)
    if key not in _fp8_buf_cache:
        _fp8_buf_cache[key] = {
            "fp8_out": torch.empty(num_elements, dtype=FP8_DTYPE, device=device),
            "scale_out": torch.empty(1, dtype=torch.float32, device=device),
            "amax_buf": torch.zeros(1, dtype=torch.float32, device=device),
        }
    return _fp8_buf_cache[key]


def fused_quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    flat = tensor.reshape(-1)
    N = flat.numel()
    bufs = _get_fp8_buffers(N, tensor.shape, tensor.device)
    bufs["amax_buf"].zero_()
    BLOCK = 1024
    grid_size = (N + BLOCK - 1) // BLOCK
    _amax_kernel[(grid_size,)](flat, bufs["amax_buf"], N, BLOCK=BLOCK)
    _quantize_fp8_kernel[(grid_size,)](
        flat, bufs["fp8_out"], bufs["amax_buf"], bufs["scale_out"],
        finfo.max, finfo.min, N, BLOCK=BLOCK,
    )
    return bufs["fp8_out"].view(tensor.shape), bufs["scale_out"]


# ===============================================================
# MXFP4 STAGE 1 (multi-split, with v_chunks)
# ===============================================================

@triton.jit
def _mla_mxfp4_stage1(
    Q_packed_ptr, Q_scale_ptr,
    K_packed_ptr, K_scale_ptr,
    V_bf16_ptr,
    Partial_O_ptr, Partial_m_ptr, Partial_l_ptr,
    kv_indptr_ptr,
    stride_q_packed, stride_q_scale,
    stride_kv_packed, stride_kv_scale, stride_v_tok,
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
                mask=(q_d_offs[None, :] < 288), other=0,
            )
            qs_offs = k_scale_start + tl.arange(0, 4)
            q_scale_chunk = tl.load(
                Q_scale_ptr + (q_row_base + offs_m[:, None]) * stride_q_scale + qs_offs[None, :],
                mask=(qs_offs[None, :] < 18), other=0,
            )
            k_d_offs = k_packed_start + tl.arange(0, 64)
            k_chunk = tl.load(
                K_packed_ptr + kv_idx[None, :] * stride_kv_packed + k_d_offs[:, None],
                mask=mask_kv[None, :] & (k_d_offs[:, None] < 288), other=0,
            )
            ks_offs = k_scale_start + tl.arange(0, 4)
            k_scale_chunk = tl.load(
                K_scale_ptr + kv_idx[:, None] * stride_kv_scale + ks_offs[None, :],
                mask=mask_kv[:, None] & (ks_offs[None, :] < 18), other=0,
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
            mask=mask_kv[:, None], other=0.0,
        )
        acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

    po_base = Partial_O_ptr + pid_b * stride_po_b + pid_s * stride_po_s + vd_start
    head_offs = tl.arange(0, NUM_HEADS)
    v_offs = tl.arange(0, V_CHUNK_D)
    tl.store(po_base + head_offs[:, None] * stride_po_h + v_offs[None, :], acc)
    if pid_v == 0:
        ml_base = pid_b * stride_ml_b + pid_s * stride_ml_s
        tl.store(Partial_m_ptr + ml_base + head_offs * stride_ml_h, m_prev)
        tl.store(Partial_l_ptr + ml_base + head_offs * stride_ml_h, l_prev)


# ===============================================================
# FUSED SINGLE-SPLIT KERNEL (splits=1, no reduce)
# ===============================================================

@triton.jit
def _mla_mxfp4_fused_single(
    Q_packed_ptr, Q_scale_ptr,
    K_packed_ptr, K_scale_ptr,
    V_bf16_ptr,
    O_ptr,            # [batch, 16, V_DIM] bf16 -- FINAL output
    kv_indptr_ptr,
    stride_q_packed, stride_q_scale,
    stride_kv_packed, stride_kv_scale, stride_v_tok,
    stride_o_batch, stride_o_head,
    sm_scale,
    BLOCK_N: tl.constexpr,
    V_CHUNK_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """
    Fused single-split: computes full attention and writes bf16 output directly.
    Grid: (batch, 1, num_v_chunks)
    """
    LOG2E: tl.constexpr = 1.4426950408889634

    pid_b = tl.program_id(0)
    pid_v = tl.program_id(2)

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    q_row_base = pid_b * NUM_HEADS
    offs_m = tl.arange(0, NUM_HEADS)
    vd_start = pid_v * V_CHUNK_D

    m_prev = tl.full([NUM_HEADS], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([NUM_HEADS, V_CHUNK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(kv_len, BLOCK_N)
    for tile_idx in range(num_tiles):
        tile_start = tile_idx * BLOCK_N
        kv_offsets = tile_start + tl.arange(0, BLOCK_N)
        mask_kv = kv_offsets < kv_len
        kv_idx = kv_start + kv_offsets

        qk = tl.zeros([NUM_HEADS, BLOCK_N], dtype=tl.float32)
        for k_tile in tl.static_range(5):
            k_packed_start = k_tile * 64
            k_scale_start = k_tile * 4

            q_d_offs = k_packed_start + tl.arange(0, 64)
            q_chunk = tl.load(
                Q_packed_ptr + (q_row_base + offs_m[:, None]) * stride_q_packed + q_d_offs[None, :],
                mask=(q_d_offs[None, :] < 288), other=0,
            )
            qs_offs = k_scale_start + tl.arange(0, 4)
            q_scale_chunk = tl.load(
                Q_scale_ptr + (q_row_base + offs_m[:, None]) * stride_q_scale + qs_offs[None, :],
                mask=(qs_offs[None, :] < 18), other=0,
            )
            k_d_offs = k_packed_start + tl.arange(0, 64)
            k_chunk = tl.load(
                K_packed_ptr + kv_idx[None, :] * stride_kv_packed + k_d_offs[:, None],
                mask=mask_kv[None, :] & (k_d_offs[:, None] < 288), other=0,
            )
            ks_offs = k_scale_start + tl.arange(0, 4)
            k_scale_chunk = tl.load(
                K_scale_ptr + kv_idx[:, None] * stride_kv_scale + ks_offs[None, :],
                mask=mask_kv[:, None] & (ks_offs[None, :] < 18), other=0,
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
            mask=mask_kv[:, None], other=0.0,
        )
        acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

    # Normalize and write final output directly
    acc = acc / (l_prev[:, None] + 1e-10)
    head_offs = tl.arange(0, NUM_HEADS)
    v_offs = tl.arange(0, V_CHUNK_D)
    o_base = O_ptr + pid_b * stride_o_batch + vd_start
    tl.store(
        o_base + head_offs[:, None] * stride_o_head + v_offs[None, :],
        acc.to(tl.bfloat16),
    )


# ===============================================================
# STAGE 2: REDUCE
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
# BUFFER CACHES
# ===============================================================

def _mxfp4_get_buffers(batch_size, num_kv_splits, device):
    key = (batch_size, num_kv_splits)
    if key not in _mxfp4_buf_cache:
        _mxfp4_buf_cache[key] = {
            "partial_o": torch.empty(
                (batch_size, num_kv_splits, NUM_HEADS, V_HEAD_DIM),
                dtype=torch.float32, device=device),
            "partial_m": torch.empty(
                (batch_size, num_kv_splits, NUM_HEADS),
                dtype=torch.float32, device=device),
            "partial_l": torch.empty(
                (batch_size, num_kv_splits, NUM_HEADS),
                dtype=torch.float32, device=device),
            "output": torch.empty(
                (batch_size, NUM_HEADS, V_HEAD_DIM),
                dtype=torch.bfloat16, device=device),
        }
    return _mxfp4_buf_cache[key]


def _mxfp4_get_output(batch_size, device):
    key = ("fused", batch_size)
    if key not in _mxfp4_buf_cache:
        _mxfp4_buf_cache[key] = torch.empty(
            (batch_size, NUM_HEADS, V_HEAD_DIM),
            dtype=torch.bfloat16, device=device,
        )
    return _mxfp4_buf_cache[key]


# ===============================================================
# MXFP4 DECODE PATHS
# ===============================================================

def _mxfp4_path_fused(q, kv_data, kv_indptr, config):
    """Fused single-split for (4,1024). No reduce kernel."""
    batch_size = config["batch_size"]

    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    q_2d = q.view(-1, QK_HEAD_DIM)
    q_packed, q_scale = quantize_q_mxfp4(q_2d)

    kv_fp4_2d = kv_fp4.reshape(-1, PACKED_QK).view(torch.uint8)
    kv_scale_2d = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)

    BLOCK_N = 64
    V_CHUNK_D = 128
    num_v_chunks = V_HEAD_DIM // V_CHUNK_D

    output = _mxfp4_get_output(batch_size, q.device)

    _mla_mxfp4_fused_single[(batch_size, 1, num_v_chunks)](
        q_packed, q_scale,
        kv_fp4_2d, kv_scale_2d, v_bf16_2d,
        output, kv_indptr,
        q_packed.stride(0), q_scale.stride(0),
        kv_fp4_2d.stride(0), kv_scale_2d.stride(0), v_bf16_2d.stride(0),
        output.stride(0), output.stride(1),
        SM_SCALE,
        BLOCK_N=BLOCK_N, V_CHUNK_D=V_CHUNK_D, NUM_HEADS=NUM_HEADS,
    )
    return output


def _mxfp4_path(q, kv_data, kv_indptr, config):
    """Multi-split MXFP4 path with Triton Q quant."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    num_kv_splits = MXFP4_KV_SPLITS_MAP.get(
        (batch_size, kv_seq_len), MXFP4_DEFAULT_KV_SPLITS)

    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]

    q_2d = q.view(-1, QK_HEAD_DIM)
    q_packed, q_scale = quantize_q_mxfp4(q_2d)

    kv_fp4_2d = kv_fp4.reshape(-1, PACKED_QK).view(torch.uint8)
    kv_scale_2d = kv_scale.view(torch.uint8) if kv_scale.dtype != torch.uint8 else kv_scale
    v_bf16_2d = kv_bf16.view(-1, QK_HEAD_DIM)

    BLOCK_N = 64
    V_CHUNK_D = 128
    num_v_chunks = V_HEAD_DIM // V_CHUNK_D

    bufs = _mxfp4_get_buffers(batch_size, num_kv_splits, q.device)

    _mla_mxfp4_stage1[(batch_size * num_kv_splits, 1, num_v_chunks)](
        q_packed, q_scale,
        kv_fp4_2d, kv_scale_2d, v_bf16_2d,
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"],
        kv_indptr,
        q_packed.stride(0), q_scale.stride(0),
        kv_fp4_2d.stride(0), kv_scale_2d.stride(0), v_bf16_2d.stride(0),
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        SM_SCALE,
        BLOCK_N=BLOCK_N, V_CHUNK_D=V_CHUNK_D,
        NUM_KV_SPLITS=num_kv_splits, NUM_HEADS=NUM_HEADS,
    )

    _mla_mxfp4_reduce[(batch_size, NUM_HEADS, num_v_chunks)](
        bufs["partial_o"], bufs["partial_m"], bufs["partial_l"], bufs["output"],
        bufs["partial_o"].stride(0), bufs["partial_o"].stride(1), bufs["partial_o"].stride(2),
        bufs["partial_m"].stride(0), bufs["partial_m"].stride(1), bufs["partial_m"].stride(2),
        bufs["output"].stride(0), bufs["output"].stride(1),
        NUM_KV_SPLITS=num_kv_splits, V_CHUNK_D=V_CHUNK_D, NUM_HEADS=NUM_HEADS,
    )
    return bufs["output"]


# ===============================================================
# AITER PATH
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


def _aiter_path(q, kv_data, qo_indptr, kv_indptr, config):
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]
    num_kv_splits = AITER_KV_SPLITS_MAP.get((bs, kvlen), AITER_DEFAULT_KV_SPLITS)

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

    num_partials = meta["reduce_partial_map"].size(0)
    logits = torch.empty(
        (num_partials, 1, NUM_HEADS, V_HEAD_DIM),
        dtype=torch.float32, device="cuda",
    )
    attn_lse = torch.empty(
        (num_partials, 1, NUM_HEADS, 1),
        dtype=torch.float32, device="cuda",
    )

    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d, qo_indptr, kv_indptr,
        meta["kv_indices"], meta["kv_last_page_len"],
        None,
        meta["work_meta_data"], meta["work_indptr"], meta["work_info_set"],
        1, PAGE_SIZE, NUM_KV_HEADS, SM_SCALE,
        logits, attn_lse, o, q_scale, kv_scale,
    )

    aiter.mla_reduce_v1(
        logits, attn_lse,
        meta["reduce_indptr"], meta["reduce_final_map"], meta["reduce_partial_map"],
        1, o, None,
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

    if (bs, kvlen) in MXFP4_SINGLE_SPLIT_CONFIGS:
        return _mxfp4_path_fused(q, kv_data, kv_indptr, config)
    elif (bs, kvlen) in MXFP4_CONFIGS:
        return _mxfp4_path(q, kv_data, kv_indptr, config)
    else:
        return _aiter_path(q, kv_data, qo_indptr, kv_indptr, config)
