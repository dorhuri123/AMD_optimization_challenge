"""
Optimized MLA decode submission.

Optimization v1 — two low-hanging fruit over the reference:
  1. Adaptive num_kv_splits per (batch_size, kv_seq_len) config
     Reference hardcodes 32; optimal value depends on total KV tokens vs CU count.
  2. Adaptive Q dtype: skip fp8 Q quantization for small batches (a16w8 path)
     where the quantization kernel launch overhead exceeds the BW savings.

Both use the same AITER persistent kernel — no custom Triton needed yet.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# ---------------------------------------------------------------------------
# Constants (same as reference)
# ---------------------------------------------------------------------------
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE = 1
FP8_DTYPE = aiter_dtypes.fp8

# ---------------------------------------------------------------------------
# Tuning parameters — FILL IN after running sweep_kv_splits.py on MI355X
# ---------------------------------------------------------------------------
# Map (batch_size, kv_seq_len) -> optimal num_kv_splits
# Default to 32 (same as reference) until we have profiling data.
KV_SPLITS_MAP = {
    # (bs, kvlen): splits
    (4, 1024): 32,
    (4, 8192): 32,
    (32, 1024): 32,
    (32, 8192): 32,
    (64, 1024): 32,
    (64, 8192): 32,
    (256, 1024): 32,
    (256, 8192): 32,
}
DEFAULT_KV_SPLITS = 32

# Batch size threshold: use bf16 Q (a16w8) below this, fp8 Q (a8w8) above.
# Set to 0 to always use fp8 Q (same as reference).
# TODO: tune on MI355X — try 16 or 32
Q_FP8_BS_THRESHOLD = 0


# ---------------------------------------------------------------------------
# FP8 quantization (same as reference)
# ---------------------------------------------------------------------------
def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# Persistent mode metadata (parameterized on num_kv_splits)
# ---------------------------------------------------------------------------
def _make_mla_decode_metadata(
    batch_size, max_q_len, nhead, nhead_kv,
    q_dtype, kv_dtype,
    qo_indptr, kv_indptr, kv_last_page_len,
    num_kv_splits,
):
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv, nhead_kv, True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


# ---------------------------------------------------------------------------
# Core MLA decode with configurable splits
# ---------------------------------------------------------------------------
def _mla_decode(
    q, kv_buffer, qo_indptr, kv_indptr, config,
    q_scale=None, kv_scale=None, num_kv_splits=DEFAULT_KV_SPLITS,
):
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    total_kv_len = int(kv_indptr[-1].item())

    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])
    max_q_len = q_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    meta = _make_mla_decode_metadata(
        batch_size, max_q_len, nq, nkv,
        q.dtype, kv_buffer.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=num_kv_splits,
    )

    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]

    # Pick optimal num_kv_splits for this config
    num_kv_splits = KV_SPLITS_MAP.get((bs, kvlen), DEFAULT_KV_SPLITS)

    # Adaptive Q dtype: skip fp8 quantization for small batches
    use_fp8_q = (Q_FP8_BS_THRESHOLD > 0 and bs >= Q_FP8_BS_THRESHOLD) or \
                (Q_FP8_BS_THRESHOLD == 0)

    if use_fp8_q:
        q_input, q_scale = quantize_fp8(q)
    else:
        q_input, q_scale = q, None

    # Always use fp8 KV (pre-quantized, no overhead)
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    return _mla_decode(
        q_input, kv_buffer_fp8, qo_indptr, kv_indptr, config,
        q_scale=q_scale, kv_scale=kv_scale,
        num_kv_splits=num_kv_splits,
    )
