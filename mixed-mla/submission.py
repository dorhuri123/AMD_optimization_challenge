"""
mixed-mla — custom_kernel submission
=====================================
DeepSeek R1 forward_absorb MLA decode, AMD MI355X.
Matches the reference fp8 Q + fp8 KV path.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                  # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE = 1
NUM_KV_SPLITS = 32
FP8_DTYPE = aiter_dtypes.fp8


def _quant_fp8(x):
    finfo = torch.finfo(FP8_DTYPE)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (x / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)


def _make_metadata(batch_size, max_q_len, nq, nkv, q_dtype, kv_dtype,
                   qo_indptr, kv_indptr, kv_last_page_len, num_kv_splits):
    meta_info = get_mla_metadata_info_v1(
        batch_size=batch_size,
        max_q_len=max_q_len,
        nq=nq,
        nkv=nkv,
        hq=QK_HEAD_DIM,
        hv=V_HEAD_DIM,
        num_kv_splits=num_kv_splits,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    return get_mla_metadata_v1(
        meta_info=meta_info,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        page_size=PAGE_SIZE,
        num_kv_splits=num_kv_splits,
    )


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]

    # FP8 Q + FP8 KV path (matches reference)
    q_fp8, q_scale = _quant_fp8(q)
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    total_kv = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    meta = _make_metadata(
        batch_size, q_seq_len, nq, nkv,
        q_fp8.dtype, kv_buffer_fp8.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=NUM_KV_SPLITS,
    )

    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")

    mla_decode_fwd(
        q_fp8.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        sm_scale=SM_SCALE,
        page_size=PAGE_SIZE,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale,
        meta=meta,
    )

    return o
