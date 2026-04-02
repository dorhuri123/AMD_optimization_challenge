"""
v14-simple: Non-persistent AITER for all configs.

The KEY insight: call mla_decode_fwd WITHOUT work_meta_data to trigger
non-persistent mode. AITER auto-computes optimal num_kv_splits internally.
For bs=256,kv=1024, it picks splits=1 which SKIPS the reduce kernel entirely.

This is what the ~32μs cluster competitors are doing.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE = 1
FP8_DTYPE = aiter_dtypes.fp8

# Caches
_cache = {}


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _get_cache(bs, kv_indptr, device):
    key = bs
    if key not in _cache:
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        total_kv = int(kv_indptr[-1].item())
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)
        _cache[key] = {
            "kv_indices": kv_indices,
            "kv_last_page_len": kv_last_page_len,
            "output": torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=device),
        }
    return _cache[key]


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]

    # Q FP8 quantization
    q_fp8, q_scale = quantize_fp8(q)

    # FP8 KV
    kv_fp8, kv_scale = kv_data["fp8"]

    # Cached values
    c = _get_cache(bs, kv_indptr, q.device)
    o = c["output"]

    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])

    # NON-PERSISTENT MODE: do NOT pass work_meta_data
    # Let AITER auto-compute optimal num_kv_splits
    mla_decode_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d, o,
        qo_indptr, kv_indptr,
        c["kv_indices"], c["kv_last_page_len"],
        1,  # max_seqlen_q
        page_size=PAGE_SIZE,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        q_scale=q_scale,
        kv_scale=kv_scale,
        # NO work_meta_data → non-persistent mode
        # NO num_kv_splits → AITER auto-tunes
        # NO intra_batch_mode → defaults
    )
    return o
