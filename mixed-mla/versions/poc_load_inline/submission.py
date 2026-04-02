"""
POC: Test if torch.utils.cpp_extension.load_inline works on MI355X popcorn runner.

This submission uses AITER for actual computation but embeds a trivial HIP kernel
via load_inline to verify the compilation pipeline works.
If this passes, we can build a full custom HIP MLA decode kernel.
"""

import os
import torch
from task import input_t, output_t

# Must set arch before any HIP compilation
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

# ═══════════════════════════════════════════════════════════
# POC: Test load_inline compilation
# ═══════════════════════════════════════════════════════════

_hip_module = None
_hip_available = False

def _try_load_inline():
    """Try to compile a trivial HIP kernel. Returns True if successful."""
    global _hip_module, _hip_available
    if _hip_module is not None:
        return _hip_available
    try:
        from torch.utils.cpp_extension import load_inline

        cpp_source = """
        torch::Tensor hip_test(torch::Tensor input) {
            return input.clone();
        }
        """

        hip_source = """
        // Trivial HIP kernel -- just copies data
        __global__ void copy_kernel(const float* __restrict__ in,
                                     float* __restrict__ out, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] = in[i];
        }
        """

        _hip_module = load_inline(
            name="hip_poc_test",
            cpp_sources=cpp_source,
            cuda_sources=hip_source,  # ROCm uses cuda_sources for HIP
            functions=["hip_test"],
            verbose=False,
        )
        _hip_available = True
        print("[POC] load_inline SUCCESS on MI355X!")
    except Exception as e:
        _hip_available = False
        print(f"[POC] load_inline FAILED: {e}")
    return _hip_available


# ═══════════════════════════════════════════════════════════
# Standard AITER path (fallback and actual computation)
# ═══════════════════════════════════════════════════════════

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE = 1
FP8_DTYPE = aiter_dtypes.fp8

KV_SPLITS_MAP = {
    (4, 1024): 16, (4, 8192): 32,
    (32, 1024): 16, (32, 8192): 48,
    (64, 1024): 16, (64, 8192): 24,
    (256, 1024): 16, (256, 8192): 24,
}
DEFAULT_KV_SPLITS = 16

_meta_cache = {}
_alloc_cache = {}


def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


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


# Try load_inline on first import
_try_load_inline()


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]
    num_kv_splits = KV_SPLITS_MAP.get((bs, kvlen), DEFAULT_KV_SPLITS)

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
