"""
Phase 1: Custom HIP MLA decode kernel -- CORRECTNESS FIRST.

Single-thread-per-(batch,head) implementation. Extremely slow but validates:
1. load_inline compilation pipeline on MI355X (gfx950)
2. FP8 e4m3fnuz dequantization
3. Online softmax + V accumulation
4. Correct output shape/dtype matching reference

The HIP kernel is embedded as a string and compiled via load_inline at import time.
Falls back to AITER if compilation fails.
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = 576   # KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM: int = 512    # KV_LORA_RANK
SM_SCALE: float = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE: int = 1


# ═══════════════════════════════════════════════════════════════
# HIP KERNEL + TORCH WRAPPER (all in cuda_sources for ROCm)
#
# Everything that needs __global__ or hipLaunchKernelGGL must be
# in the same compilation unit. The torch C++ wrapper function
# is defined here too so it can directly call the kernel.
# ═══════════════════════════════════════════════════════════════

hip_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------
// FP8 E4M3FNUZ dequantization (AMD format)
// E4M3FNUZ: 1 sign, 4 exponent, 3 mantissa
// Bias = 8, 0x80 = NaN (treat as 0), no negative zero
// Max representable = 240
// ---------------------------------------------------------------
__device__ __forceinline__ float dequant_fp8_e4m3fnuz(unsigned char val) {
    if (val == 0) return 0.0f;
    if (val == 0x80) return 0.0f;  // fnuz: 0x80 is NaN, treat as 0

    int sign = (val >> 7) & 1;
    int exp_bits = (val >> 3) & 0xF;
    int mant_bits = val & 0x7;

    float mantissa;
    float result;

    if (exp_bits == 0) {
        // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mantissa)
        mantissa = (float)mant_bits / 8.0f;
        result = ldexpf(mantissa, 1 - 8);  // * 2^(-7)
    } else {
        // Normal: value = (-1)^sign * 2^(exp-bias) * (1.mantissa)
        mantissa = 1.0f + (float)mant_bits / 8.0f;
        result = ldexpf(mantissa, exp_bits - 8);
    }

    return sign ? -result : result;
}


// ---------------------------------------------------------------
// Phase 1 MLA decode kernel
// One thread per (batch, head) pair -- simple but correct
//
// Algorithm:
//   For each KV token, compute QK dot product over 576 dims,
//   apply online softmax, accumulate weighted V (first 512 dims).
// ---------------------------------------------------------------
__global__ void mla_decode_phase1(
    const float* __restrict__ Q_float,       // [batch*16, 576]
    const unsigned char* __restrict__ KV_fp8, // [total_kv, 576] raw fp8 bytes
    float* __restrict__ output,              // [batch*16, 512]
    const int* __restrict__ kv_indptr,       // [batch+1]
    float kv_scale,
    float sm_scale,
    int batch_size,
    int qk_dim,                             // 576
    int v_dim                               // 512
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = batch_size * 16;
    if (idx >= total_heads) return;

    int batch = idx / 16;

    int kv_start = kv_indptr[batch];
    int kv_end = kv_indptr[batch + 1];

    // Pointer to this head's query (576 floats)
    const float* q = Q_float + (long long)idx * qk_dim;

    // Online softmax state
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // V accumulator -- 512 floats
    // Large register usage but only batch*16 threads total
    float acc[512];
    for (int v = 0; v < 512; v++) {
        acc[v] = 0.0f;
    }

    float combined_scale = kv_scale * sm_scale;

    for (int k = kv_start; k < kv_end; k++) {
        // KV layout: [total_kv, 1, 576] stored contiguously as [total_kv, 576]
        const unsigned char* kv_row = KV_fp8 + (long long)k * qk_dim;

        // Compute QK dot product over all 576 dims
        float score = 0.0f;
        for (int d = 0; d < qk_dim; d++) {
            float kv_val = dequant_fp8_e4m3fnuz(kv_row[d]);
            score += q[d] * kv_val;
        }
        score *= combined_scale;

        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float old_scale_factor = expf(max_score - new_max);
        float new_weight = expf(score - new_max);

        // Rescale existing accumulator
        for (int v = 0; v < 512; v++) {
            acc[v] *= old_scale_factor;
        }
        sum_exp = sum_exp * old_scale_factor + new_weight;
        max_score = new_max;

        // Accumulate weighted V (first 512 dims of KV row)
        for (int v = 0; v < 512; v++) {
            float v_val = dequant_fp8_e4m3fnuz(kv_row[v]);
            acc[v] += new_weight * v_val;
        }
    }

    // Normalize and write output
    // V values were dequantized without kv_scale, so multiply here
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    float* out_row = output + (long long)idx * v_dim;
    for (int v = 0; v < 512; v++) {
        out_row[v] = acc[v] * inv_sum * kv_scale;
    }
}


// ---------------------------------------------------------------
// Torch C++ wrapper -- called from Python via pybind11
// ---------------------------------------------------------------
torch::Tensor mla_hip_forward(
    torch::Tensor Q,          // [batch, 16, 576] bfloat16
    torch::Tensor KV_fp8,     // [total_kv, 576] int8 (raw fp8 bytes)
    torch::Tensor kv_indptr,  // [batch+1] int32
    float kv_scale,
    float sm_scale,
    int batch_size
) {
    const int qk_dim = 576;
    const int v_dim = 512;
    const int num_heads = 16;

    // Convert Q from bf16 to float, reshape to [batch*16, 576]
    auto Q_float = Q.to(torch::kFloat32).contiguous().view({-1, qk_dim});

    // KV bytes: ensure contiguous [total_kv, 576]
    auto KV_bytes = KV_fp8.contiguous().view({-1, qk_dim});

    // Allocate float output [batch*16, 512]
    int total_heads = batch_size * num_heads;
    auto output = torch::zeros({total_heads, v_dim},
                               torch::dtype(torch::kFloat32).device(Q.device()));

    // Launch: one thread per (batch, head)
    int threads = 256;
    int blocks = (total_heads + threads - 1) / threads;

    hipLaunchKernelGGL(
        mla_decode_phase1,
        dim3(blocks), dim3(threads), 0, 0,
        Q_float.data_ptr<float>(),
        (const unsigned char*)KV_bytes.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        kv_indptr.data_ptr<int>(),
        kv_scale,
        sm_scale,
        batch_size,
        qk_dim,
        v_dim
    );

    // Reshape to [batch, 16, 512] and convert to bf16
    return output.view({batch_size, num_heads, v_dim}).to(torch::kBFloat16);
}
"""


# ═══════════════════════════════════════════════════════════════
# C++ SOURCES (minimal -- just declares the function for pybind)
# ═══════════════════════════════════════════════════════════════

cpp_source = r"""
#include <torch/extension.h>

// Forward declaration -- implemented in cuda_sources (HIP)
torch::Tensor mla_hip_forward(
    torch::Tensor Q,
    torch::Tensor KV_fp8,
    torch::Tensor kv_indptr,
    float kv_scale,
    float sm_scale,
    int batch_size
);
"""


# ═══════════════════════════════════════════════════════════════
# COMPILE HIP MODULE
# ═══════════════════════════════════════════════════════════════

_hip_module = None
_hip_available = False


def _try_compile():
    global _hip_module, _hip_available
    if _hip_module is not None:
        return _hip_available
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_module = load_inline(
            name="mla_hip_phase1",
            cpp_sources=cpp_source,
            cuda_sources=hip_source,  # ROCm maps cuda_sources -> HIP
            functions=["mla_hip_forward"],
            verbose=False,
            extra_cuda_cflags=["-O2"],
        )
        _hip_available = True
        print("[HIP Phase 1] Compilation SUCCESS")
    except Exception as e:
        _hip_available = False
        print(f"[HIP Phase 1] Compilation FAILED: {e}")
        import traceback
        traceback.print_exc()
    return _hip_available


# Compile at import time
_try_compile()


# ═══════════════════════════════════════════════════════════════
# AITER FALLBACK
# ═══════════════════════════════════════════════════════════════

_aiter_available = False
try:
    from aiter.mla import mla_decode_fwd
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    _aiter_available = True
    FP8_DTYPE = aiter_dtypes.fp8
except ImportError:
    FP8_DTYPE = torch.float8_e4m3fnuz

_meta_cache = {}


def _aiter_fallback(data):
    """Standard AITER decode path as fallback."""
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]

    finfo = torch.finfo(FP8_DTYPE)
    amax = q.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    q_fp8 = (q / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    q_scale = scale.to(torch.float32).reshape(1)

    kv_fp8, kv_scale = kv_data["fp8"]
    total_kv = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    num_kv_splits = 16
    key = (bs, num_kv_splits, q_fp8.dtype, kv_fp8.dtype)
    if key not in _meta_cache:
        info = get_mla_metadata_info_v1(
            bs, 1, NUM_HEADS, q_fp8.dtype, kv_fp8.dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=q_fp8.dtype, dtype_kv=kv_fp8.dtype,
        )
        _meta_cache[key] = {
            "work_meta_data": wm, "work_indptr": wi, "work_info_set": wis,
            "reduce_indptr": ri, "reduce_final_map": rfm, "reduce_partial_map": rpm,
            "kv_indices": kv_indices, "kv_last_page_len": kv_last_page_len,
        }
    meta = _meta_cache[key]

    o = torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])

    mla_decode_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d, o,
        qo_indptr, kv_indptr,
        meta["kv_indices"], meta["kv_last_page_len"],
        1, page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS,
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


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]

    if not _hip_available:
        print("[HIP Phase 1] Using AITER fallback")
        return _aiter_fallback(data)

    # Extract FP8 KV data and scale
    kv_fp8, kv_scale = kv_data["fp8"]

    # kv_scale is a scalar tensor (shape [1])
    kv_scale_val = kv_scale.item()

    # KV_fp8 is [total_kv, 1, 576] fp8_e4m3fnuz
    # View as raw int8 bytes for the HIP kernel
    kv_bytes = kv_fp8.view(torch.int8).contiguous()

    # Q is [batch, 16, 576] bf16
    # kv_indptr is [batch+1] int32

    output = _hip_module.mla_hip_forward(
        q,              # [batch, 16, 576] bf16
        kv_bytes,       # [total_kv, 576] as int8
        kv_indptr,      # [batch+1] int32
        kv_scale_val,   # float scalar
        SM_SCALE,       # float scalar
        bs,             # int
    )

    return output
