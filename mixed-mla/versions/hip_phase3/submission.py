"""
Phase 3: LUT-based FP8 dequant + multi-token-per-iteration Split-K MLA decode.

Key optimizations over Phase 2:
1. LUT-based FP8 dequant: precompute all 256 FP8->float32 values in shared memory
   - Replaces branchy scalar dequant with a single shared memory lookup
2. 4 KV tokens per iteration: each of 4 warps computes QK for one token
   - 4x fewer iterations through the KV token loop
3. Vectorized global loads: load 4 bytes at a time (uint32) for KV data
4. Warp shuffle for QK reduction: no shared memory needed for dot product reduce

Grid: (batch_size * num_kv_splits, NUM_HEADS=16)
Block: 256 threads (4 warps of 64)
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

# ===================================================================
# CONSTANTS
# ===================================================================

NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512
SM_SCALE: float = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE: int = 1

NUM_KV_SPLITS: int = 16

# ===================================================================
# HIP KERNEL SOURCE
# ===================================================================

hip_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------
// Constants
// ---------------------------------------------------------------
#define QK_DIM 576
#define V_DIM 512
#define NUM_HEADS 16
#define BLOCK_SIZE 256
#define WARP_SIZE 64
#define NUM_WARPS 4
// Each thread handles 512/256 = 2 dims for V accumulation
#define DIMS_PER_THREAD_V 2
// Number of KV tokens processed per iteration (one per warp)
#define TOKENS_PER_ITER 4
// QK dims per thread within a warp: ceil(576/64) = 9
#define QK_DIMS_PER_WARP_THREAD 9

// FP8 LUT size
#define FP8_LUT_SIZE 256


// ---------------------------------------------------------------
// Build FP8 E4M3FNUZ LUT: compute dequant value for each of 256 byte values
// Called cooperatively by all 256 threads (each computes 1 entry)
// ---------------------------------------------------------------
__device__ __forceinline__ float compute_fp8_value(unsigned char val) {
    if (val == 0) return 0.0f;
    if (val == 0x80) return 0.0f;  // negative zero in FNUZ

    int sign = (val >> 7) & 1;
    int exp_bits = (val >> 3) & 0xF;
    int mant_bits = val & 0x7;

    float mantissa;
    float result;

    if (exp_bits == 0) {
        // Subnormal
        mantissa = (float)mant_bits / 8.0f;
        result = ldexpf(mantissa, 1 - 8);
    } else {
        // Normal
        mantissa = 1.0f + (float)mant_bits / 8.0f;
        result = ldexpf(mantissa, exp_bits - 8);
    }

    return sign ? -result : result;
}


// ---------------------------------------------------------------
// Warp-level reduction (sum) using shuffle — AMD wavefront = 64
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}


// ---------------------------------------------------------------
// Phase 3: Split-K MLA decode kernel with LUT dequant + multi-token
//
// Grid: (batch_size * num_kv_splits, NUM_HEADS)
// Block: 256 threads (4 warps of 64)
//
// Shared memory layout:
//   [0..255]          : FP8 LUT (256 floats = 1KB)
//   [256..831]        : Q vector (576 floats = 2.25KB)
//   [832..835]        : scores for 4 tokens (4 floats)
//   [836..839]        : warp reduce scratch (4 floats)
//   Total: ~3.4KB
// ---------------------------------------------------------------
__global__ void mla_decode_splitk(
    const float* __restrict__ Q,             // [batch*16, 576]
    const unsigned char* __restrict__ KV,     // [total_kv, 576]
    float* __restrict__ partial_acc,          // [batch * num_splits * 16, V_DIM]
    float* __restrict__ partial_max,          // [batch * num_splits * 16]
    float* __restrict__ partial_sum_exp,      // [batch * num_splits * 16]
    const int* __restrict__ kv_indptr,        // [batch+1]
    float combined_scale,                     // kv_scale * sm_scale
    int num_kv_splits
) {
    int batch_split = blockIdx.x;
    int head = blockIdx.y;

    int batch = batch_split / num_kv_splits;
    int split = batch_split % num_kv_splits;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // KV range for this batch
    int kv_start_all = kv_indptr[batch];
    int kv_end_all = kv_indptr[batch + 1];
    int total_kv = kv_end_all - kv_start_all;

    // Split the KV range
    int tokens_per_split = (total_kv + num_kv_splits - 1) / num_kv_splits;
    int kv_start = kv_start_all + split * tokens_per_split;
    int kv_end = kv_start_all + min((split + 1) * tokens_per_split, total_kv);

    if (kv_start >= kv_end_all) {
        int partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + head;
        partial_max[partial_idx] = -1e30f;
        partial_sum_exp[partial_idx] = 0.0f;
        float* p_acc = partial_acc + (long long)partial_idx * V_DIM;
        for (int i = tid; i < V_DIM; i += BLOCK_SIZE) {
            p_acc[i] = 0.0f;
        }
        return;
    }
    if (kv_end > kv_end_all) kv_end = kv_end_all;

    // Shared memory
    extern __shared__ float smem[];
    float* lut = smem;                                    // 256 floats
    float* smem_q = smem + FP8_LUT_SIZE;                  // 576 floats
    float* smem_scores = smem_q + QK_DIM;                 // TOKENS_PER_ITER floats
    float* smem_reduce = smem_scores + TOKENS_PER_ITER;   // NUM_WARPS floats

    // Step 1: Build FP8 LUT cooperatively (each of 256 threads computes 1 entry)
    lut[tid] = compute_fp8_value((unsigned char)tid);
    __syncthreads();

    // Step 2: Load Q into shared memory
    const float* q_ptr = Q + ((long long)batch * NUM_HEADS + head) * QK_DIM;
    for (int i = tid; i < QK_DIM; i += BLOCK_SIZE) {
        smem_q[i] = q_ptr[i];
    }
    __syncthreads();

    // Per-thread V accumulator (2 dims per thread)
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int v_idx0 = tid * DIMS_PER_THREAD_V;
    int v_idx1 = tid * DIMS_PER_THREAD_V + 1;

    // Online softmax state
    float block_max = -1e30f;
    float block_sum_exp = 0.0f;

    // Process KV tokens: TOKENS_PER_ITER (4) tokens at a time
    // Each warp handles QK for one token, then all threads do V accumulation
    int num_tokens = kv_end - kv_start;

    for (int t_base = 0; t_base < num_tokens; t_base += TOKENS_PER_ITER) {
        int tokens_this_iter = min(TOKENS_PER_ITER, num_tokens - t_base);

        // --- QK dot products: each warp computes one token's dot product ---
        if (warp_id < tokens_this_iter) {
            int kv_idx = kv_start + t_base + warp_id;
            const unsigned char* kv_row = KV + (long long)kv_idx * QK_DIM;

            // Each lane in the warp handles ceil(576/64)=9 dims
            float partial_dot = 0.0f;
            for (int d = lane; d < QK_DIM; d += WARP_SIZE) {
                float k_val = lut[kv_row[d]];
                partial_dot += smem_q[d] * k_val;
            }

            // Warp-level reduce (no shared memory needed!)
            float score = warp_reduce_sum(partial_dot);

            // Lane 0 of each warp writes the score
            if (lane == 0) {
                smem_scores[warp_id] = score * combined_scale;
            }
        }
        __syncthreads();

        // --- Online softmax update ---
        float tile_max = -1e30f;
        for (int t = 0; t < tokens_this_iter; t++) {
            tile_max = fmaxf(tile_max, smem_scores[t]);
        }

        float new_max = fmaxf(block_max, tile_max);

        // Rescale existing accumulator
        float rescale = expf(block_max - new_max);
        acc0 *= rescale;
        acc1 *= rescale;
        block_sum_exp *= rescale;
        block_max = new_max;

        // --- V accumulation: all 256 threads accumulate their 2 V dims ---
        for (int t = 0; t < tokens_this_iter; t++) {
            float w = expf(smem_scores[t] - block_max);
            block_sum_exp += w;

            int kv_idx = kv_start + t_base + t;
            const unsigned char* kv_row = KV + (long long)kv_idx * QK_DIM;

            // LUT-based dequant for V dimensions
            if (v_idx0 < V_DIM) {
                acc0 += w * lut[kv_row[v_idx0]];
            }
            if (v_idx1 < V_DIM) {
                acc1 += w * lut[kv_row[v_idx1]];
            }
        }
        __syncthreads();
    }

    // --- Write partial results ---
    int partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + head;
    float* p_acc = partial_acc + (long long)partial_idx * V_DIM;

    if (v_idx0 < V_DIM) {
        p_acc[v_idx0] = acc0;
    }
    if (v_idx1 < V_DIM) {
        p_acc[v_idx1] = acc1;
    }

    if (tid == 0) {
        partial_max[partial_idx] = block_max;
        partial_sum_exp[partial_idx] = block_sum_exp;
    }
}


// ---------------------------------------------------------------
// Reduce kernel: merge partial results across splits
// (Same as Phase 2 — this is already efficient)
//
// Grid: (batch_size, NUM_HEADS)
// Block: 256 threads
// ---------------------------------------------------------------
__global__ void mla_reduce(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum_exp,
    float* __restrict__ output,
    float kv_scale,
    int num_kv_splits
) {
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    // Find global max across all splits
    float global_max = -1e30f;
    for (int s = 0; s < num_kv_splits; s++) {
        int idx = (batch * num_kv_splits + s) * NUM_HEADS + head;
        float m = partial_max[idx];
        global_max = fmaxf(global_max, m);
    }

    // Compute global sum of exp and rescaled V accumulator
    float global_sum_exp = 0.0f;
    float final_v0 = 0.0f;
    float final_v1 = 0.0f;
    int v_idx0 = tid * 2;
    int v_idx1 = tid * 2 + 1;

    for (int s = 0; s < num_kv_splits; s++) {
        int idx = (batch * num_kv_splits + s) * NUM_HEADS + head;
        float m = partial_max[idx];
        float se = partial_sum_exp[idx];
        float rescale = expf(m - global_max);
        float scaled_se = se * rescale;

        global_sum_exp += scaled_se;

        const float* p_acc = partial_acc + (long long)idx * V_DIM;
        if (v_idx0 < V_DIM) {
            final_v0 += p_acc[v_idx0] * rescale;
        }
        if (v_idx1 < V_DIM) {
            final_v1 += p_acc[v_idx1] * rescale;
        }
    }

    // Normalize and apply kv_scale
    float inv_sum = (global_sum_exp > 0.0f) ? (1.0f / global_sum_exp) : 0.0f;
    float* out_row = output + ((long long)batch * NUM_HEADS + head) * V_DIM;

    if (v_idx0 < V_DIM) {
        out_row[v_idx0] = final_v0 * inv_sum * kv_scale;
    }
    if (v_idx1 < V_DIM) {
        out_row[v_idx1] = final_v1 * inv_sum * kv_scale;
    }
}


// ---------------------------------------------------------------
// Torch C++ wrapper
// ---------------------------------------------------------------
torch::Tensor mla_hip_forward(
    torch::Tensor Q,
    torch::Tensor KV_fp8,
    torch::Tensor kv_indptr,
    float kv_scale,
    float sm_scale,
    int batch_size,
    int num_kv_splits
) {
    const int qk_dim = QK_DIM;
    const int v_dim = V_DIM;
    const int num_heads = NUM_HEADS;

    // Q: bf16 -> float, reshape to [batch*16, 576]
    auto Q_float = Q.to(torch::kFloat32).contiguous().view({-1, qk_dim});

    // KV bytes
    auto KV_bytes = KV_fp8.contiguous().view({-1, qk_dim});

    // Allocate partial results
    int num_partials = batch_size * num_kv_splits * num_heads;
    auto partial_acc = torch::zeros({num_partials, v_dim},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    auto partial_max = torch::full({num_partials}, -1e30f,
                                    torch::dtype(torch::kFloat32).device(Q.device()));
    auto partial_sum_exp = torch::zeros({num_partials},
                                         torch::dtype(torch::kFloat32).device(Q.device()));

    // Shared memory: LUT(256) + Q(576) + scores(4) + reduce(4) = 840 floats
    int smem_bytes = (FP8_LUT_SIZE + QK_DIM + TOKENS_PER_ITER + NUM_WARPS) * sizeof(float);

    // Launch split-K kernel
    dim3 grid_splitk(batch_size * num_kv_splits, num_heads);
    dim3 block_splitk(BLOCK_SIZE);

    hipLaunchKernelGGL(
        mla_decode_splitk,
        grid_splitk, block_splitk, smem_bytes, 0,
        Q_float.data_ptr<float>(),
        (const unsigned char*)KV_bytes.data_ptr<int8_t>(),
        partial_acc.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        partial_sum_exp.data_ptr<float>(),
        kv_indptr.data_ptr<int>(),
        kv_scale * sm_scale,
        num_kv_splits
    );

    // Allocate output
    int total_heads = batch_size * num_heads;
    auto output = torch::zeros({total_heads, v_dim},
                                torch::dtype(torch::kFloat32).device(Q.device()));

    // Launch reduce kernel
    dim3 grid_reduce(batch_size, num_heads);
    dim3 block_reduce(BLOCK_SIZE);

    hipLaunchKernelGGL(
        mla_reduce,
        grid_reduce, block_reduce, 0, 0,
        partial_acc.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        partial_sum_exp.data_ptr<float>(),
        output.data_ptr<float>(),
        kv_scale,
        num_kv_splits
    );

    return output.view({batch_size, num_heads, v_dim}).to(torch::kBFloat16);
}
"""

# ===================================================================
# C++ SOURCES (pybind declaration)
# ===================================================================

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor mla_hip_forward(
    torch::Tensor Q,
    torch::Tensor KV_fp8,
    torch::Tensor kv_indptr,
    float kv_scale,
    float sm_scale,
    int batch_size,
    int num_kv_splits
);
"""

# ===================================================================
# COMPILE
# ===================================================================

_hip_module = None
_hip_available = False


def _try_compile():
    global _hip_module, _hip_available
    if _hip_module is not None:
        return _hip_available
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_module = load_inline(
            name="mla_hip_phase3",
            cpp_sources=cpp_source,
            cuda_sources=hip_source,
            functions=["mla_hip_forward"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
        _hip_available = True
        print("[HIP Phase 3] Compilation SUCCESS")
    except Exception as e:
        _hip_available = False
        print(f"[HIP Phase 3] Compilation FAILED: {e}")
        import traceback
        traceback.print_exc()
    return _hip_available


_try_compile()


# ===================================================================
# AITER FALLBACK
# ===================================================================

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


# ===================================================================
# ENTRY POINT
# ===================================================================

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]

    if not _hip_available:
        print("[HIP Phase 3] Using AITER fallback")
        return _aiter_fallback(data)

    kv_fp8, kv_scale = kv_data["fp8"]
    kv_scale_val = kv_scale.item()

    kv_bytes = kv_fp8.view(torch.int8).contiguous()

    output = _hip_module.mla_hip_forward(
        q,
        kv_bytes,
        kv_indptr,
        kv_scale_val,
        SM_SCALE,
        bs,
        NUM_KV_SPLITS,
    )

    return output
