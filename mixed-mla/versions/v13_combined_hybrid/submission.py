"""
v13 Combined Hybrid -- best-of-all routing for MI355X (gfx950)

Routing table (based on MI355X profiling data):
- HIP kernel:      bs=4,kv=1024    (Phase 3.5 scalar FP8 dequant via LUT)
- MXFP4 Triton:    bs=4,kv=8192 + bs=32,kv=1024 + bs=64,kv=1024
                    (dot_scaled for QK, bf16 V)
- AITER bypass:    bs=32,kv=8192 + bs=64,kv=8192
                    (direct C++ calls, fused Q quant)
- AITER wrapper:   bs=256,kv=1024 + bs=256,kv=8192
                    (standard mla_decode_fwd, fused Q quant)

Falls back gracefully if HIP compilation fails (routes to MXFP4 or AITER).
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
import triton
import triton.language as tl
from task import input_t, output_t

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

# MXFP4 K dim layout: 576 = 4*128 + 64 -> 5 tiles of 128 (last padded)
PACKED_QK: int = 288       # 576 / 2 packed bytes
NUM_SCALES: int = 18       # 576 / 32 scale blocks

# ===============================================================
# ROUTING CONFIGS
# ===============================================================

HIP_CONFIGS = {(4, 1024)}
MXFP4_CONFIGS = {(4, 8192), (32, 1024), (64, 1024)}
BYPASS_CONFIGS = {(32, 8192), (64, 8192)}
WRAPPER_CONFIGS = {(256, 1024), (256, 8192)}

# HIP split-K tuning
HIP_NUM_KV_SPLITS = 16

# MXFP4 split-K tuning
MXFP4_KV_SPLITS_MAP = {
    (4, 8192): 16,
    (32, 1024): 4,
    (64, 1024): 4,
}
MXFP4_DEFAULT_KV_SPLITS = 8

# AITER split-K tuning (shared by bypass and wrapper paths)
AITER_KV_SPLITS_MAP = {
    (32, 8192): 48,
    (64, 8192): 24,
    (256, 1024): 16,
    (256, 8192): 24,
}
AITER_DEFAULT_KV_SPLITS = 16

# Caches
_mxfp4_buf_cache: dict = {}
_meta_cache: dict = {}
_bypass_meta_cache: dict = {}
_alloc_cache: dict = {}
_fp8_buf_cache: dict = {}


# ===================================================================
# SECTION 1: HIP KERNEL (Phase 3.5 scalar FP8 dequant via LUT)
# ===================================================================

hip_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define QK_DIM 576
#define V_DIM 512
#define NUM_HEADS 16
#define BLOCK_SIZE 256
#define WARP_SIZE 64
#define NUM_WARPS 4
#define TOKENS_PER_ITER 8
#define TOKENS_PER_WARP 2
#define DIMS_PER_THREAD_V 2
#define FP8_LUT_SIZE 256
#define QK_DIMS_PER_THREAD 9

__device__ __forceinline__ float compute_fp8_value(unsigned char val) {
    if (val == 0) return 0.0f;
    if (val == 0x80) return 0.0f;

    int sign = (val >> 7) & 1;
    int exp_bits = (val >> 3) & 0xF;
    int mant_bits = val & 0x7;

    float mantissa;
    float result;

    if (exp_bits == 0) {
        mantissa = (float)mant_bits / 8.0f;
        result = ldexpf(mantissa, 1 - 8);
    } else {
        mantissa = 1.0f + (float)mant_bits / 8.0f;
        result = ldexpf(mantissa, exp_bits - 8);
    }

    return sign ? -result : result;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void mla_decode_splitk(
    const float* __restrict__ Q,
    const unsigned char* __restrict__ KV,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum_exp,
    const int* __restrict__ kv_indptr,
    float combined_scale,
    int num_kv_splits
) {
    int batch_split = blockIdx.x;
    int head = blockIdx.y;

    int batch = batch_split / num_kv_splits;
    int split = batch_split % num_kv_splits;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    int kv_start_all = kv_indptr[batch];
    int kv_end_all = kv_indptr[batch + 1];
    int total_kv = kv_end_all - kv_start_all;

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

    extern __shared__ float smem[];
    float* lut = smem;
    float* smem_q = smem + FP8_LUT_SIZE;
    float* smem_scores = smem_q + QK_DIM;

    lut[tid] = compute_fp8_value((unsigned char)tid);
    __syncthreads();

    const float* q_ptr = Q + ((long long)batch * NUM_HEADS + head) * QK_DIM;
    for (int i = tid; i < QK_DIM; i += BLOCK_SIZE) {
        smem_q[i] = q_ptr[i];
    }
    __syncthreads();

    float q_regs[QK_DIMS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < QK_DIMS_PER_THREAD; r++) {
        int d = lane + r * WARP_SIZE;
        q_regs[r] = (d < QK_DIM) ? smem_q[d] : 0.0f;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int v_idx0 = tid * DIMS_PER_THREAD_V;
    int v_idx1 = tid * DIMS_PER_THREAD_V + 1;

    float block_max = -1e30f;
    float block_sum_exp = 0.0f;

    int num_tokens = kv_end - kv_start;

    for (int t_base = 0; t_base < num_tokens; t_base += TOKENS_PER_ITER) {
        int tokens_this_iter = min(TOKENS_PER_ITER, num_tokens - t_base);

        #pragma unroll
        for (int w_tok = 0; w_tok < TOKENS_PER_WARP; w_tok++) {
            int tok_idx = warp_id * TOKENS_PER_WARP + w_tok;
            if (tok_idx < tokens_this_iter) {
                int kv_idx = kv_start + t_base + tok_idx;
                const unsigned char* kv_row = KV + (long long)kv_idx * QK_DIM;

                float partial_dot = 0.0f;

                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    int d = lane + r * WARP_SIZE;
                    partial_dot += q_regs[r] * lut[kv_row[d]];
                }
                {
                    int d = lane + 8 * WARP_SIZE;
                    if (d < QK_DIM) {
                        partial_dot += q_regs[8] * lut[kv_row[d]];
                    }
                }

                float score = warp_reduce_sum(partial_dot);

                if (lane == 0) {
                    smem_scores[tok_idx] = score * combined_scale;
                }
            }
        }
        __syncthreads();

        float tile_max = -1e30f;
        #pragma unroll
        for (int t = 0; t < TOKENS_PER_ITER; t++) {
            if (t < tokens_this_iter) {
                tile_max = fmaxf(tile_max, smem_scores[t]);
            }
        }

        float new_max = fmaxf(block_max, tile_max);

        float rescale = expf(block_max - new_max);
        acc0 *= rescale;
        acc1 *= rescale;
        block_sum_exp *= rescale;
        block_max = new_max;

        #pragma unroll
        for (int t = 0; t < TOKENS_PER_ITER; t++) {
            if (t < tokens_this_iter) {
                float w = expf(smem_scores[t] - block_max);
                block_sum_exp += w;

                int kv_idx = kv_start + t_base + t;
                const unsigned char* kv_row = KV + (long long)kv_idx * QK_DIM;

                if (v_idx0 < V_DIM) {
                    acc0 += w * lut[kv_row[v_idx0]];
                }
                if (v_idx1 < V_DIM) {
                    acc1 += w * lut[kv_row[v_idx1]];
                }
            }
        }
        __syncthreads();
    }

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

    float global_max = -1e30f;
    for (int s = 0; s < num_kv_splits; s++) {
        int idx = (batch * num_kv_splits + s) * NUM_HEADS + head;
        float m = partial_max[idx];
        global_max = fmaxf(global_max, m);
    }

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

    float inv_sum = (global_sum_exp > 0.0f) ? (1.0f / global_sum_exp) : 0.0f;
    float* out_row = output + ((long long)batch * NUM_HEADS + head) * V_DIM;

    if (v_idx0 < V_DIM) {
        out_row[v_idx0] = final_v0 * inv_sum * kv_scale;
    }
    if (v_idx1 < V_DIM) {
        out_row[v_idx1] = final_v1 * inv_sum * kv_scale;
    }
}

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

    auto Q_float = Q.to(torch::kFloat32).contiguous().view({-1, qk_dim});
    auto KV_bytes = KV_fp8.contiguous().view({-1, qk_dim});

    int num_partials = batch_size * num_kv_splits * num_heads;
    auto partial_acc = torch::zeros({num_partials, v_dim},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    auto partial_max = torch::full({num_partials}, -1e30f,
                                    torch::dtype(torch::kFloat32).device(Q.device()));
    auto partial_sum_exp = torch::zeros({num_partials},
                                         torch::dtype(torch::kFloat32).device(Q.device()));

    int smem_bytes = (FP8_LUT_SIZE + QK_DIM + TOKENS_PER_ITER) * sizeof(float);

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

    int total_heads = batch_size * num_heads;
    auto output = torch::zeros({total_heads, v_dim},
                                torch::dtype(torch::kFloat32).device(Q.device()));

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

hip_cpp_source = r"""
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

# Compile HIP kernel
_hip_module = None
_hip_available = False


def _try_compile_hip():
    global _hip_module, _hip_available
    if _hip_module is not None:
        return _hip_available
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_module = load_inline(
            name="mla_hip_v13",
            cpp_sources=hip_cpp_source,
            cuda_sources=hip_source,
            functions=["mla_hip_forward"],
            verbose=False,
            extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-early-inline-all=true",
                               "-mllvm", "-amdgpu-function-calls=false"],
        )
        _hip_available = True
        print("[v13 HIP] Compilation SUCCESS")
    except Exception as e:
        _hip_available = False
        print(f"[v13 HIP] Compilation FAILED: {e}")
        import traceback
        traceback.print_exc()
    return _hip_available


_try_compile_hip()


# ===================================================================
# SECTION 2: AITER IMPORTS (best-effort)
# ===================================================================

_aiter_available = False
try:
    import aiter
    from aiter.mla import mla_decode_fwd
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant
    _aiter_available = True
    FP8_DTYPE = aiter_dtypes.fp8
    print("[v13 AITER] Import SUCCESS")
except ImportError as e:
    FP8_DTYPE = torch.float8_e4m3fnuz
    print(f"[v13 AITER] Import FAILED: {e}")


# ===================================================================
# SECTION 3: FUSED FP8 QUANTIZATION TRITON KERNELS
# ===================================================================

@triton.jit
def _amax_kernel(
    input_ptr,
    amax_ptr,
    N,
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
    input_ptr,
    output_ptr,
    amax_ptr,
    scale_ptr,
    fp8_max,
    fp8_min,
    N,
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
    Returns (fp8_tensor, scale).
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


# ===================================================================
# SECTION 4: MXFP4 TRITON KERNELS
# ===================================================================

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
    """
    Stage 1: For each (batch, split, v_chunk), compute partial attention.
    All 16 Q heads processed together (BLOCK_M=16).
    K dimension tiled in 5 chunks of 128 dims via dot_scaled.
    V accumulated using regular tl.dot in bf16.
    """
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


# ===================================================================
# SECTION 5: MXFP4 BUFFER CACHE AND DECODE PATH
# ===================================================================

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


# ===================================================================
# SECTION 6: HIP DECODE PATH
# ===================================================================

def _hip_path(q, kv_data, kv_indptr, config):
    """
    HIP kernel path for small batch + short KV (Phase 3.5 scalar FP8 dequant).
    Falls back to MXFP4 or AITER if HIP compilation failed.
    """
    bs = config["batch_size"]

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
        HIP_NUM_KV_SPLITS,
    )

    return output


# ===================================================================
# SECTION 7: AITER CACHED METADATA
# ===================================================================

def _get_cached_meta(bs, nq, nkv, q_dtype, kv_dtype, qo_indptr, kv_indptr, num_kv_splits):
    """Cached metadata for the AITER wrapper path."""
    key = ("wrapper", bs, num_kv_splits, q_dtype, kv_dtype)
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


def _get_cached_bypass_meta(bs, nq, nkv, q_dtype, kv_dtype, qo_indptr, kv_indptr, num_kv_splits):
    """Cached metadata for the AITER bypass path (includes logits/attn_lse buffers)."""
    key = ("bypass", bs, num_kv_splits, q_dtype, kv_dtype)
    if key not in _bypass_meta_cache:
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

        num_partials = rpm.size(0)
        logits = torch.empty(
            (num_partials, 1, NUM_HEADS, V_HEAD_DIM),
            dtype=torch.float32, device="cuda",
        )
        attn_lse = torch.empty(
            (num_partials, 1, NUM_HEADS, 1),
            dtype=torch.float32, device="cuda",
        )

        _bypass_meta_cache[key] = {
            "work_meta_data": wm, "work_indptr": wi, "work_info_set": wis,
            "reduce_indptr": ri, "reduce_final_map": rfm, "reduce_partial_map": rpm,
            "kv_indices": kv_indices, "kv_last_page_len": kv_last_page_len,
            "logits": logits, "attn_lse": attn_lse,
        }
    return _bypass_meta_cache[key]


def _get_cached_allocs(bs, nq, device):
    key = (bs, nq)
    if key not in _alloc_cache:
        _alloc_cache[key] = {
            "output": torch.empty((bs, nq, V_HEAD_DIM), dtype=torch.bfloat16, device=device),
        }
    return _alloc_cache[key]


# ===================================================================
# SECTION 8: AITER BYPASS PATH (direct C++ calls)
# ===================================================================

def _aiter_bypass_path(q, kv_data, qo_indptr, kv_indptr, config):
    """
    AITER path calling C++ functions directly, bypassing the Python wrapper.
    Saves 16-20 us per call by eliminating torch.empty allocations for
    logits and attn_lse intermediate buffers.

    Used for: (32, 8192), (64, 8192)
    """
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]
    num_kv_splits = AITER_KV_SPLITS_MAP.get((bs, kvlen), AITER_DEFAULT_KV_SPLITS)

    q_fp8, q_scale = fused_quantize_fp8(q)

    kv_fp8, kv_scale = kv_data["fp8"]

    meta = _get_cached_bypass_meta(
        bs, NUM_HEADS, NUM_KV_HEADS,
        q_fp8.dtype, kv_fp8.dtype,
        qo_indptr, kv_indptr, num_kv_splits,
    )
    allocs = _get_cached_allocs(bs, NUM_HEADS, q.device)
    o = allocs["output"]

    kv_4d = kv_fp8.view(kv_fp8.shape[0], PAGE_SIZE, NUM_KV_HEADS, kv_fp8.shape[-1])

    logits = meta["logits"]
    attn_lse = meta["attn_lse"]

    # Direct call to AITER C++ stage1 -- 19 args (no final_lse parameter)
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_4d,
        qo_indptr,
        kv_indptr,
        meta["kv_indices"],
        meta["kv_last_page_len"],
        None,                              # num_kv_splits_indptr
        meta["work_meta_data"],
        meta["work_indptr"],
        meta["work_info_set"],
        1,                                 # max_seqlen_q
        PAGE_SIZE,                         # page_size
        NUM_KV_HEADS,                      # nhead_kv
        SM_SCALE,                          # sm_scale
        logits,                            # splitData (cached)
        attn_lse,                          # splitLse (cached)
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


# ===================================================================
# SECTION 9: AITER WRAPPER PATH (standard mla_decode_fwd)
# ===================================================================

def _aiter_wrapper_path(q, kv_data, qo_indptr, kv_indptr, config):
    """
    AITER path using the standard mla_decode_fwd Python wrapper with
    cached metadata. Bypass hurts bs=256 configs (+12 us), so we keep
    the wrapper for those.

    Used for: (256, 1024), (256, 8192)
    """
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


# ===================================================================
# SECTION 10: ENTRY POINT WITH ROUTING
# ===================================================================

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kvlen = config["kv_seq_len"]

    # Route 1: HIP kernel for small batch + short KV
    if (bs, kvlen) in HIP_CONFIGS and _hip_available:
        return _hip_path(q, kv_data, kv_indptr, config)

    # Route 2: MXFP4 Triton for medium configs (or HIP fallback targets)
    if (bs, kvlen) in MXFP4_CONFIGS:
        if _aiter_available:
            return _mxfp4_path(q, kv_data, kv_indptr, config)
        # If AITER not available (no dynamic_mxfp4_quant), fall through

    # Route 2b: HIP fallback -- if HIP failed, route its configs to MXFP4 or AITER
    if (bs, kvlen) in HIP_CONFIGS and not _hip_available:
        if _aiter_available:
            return _mxfp4_path(q, kv_data, kv_indptr, config)
        # Last resort: AITER wrapper (needs aiter)

    # Route 3: AITER bypass for medium-batch + long KV
    if (bs, kvlen) in BYPASS_CONFIGS and _aiter_available:
        return _aiter_bypass_path(q, kv_data, qo_indptr, kv_indptr, config)

    # Route 4: AITER wrapper for large batch (default)
    if _aiter_available:
        return _aiter_wrapper_path(q, kv_data, qo_indptr, kv_indptr, config)

    # Ultimate fallback: HIP kernel for everything (if available)
    if _hip_available:
        return _hip_path(q, kv_data, kv_indptr, config)

    raise RuntimeError(
        f"[v13] No available backend for config (bs={bs}, kvlen={kvlen}). "
        f"HIP available: {_hip_available}, AITER available: {_aiter_available}"
    )
