"""
Phase 4a-v2: Multi-wave MFMA QK + parallel V accumulation for MLA decode on gfx950.

Key improvement over Phase 4a:
  - 256 threads (4 waves) instead of 64 threads (1 wave)
  - Each wave independently processes 16 tokens via MFMA (same proven layout)
  - 64 tokens per iteration (4 waves x 16 tokens/wave) vs 16 in Phase 4a
  - V accumulation: all 256 threads cooperate, each handles 2 V dims
  - No KV in shared memory — KV loaded directly from global (like Phase 3.5)
  - Online softmax per-head maintained across iterations

Grid: (batch_size * num_kv_splits, 1)
Block: 256 threads = 4 waves

Register layout for v_mfma_f32_16x16x128_f8f6f4 (wave64):
  A[M=16, K=128]: thread t holds A[t%16, (t/16)*32 : (t/16)*32+32] = 8 int32
  B[K=128, N=16]: thread t holds B[(t/16)*32 : (t/16)*32+32, t%16] = 8 int32
  C[M=16, N=16]:  thread t holds C[(t/16)*4+i, t%16] for i=0..3 = 4 float32

gfx950 ONLY. Cannot test on MI300X.
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
#define QK_DIM       576
#define V_DIM        512
#define NUM_HEADS    16
#define WARP_SIZE    64
#define NUM_WARPS    4
#define BLOCK_SIZE   (NUM_WARPS * WARP_SIZE)  // 256
#define BLOCK_N      16     // tokens per MFMA tile per wave
#define TOKENS_PER_ITER (NUM_WARPS * BLOCK_N)  // 64 tokens per iteration
#define MFMA_K       128    // K dimension of MFMA instruction
#define NUM_QK_MFMA  5      // ceil(576/128) = 5 MFMA calls for QK

#define FP8_LUT_SIZE 256
#define REDUCE_BLOCK 256

// V dims per thread: 512 / 256 = 2
#define V_DIMS_PER_THREAD 2

// ---------------------------------------------------------------
// MFMA type definitions
// ---------------------------------------------------------------
typedef int    int8_vec   __attribute__((ext_vector_type(8)));
typedef float  float4_vec __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------
// MFMA instruction wrapper (inline assembly only)
// v_mfma_f32_16x16x128_f8f6f4: C[16x16] += A[16x128] * B[128x16]
// cbsz=0: A is FP8 E4M3, blgp=0: B is FP8 E4M3
// ---------------------------------------------------------------
__device__ __forceinline__ float4_vec mfma_f32_16x16x128_fp8_asm(
    int8_vec a, int8_vec b, float4_vec c
) {
    float4_vec result = c;
    asm volatile(
        "v_mfma_f32_16x16x128_f8f6f4 %0, %1, %2, %0 cbsz:0 blgp:0"
        : "+v"(result)
        : "v"(a), "v"(b)
    );
    return result;
}

// ---------------------------------------------------------------
// FP32 -> FP8 E4M3FNUZ conversion (software)
// ---------------------------------------------------------------
__device__ __forceinline__ unsigned char fp32_to_fp8_e4m3fnuz(float val) {
    if (val == 0.0f) return 0;

    unsigned int bits = __float_as_uint(val);
    unsigned int sign = (bits >> 31) & 1;
    int exponent = ((bits >> 23) & 0xFF) - 127;
    unsigned int mantissa = bits & 0x7FFFFF;

    int fp8_exp = exponent + 8;

    if (fp8_exp >= 15) {
        return (sign << 7) | (14 << 3) | 7;
    }
    if (fp8_exp <= 0) {
        if (fp8_exp < -3) return 0;
        unsigned int fp8_mant = (0x800000 | mantissa) >> (1 - fp8_exp + 20);
        fp8_mant &= 0x7;
        if (fp8_mant == 0) return 0;
        return (sign << 7) | fp8_mant;
    }

    unsigned int fp8_mant = (mantissa + (1 << 19)) >> 20;
    if (fp8_mant > 7) {
        fp8_mant = 0;
        fp8_exp += 1;
        if (fp8_exp >= 15) {
            return (sign << 7) | (14 << 3) | 7;
        }
    }

    return (sign << 7) | (fp8_exp << 3) | fp8_mant;
}

// ---------------------------------------------------------------
// FP8 E4M3FNUZ -> FP32 conversion (for LUT and scalar V dequant)
// ---------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3fnuz_to_fp32(unsigned char val) {
    if (val == 0 || val == 0x80) return 0.0f;

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

// ---------------------------------------------------------------
// Warp-level reductions using shuffle (AMD wavefront=64)
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// ---------------------------------------------------------------
// Shared memory layout (byte offsets):
//   LUT:      256 floats = 1024 bytes  (offset 0)
//   Q FP8:    16 * 576   = 9216 bytes  (offset 1024)
//   Q scale:  1 float    = 4 bytes     (offset 10240)
//   Scores:   4 waves * 16 heads * 16 tokens * 4 = 16384 bytes (offset 10244, aligned to 10256)
//   Total: ~26.6 KB
// ---------------------------------------------------------------
#define SMEM_LUT_OFFSET      0
#define SMEM_LUT_SIZE        (FP8_LUT_SIZE * 4)                         // 1024
#define SMEM_Q_FP8_OFFSET    SMEM_LUT_SIZE                              // 1024
#define SMEM_Q_FP8_SIZE      (NUM_HEADS * QK_DIM)                       // 9216
#define SMEM_QSCALE_OFFSET   (SMEM_Q_FP8_OFFSET + SMEM_Q_FP8_SIZE)     // 10240
#define SMEM_QSCALE_SIZE     4                                          // 4
#define SMEM_SCORES_OFFSET_RAW (SMEM_QSCALE_OFFSET + SMEM_QSCALE_SIZE) // 10244
#define SMEM_SCORES_OFFSET   ((SMEM_SCORES_OFFSET_RAW + 15) & ~15)     // 10256 (16-byte aligned)
#define SMEM_SCORES_SIZE     (NUM_WARPS * NUM_HEADS * BLOCK_N * 4)      // 4*16*16*4 = 16384
#define TOTAL_SMEM_BYTES     (SMEM_SCORES_OFFSET + SMEM_SCORES_SIZE + 16) // ~26.7 KB


// ---------------------------------------------------------------
// Phase 4a-v2: Multi-wave MFMA QK + Parallel V Split-K kernel
//
// Grid:  (batch_size * num_kv_splits, 1)
// Block: 256 threads = 4 waves
//
// Each wave independently computes QK via MFMA for 16 tokens.
// All 256 threads cooperate for V accumulation (2 V dims/thread).
// ---------------------------------------------------------------

__global__ __launch_bounds__(256, 2)
void mla_decode_mfma_v2_splitk(
    const float* __restrict__ Q,             // [batch*16, 576] FP32
    const unsigned char* __restrict__ KV,    // [total_kv, 576] FP8
    float* __restrict__ partial_acc,         // [batch * num_splits * 16, V_DIM]
    float* __restrict__ partial_max,         // [batch * num_splits * 16]
    float* __restrict__ partial_sum_exp,     // [batch * num_splits * 16]
    const int* __restrict__ kv_indptr,       // [batch+1]
    float sm_scale,
    float kv_scale,
    int num_kv_splits
) {
    int batch_split = blockIdx.x;
    int batch = batch_split / num_kv_splits;
    int split = batch_split % num_kv_splits;

    int tid = threadIdx.x;              // 0..255
    int warp_id = tid / WARP_SIZE;      // 0..3
    int lane = tid % WARP_SIZE;         // 0..63

    // MFMA lane mapping (within each wave)
    int lane_row = lane % 16;           // 0..15: head dim (A row) / token index (B col) / C col
    int lane_kgrp = lane / 16;          // 0..3: K-group, each covers 32 K elements

    // KV range for this batch
    int kv_start_all = kv_indptr[batch];
    int kv_end_all = kv_indptr[batch + 1];
    int total_kv = kv_end_all - kv_start_all;

    // Split range
    int tokens_per_split = (total_kv + num_kv_splits - 1) / num_kv_splits;
    int kv_start = kv_start_all + split * tokens_per_split;
    int kv_end = kv_start_all + min((split + 1) * tokens_per_split, total_kv);

    // Handle empty split
    if (kv_start >= kv_end_all || kv_start >= kv_end) {
        for (int h = 0; h < NUM_HEADS; h++) {
            int partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + h;
            if (tid == 0) {
                partial_max[partial_idx] = -1e30f;
                partial_sum_exp[partial_idx] = 0.0f;
            }
            float* p_acc = partial_acc + (long long)partial_idx * V_DIM;
            for (int i = tid; i < V_DIM; i += BLOCK_SIZE) {
                p_acc[i] = 0.0f;
            }
        }
        return;
    }
    if (kv_end > kv_end_all) kv_end = kv_end_all;
    int num_tokens = kv_end - kv_start;

    // Shared memory
    extern __shared__ char smem_raw[];
    float* lut = (float*)(smem_raw + SMEM_LUT_OFFSET);
    unsigned char* smem_q_fp8 = (unsigned char*)(smem_raw + SMEM_Q_FP8_OFFSET);
    float* smem_q_scale = (float*)(smem_raw + SMEM_QSCALE_OFFSET);
    // Scores: [wave][head][token] = smem_scores[wave * NUM_HEADS * BLOCK_N + head * BLOCK_N + token]
    float* smem_scores = (float*)(smem_raw + SMEM_SCORES_OFFSET);

    // Step 1: Build FP8 LUT cooperatively (256 threads, 1 entry each)
    if (tid < FP8_LUT_SIZE) {
        lut[tid] = fp8_e4m3fnuz_to_fp32((unsigned char)tid);
    }

    // Step 2: Quantize Q from FP32 to FP8 in shared memory
    const float* q_base = Q + (long long)batch * NUM_HEADS * QK_DIM;

    // Find max |Q| across all heads and dims
    float local_amax = 0.0f;
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += BLOCK_SIZE) {
        float v = fabsf(q_base[i]);
        local_amax = fmaxf(local_amax, v);
    }
    // Warp-level reduce within each warp
    local_amax = warp_reduce_max(local_amax);
    // Cross-warp reduce via shared memory (reuse scores area temporarily)
    float* cross_warp = smem_scores;
    if (lane == 0) {
        cross_warp[warp_id] = local_amax;
    }
    __syncthreads();
    if (tid == 0) {
        float global_amax = cross_warp[0];
        for (int w = 1; w < NUM_WARPS; w++) {
            global_amax = fmaxf(global_amax, cross_warp[w]);
        }
        float qs = (global_amax > 1e-12f) ? (global_amax / 240.0f) : 1.0f;
        smem_q_scale[0] = qs;
    }
    __syncthreads();

    float q_scale_val = smem_q_scale[0];
    float q_scale_inv = 1.0f / q_scale_val;

    // Convert Q to FP8 cooperatively
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += BLOCK_SIZE) {
        smem_q_fp8[i] = fp32_to_fp8_e4m3fnuz(q_base[i] * q_scale_inv);
    }
    __syncthreads();

    // Per-head online softmax state and V accumulators
    // Each thread holds 2 V dims across all 16 heads
    float v_acc[NUM_HEADS][V_DIMS_PER_THREAD];
    float head_max[NUM_HEADS];
    float head_sum_exp[NUM_HEADS];

    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        head_max[h] = -1e30f;
        head_sum_exp[h] = 0.0f;
        v_acc[h][0] = 0.0f;
        v_acc[h][1] = 0.0f;
    }

    int v_idx0 = tid * V_DIMS_PER_THREAD;        // 0, 2, 4, ..., 510
    int v_idx1 = tid * V_DIMS_PER_THREAD + 1;

    // Combined scale for MFMA QK scores
    float combined_scale = q_scale_val * kv_scale * sm_scale;

    // ===================================================================
    // Main loop: process TOKENS_PER_ITER=64 KV tokens per iteration
    // (4 waves x 16 tokens/wave)
    // ===================================================================
    for (int t_base = 0; t_base < num_tokens; t_base += TOKENS_PER_ITER) {
        int tokens_this_iter = min(TOKENS_PER_ITER, num_tokens - t_base);

        // Each wave handles 16 tokens independently
        int wave_token_base = t_base + warp_id * BLOCK_N;
        int wave_tokens = min(BLOCK_N, max(0, num_tokens - (int)(t_base + warp_id * BLOCK_N)));

        // --- QK Phase: each wave does 5 MFMAs for its 16 tokens ---
        float4_vec qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

        if (wave_tokens > 0) {
            for (int mfma_idx = 0; mfma_idx < NUM_QK_MFMA; mfma_idx++) {
                int k_offset = mfma_idx * MFMA_K;  // 0, 128, 256, 384, 512

                // Load A operand (Q): thread lane needs Q[lane_row, k_offset + lane_kgrp*32 .. +32]
                int8_vec a_reg;
                {
                    int q_byte_base = lane_row * QK_DIM + k_offset + lane_kgrp * 32;
                    const unsigned char* q_src = smem_q_fp8 + q_byte_base;
                    if (k_offset + lane_kgrp * 32 + 31 < QK_DIM) {
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            unsigned int packed = 0;
                            packed |= (unsigned int)q_src[r * 4 + 0];
                            packed |= (unsigned int)q_src[r * 4 + 1] << 8;
                            packed |= (unsigned int)q_src[r * 4 + 2] << 16;
                            packed |= (unsigned int)q_src[r * 4 + 3] << 24;
                            a_reg[r] = (int)packed;
                        }
                    } else {
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            unsigned int packed = 0;
                            #pragma unroll
                            for (int b = 0; b < 4; b++) {
                                int dim = k_offset + lane_kgrp * 32 + r * 4 + b;
                                if (dim < QK_DIM) {
                                    packed |= (unsigned int)smem_q_fp8[lane_row * QK_DIM + dim] << (b * 8);
                                }
                            }
                            a_reg[r] = (int)packed;
                        }
                    }
                }

                // Load B operand (K): from GLOBAL memory (no smem for KV)
                // Thread lane needs KV[token = wave_token_base + lane_row, dim = k_offset + lane_kgrp*32 .. +32]
                int8_vec b_reg;
                {
                    int token_global = kv_start + wave_token_base + lane_row;
                    int k_base_dim = k_offset + lane_kgrp * 32;
                    if (lane_row < wave_tokens && k_base_dim + 31 < QK_DIM) {
                        const unsigned char* kv_src = KV + (long long)token_global * QK_DIM + k_base_dim;
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            unsigned int packed = 0;
                            packed |= (unsigned int)kv_src[r * 4 + 0];
                            packed |= (unsigned int)kv_src[r * 4 + 1] << 8;
                            packed |= (unsigned int)kv_src[r * 4 + 2] << 16;
                            packed |= (unsigned int)kv_src[r * 4 + 3] << 24;
                            b_reg[r] = (int)packed;
                        }
                    } else if (lane_row < wave_tokens) {
                        // Partial — near QK_DIM boundary
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            unsigned int packed = 0;
                            #pragma unroll
                            for (int b2 = 0; b2 < 4; b2++) {
                                int dim = k_base_dim + r * 4 + b2;
                                if (dim < QK_DIM) {
                                    packed |= (unsigned int)KV[(long long)token_global * QK_DIM + dim] << (b2 * 8);
                                }
                            }
                            b_reg[r] = (int)packed;
                        }
                    } else {
                        // Zero-pad for tokens beyond valid range
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            b_reg[r] = 0;
                        }
                    }
                }

                // Execute MFMA
                qk_acc = mfma_f32_16x16x128_fp8_asm(a_reg, b_reg, qk_acc);
            }

            // Extract QK scores to shared memory
            // qk_acc[i] = scores[head = lane_kgrp*4+i, token = lane_row]
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int head_idx = lane_kgrp * 4 + i;
                int token_idx = lane_row;
                // Only store valid token scores
                float score_val = (token_idx < wave_tokens) ? (qk_acc[i] * combined_scale) : -1e30f;
                smem_scores[warp_id * NUM_HEADS * BLOCK_N + head_idx * BLOCK_N + token_idx] = score_val;
            }
        } else {
            // This wave has no tokens — fill scores with -inf
            // Use lane to cover 16*16 = 256 entries, but we have 64 lanes
            for (int i = lane; i < NUM_HEADS * BLOCK_N; i += WARP_SIZE) {
                smem_scores[warp_id * NUM_HEADS * BLOCK_N + i] = -1e30f;
            }
        }

        __syncthreads();

        // --- Online softmax + V accumulation ---
        // All 256 threads participate. Each thread handles 2 V dims across all 16 heads.
        // Read scores from all 4 waves' tiles.

        #pragma unroll 1
        for (int h = 0; h < NUM_HEADS; h++) {
            // Find max score across all valid tokens in this iteration
            float tile_max = -1e30f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                int w_tokens = min(BLOCK_N, max(0, tokens_this_iter - w * BLOCK_N));
                float* wave_scores = smem_scores + w * NUM_HEADS * BLOCK_N + h * BLOCK_N;
                for (int t = 0; t < w_tokens; t++) {
                    tile_max = fmaxf(tile_max, wave_scores[t]);
                }
            }

            float new_max = fmaxf(head_max[h], tile_max);

            // Rescale existing accumulator
            float rescale = expf(head_max[h] - new_max);
            v_acc[h][0] *= rescale;
            v_acc[h][1] *= rescale;
            head_sum_exp[h] *= rescale;
            head_max[h] = new_max;

            // Accumulate V weighted by softmax scores
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                int w_base = t_base + w * BLOCK_N;
                int w_tokens = min(BLOCK_N, max(0, tokens_this_iter - w * BLOCK_N));
                float* wave_scores = smem_scores + w * NUM_HEADS * BLOCK_N + h * BLOCK_N;

                for (int t = 0; t < w_tokens; t++) {
                    float w_val = expf(wave_scores[t] - new_max);
                    head_sum_exp[h] += w_val;

                    int kv_idx = kv_start + w_base + t;
                    const unsigned char* kv_row = KV + (long long)kv_idx * QK_DIM;

                    if (v_idx0 < V_DIM) {
                        v_acc[h][0] += w_val * lut[kv_row[v_idx0]];
                    }
                    if (v_idx1 < V_DIM) {
                        v_acc[h][1] += w_val * lut[kv_row[v_idx1]];
                    }
                }
            }
        }

        __syncthreads();
    }

    // --- Write partial results for all heads ---
    for (int h = 0; h < NUM_HEADS; h++) {
        int partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + h;
        float* p_acc = partial_acc + (long long)partial_idx * V_DIM;

        if (v_idx0 < V_DIM) {
            p_acc[v_idx0] = v_acc[h][0];
        }
        if (v_idx1 < V_DIM) {
            p_acc[v_idx1] = v_acc[h][1];
        }

        if (tid == 0) {
            partial_max[partial_idx] = head_max[h];
            partial_sum_exp[partial_idx] = head_sum_exp[h];
        }
    }
}


// ---------------------------------------------------------------
// Reduce kernel: merge partial results across splits
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

    // Normalize and apply kv_scale to V
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

    // Shared memory
    int smem_bytes = TOTAL_SMEM_BYTES;

    // Launch split-K MFMA kernel: 4 waves per (batch, split)
    dim3 grid_splitk(batch_size * num_kv_splits, 1);
    dim3 block_splitk(BLOCK_SIZE);  // 256 threads

    // Launch using CUDA-style syntax (maps to HIP on ROCm)
    mla_decode_mfma_v2_splitk<<<grid_splitk, block_splitk, smem_bytes>>>(
        Q_float.data_ptr<float>(),
        (const unsigned char*)KV_bytes.data_ptr<int8_t>(),
        partial_acc.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        partial_sum_exp.data_ptr<float>(),
        kv_indptr.data_ptr<int>(),
        sm_scale,
        kv_scale,
        num_kv_splits
    );

    // Allocate output
    int total_heads = batch_size * num_heads;
    auto output = torch::zeros({total_heads, v_dim},
                                torch::dtype(torch::kFloat32).device(Q.device()));

    // Launch reduce kernel
    dim3 grid_reduce(batch_size, num_heads);
    dim3 block_reduce(REDUCE_BLOCK);

    mla_reduce<<<grid_reduce, block_reduce>>>(
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
            name="mla_hip_phase4a_v2",
            cpp_sources=cpp_source,
            cuda_sources=hip_source,
            functions=["mla_hip_forward"],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "-mcpu=gfx950",
                "-mllvm", "-amdgpu-early-inline-all=true",
                "-mllvm", "-amdgpu-function-calls=false",
            ],
        )
        _hip_available = True
        print("[HIP Phase 4a-v2 MFMA] Compilation SUCCESS")
    except Exception as e:
        _hip_available = False
        print(f"[HIP Phase 4a-v2 MFMA] Compilation FAILED: {e}")
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
        if _aiter_available:
            print("[HIP Phase 4a-v2] MFMA compilation failed, using AITER fallback")
            return _aiter_fallback(data)
        else:
            raise RuntimeError("Neither HIP MFMA kernel nor AITER available")

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
