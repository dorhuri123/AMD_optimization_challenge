"""
Phase 4a: MFMA-accelerated QK dot product for MLA decode on gfx950.

Uses v_mfma_f32_16x16x128_f8f6f4 hardware instruction for the QK phase:
- 5 MFMA calls cover 576 QK dims (5 x 128 = 640, last 64 padded)
- Each MFMA computes: C[16x16] += A[16x128] * B[128x16]
  where M=16 heads, K=128 dims, N=16 tokens
- Scalar softmax + PV accumulation (same as Phase 3.5)

Grid: (batch_size * num_kv_splits, 1)
Block: 64 threads = 1 wave (minimal for MFMA)

Register layout for v_mfma_f32_16x16x128_f8f6f4 (wave64):
  A[M=16, K=128]: thread t holds A[t%16, (t/16)*32 : (t/16)*32+32] = 32 bytes = 8 int32
  B[K=128, N=16]: thread t holds B[(t/16)*32 : (t/16)*32+32, t%16] = 32 bytes = 8 int32
  C[M=16, N=16]:  thread t holds C[t%16, (t/16)*4 : (t/16)*4+4] = 4 float32

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
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------
// Constants
// ---------------------------------------------------------------
#define QK_DIM       576
#define V_DIM        512
#define NUM_HEADS    16
#define WARP_SIZE    64
#define BLOCK_N      16     // tokens per MFMA tile (matches MFMA N=16)
#define MFMA_K       128    // K dimension of MFMA instruction
#define NUM_QK_MFMA  5      // ceil(576/128) = 5 MFMA calls for QK
// Define USE_MFMA_BUILTIN to use the compiler builtin instead of inline asm.
// The builtin is preferred when the compiler supports it (ROCm 6.4+ with gfx950).
// Comment out to use inline assembly fallback.
#define USE_MFMA_BUILTIN 1

#define FP8_LUT_SIZE 256

// For the 256-thread reduce kernel
#define REDUCE_BLOCK 256

// ---------------------------------------------------------------
// MFMA type definitions
// ---------------------------------------------------------------
// int8_vec: 8 x int32 = 32 bytes = 128 FP8 elements per thread
// float4_vec: 4 x float32 = MFMA accumulator per thread
typedef int    int8_vec   __attribute__((ext_vector_type(8)));
typedef float  float4_vec __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------
// MFMA instruction wrapper
// Computes: C[16x16] += A[16x128] * B[128x16] in FP8
// cbsz=0: A is FP8 E4M3, blgp=0: B is FP8 E4M3
//
// Try builtin first; if it fails at compile time, the inline asm
// fallback (mfma_f32_16x16x128_fp8) is used instead.
// ---------------------------------------------------------------

// Method 1: Compiler builtin (available in ROCm 6.4+ targeting gfx950)
// No forward declaration needed -- the compiler recognizes __builtin_* names

// Method 2: Inline assembly fallback
// The v_mfma_f32_16x16x128_f8f6f4 instruction:
//   dst/srcC: 4 VGPRs (float4), srcA: 8 VGPRs (int8), srcB: 8 VGPRs (int8)
//   cbsz and blgp are immediates in the instruction word.
//
// With ext_vector_type, the inline asm constraint "v" on a vector type
// automatically allocates the correct number of consecutive VGPRs.
// %0 = v[dst:dst+3], %1 = v[a:a+7], %2 = v[b:b+7]
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
// Matches the FNUZ format used in ROCm: no negative zero, no inf/nan
// ---------------------------------------------------------------
__device__ __forceinline__ unsigned char fp32_to_fp8_e4m3fnuz(float val) {
    if (val == 0.0f) return 0;

    unsigned int bits = __float_as_uint(val);
    unsigned int sign = (bits >> 31) & 1;
    int exponent = ((bits >> 23) & 0xFF) - 127;  // unbiased
    unsigned int mantissa = bits & 0x7FFFFF;  // 23-bit

    // FP8 E4M3 FNUZ: bias=8, exp range [-7, 7], max = 240
    int fp8_exp = exponent + 8;  // re-bias to FP8

    if (fp8_exp >= 15) {
        // Clamp to max normal: sign | exp=14 | mant=7 -> +-240
        return (sign << 7) | (14 << 3) | 7;
    }
    if (fp8_exp <= 0) {
        // Subnormal or zero
        if (fp8_exp < -3) return 0;  // too small
        // Subnormal: shift mantissa right
        unsigned int fp8_mant = (0x800000 | mantissa) >> (1 - fp8_exp + 20);
        fp8_mant &= 0x7;
        if (fp8_mant == 0) return 0;
        return (sign << 7) | fp8_mant;
    }

    // Normal: round mantissa from 23 bits to 3 bits
    unsigned int fp8_mant = (mantissa + (1 << 19)) >> 20;  // round to nearest
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
// FP8 E4M3FNUZ -> FP32 conversion (for scalar dequant in PV phase)
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
// Build FP8 LUT in shared memory (256 entries)
// ---------------------------------------------------------------
__device__ void build_fp8_lut(float* lut, int tid, int nthreads) {
    for (int i = tid; i < FP8_LUT_SIZE; i += nthreads) {
        lut[i] = fp8_e4m3fnuz_to_fp32((unsigned char)i);
    }
}

// ---------------------------------------------------------------
// Warp-level reduction (sum) using shuffle -- AMD wavefront=64
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------
// Warp-level reduction (max) using shuffle
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}


// ---------------------------------------------------------------
// Phase 4a: MFMA QK + Scalar PV Split-K kernel
//
// Grid:  (batch_size * num_kv_splits, 1)
// Block: 64 threads = 1 wave
//
// One wave handles ALL 16 heads via the MFMA M=16 dimension.
// Processes BLOCK_N=16 KV tokens per iteration.
//
// Shared memory layout:
//   [0..255]           : FP8 LUT (256 floats = 1024 bytes)
//   [256..9471]        : Q as FP8: 16 heads x 576 dims = 9216 bytes
//                        stored as bytes at float offset 256
//   [256+2304..?]      : KV tile: 16 tokens x 576 dims = 9216 bytes
//   [scores area]      : 16x16 float scores = 256 floats = 1024 bytes
//   Total: ~20KB
// ---------------------------------------------------------------

// Shared memory byte offsets
#define SMEM_LUT_FLOATS   256
#define SMEM_Q_FP8_OFFSET (SMEM_LUT_FLOATS * 4)                    // 1024 bytes
#define SMEM_Q_FP8_SIZE   (NUM_HEADS * QK_DIM)                     // 9216 bytes
#define SMEM_KV_OFFSET    (SMEM_Q_FP8_OFFSET + SMEM_Q_FP8_SIZE)   // 10240
// Align KV to 16 bytes
#define SMEM_KV_ALIGNED   ((SMEM_KV_OFFSET + 15) & ~15)            // 10240 (already aligned)
#define SMEM_KV_SIZE      (BLOCK_N * QK_DIM)                       // 9216 bytes
#define SMEM_SCORES_OFFSET (SMEM_KV_ALIGNED + SMEM_KV_SIZE)        // 19456
#define SMEM_SCORES_SIZE   (NUM_HEADS * BLOCK_N * 4)               // 1024 bytes (256 floats)
#define SMEM_QSCALE_OFFSET (SMEM_SCORES_OFFSET + SMEM_SCORES_SIZE) // after scores
#define SMEM_QSCALE_SIZE   4                                       // 1 float = 4 bytes
#define TOTAL_SMEM_BYTES   (SMEM_QSCALE_OFFSET + SMEM_QSCALE_SIZE + 16) // ~20.5KB + padding


__global__ __launch_bounds__(64)
void mla_decode_mfma_splitk(
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

    int tid = threadIdx.x;  // 0..63 (one wave)
    int lane = tid;

    // MFMA lane mapping
    int lane_row = lane % 16;     // which head (for A) / which token col (for B) / which C row
    int lane_kgrp = lane / 16;    // which K-group (0..3), each covers 32 K elements

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
        // Write zeros for all heads
        for (int h = 0; h < NUM_HEADS; h++) {
            int partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + h;
            partial_max[partial_idx] = -1e30f;
            partial_sum_exp[partial_idx] = 0.0f;
            float* p_acc = partial_acc + (long long)partial_idx * V_DIM;
            for (int i = tid; i < V_DIM; i += WARP_SIZE) {
                p_acc[i] = 0.0f;
            }
        }
        return;
    }
    if (kv_end > kv_end_all) kv_end = kv_end_all;
    int num_tokens = kv_end - kv_start;

    // Shared memory (raw bytes)
    extern __shared__ char smem_raw[];
    float* lut = (float*)smem_raw;                                // 256 floats at offset 0
    unsigned char* smem_q_fp8 = (unsigned char*)(smem_raw + SMEM_Q_FP8_OFFSET);
    unsigned char* smem_kv = (unsigned char*)(smem_raw + SMEM_KV_ALIGNED);
    float* smem_scores = (float*)(smem_raw + SMEM_SCORES_OFFSET);
    float* smem_q_scale = (float*)(smem_raw + SMEM_QSCALE_OFFSET);

    // Step 1: Build FP8 LUT (64 threads, 4 iterations each)
    for (int i = tid; i < FP8_LUT_SIZE; i += WARP_SIZE) {
        lut[i] = fp8_e4m3fnuz_to_fp32((unsigned char)i);
    }

    // Step 2: Compute Q scale and convert Q from FP32 to scaled FP8
    // Find max |Q| across all heads and dims, then scale Q to fit in FP8 range.
    // FP8 E4M3FNUZ max = 240.0
    const float FP8_MAX = 240.0f;
    const float* q_base = Q + (long long)batch * NUM_HEADS * QK_DIM;

    // Each thread finds local max across its assigned elements
    float local_amax = 0.0f;
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += WARP_SIZE) {
        float v = fabsf(q_base[i]);
        local_amax = fmaxf(local_amax, v);
    }
    // Warp reduce to find global max
    local_amax = warp_reduce_max(local_amax);

    // Compute scale: q_scale = amax / FP8_MAX (so Q_fp8 = Q / q_scale fits in [-240, 240])
    float q_scale_val;
    if (tid == 0) {
        q_scale_val = (local_amax > 1e-12f) ? (local_amax / FP8_MAX) : 1.0f;
        smem_q_scale[0] = q_scale_val;
    }
    // All threads in the wave see the same value from warp_reduce_max, so:
    q_scale_val = (local_amax > 1e-12f) ? (local_amax / FP8_MAX) : 1.0f;
    float q_scale_inv = 1.0f / q_scale_val;

    // Convert Q to FP8 with scaling
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += WARP_SIZE) {
        smem_q_fp8[i] = fp32_to_fp8_e4m3fnuz(q_base[i] * q_scale_inv);
    }

    __syncthreads();

    // Per-head V accumulators: each thread handles 8 V dims across all 16 heads.
    // 64 threads x 8 dims = 512 = V_DIM.
    // Per thread: 16 heads x 8 V accumulators = 128 floats + 32 softmax state = 160 VGPRs.
    // gfx950 has 512 VGPRs per wave, so this fits.
    #define V_DIMS_PER_THREAD 8
    float v_acc[NUM_HEADS][V_DIMS_PER_THREAD];
    float head_max[NUM_HEADS];
    float head_sum_exp[NUM_HEADS];

    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        head_max[h] = -1e30f;
        head_sum_exp[h] = 0.0f;
        #pragma unroll
        for (int d = 0; d < V_DIMS_PER_THREAD; d++) {
            v_acc[h][d] = 0.0f;
        }
    }

    // V dim assignments for this thread
    int v_base = tid * V_DIMS_PER_THREAD;  // 0, 8, 16, ..., 504

    // ===================================================================
    // Main loop: process BLOCK_N=16 KV tokens per iteration
    // ===================================================================
    for (int t_base = 0; t_base < num_tokens; t_base += BLOCK_N) {
        int tokens_this_iter = min(BLOCK_N, num_tokens - t_base);

        // --- Load KV tile into shared memory ---
        // KV[kv_start + t_base + token, dim] -> smem_kv[token * QK_DIM + dim]
        // Total: tokens_this_iter * 576 bytes. 64 threads cooperative load.
        int total_bytes = tokens_this_iter * QK_DIM;
        for (int i = tid; i < total_bytes; i += WARP_SIZE) {
            int kv_idx = kv_start + t_base + i / QK_DIM;
            int dim = i % QK_DIM;
            smem_kv[i] = KV[(long long)kv_idx * QK_DIM + dim];
        }

        // Zero-pad remaining token slots if tokens_this_iter < BLOCK_N
        if (tokens_this_iter < BLOCK_N) {
            int pad_start = tokens_this_iter * QK_DIM;
            int pad_end = BLOCK_N * QK_DIM;
            for (int i = pad_start + tid; i < pad_end; i += WARP_SIZE) {
                smem_kv[i] = 0;
            }
        }

        __syncthreads();

        // --- QK Phase: 5 MFMAs ---
        // C[16 heads, 16 tokens] += Q_fp8[16, 128] * K_fp8[128, 16]
        float4_vec qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int mfma_idx = 0; mfma_idx < NUM_QK_MFMA; mfma_idx++) {
            int k_offset = mfma_idx * MFMA_K;  // 0, 128, 256, 384, 512

            // Load A operand (Q data) into int8_vec register
            // Thread tid (lane) needs Q[lane_row, k_offset + lane_kgrp*32 .. +32]
            // = smem_q_fp8[lane_row * QK_DIM + k_offset + lane_kgrp * 32 + 0..31]
            int8_vec a_reg;
            {
                int q_byte_base = lane_row * QK_DIM + k_offset + lane_kgrp * 32;
                // Load 32 bytes (8 x int32) from smem_q_fp8
                // Each int32 packs 4 FP8 bytes
                const unsigned char* q_src = smem_q_fp8 + q_byte_base;
                // Handle out-of-bounds (dim >= 576 in the last MFMA)
                if (k_offset + lane_kgrp * 32 + 31 < QK_DIM) {
                    // All 32 bytes are valid
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
                    // Partial -- zero-pad bytes beyond QK_DIM
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

            // Load B operand (K data, transposed) into int8_vec register
            // For B[K=128, N=16]: thread tid needs K[(lane_kgrp*32)..+32, lane_row]
            // In memory, K is stored as KV[token, dim], so K transposed has:
            //   B[k, n] = KV[token_n, k_offset + k]
            // Thread tid needs: KV[token = lane_row, dim = k_offset + lane_kgrp*32..+32]
            // = smem_kv[lane_row * QK_DIM + k_offset + lane_kgrp * 32 + 0..31]
            //
            // NOTE: For B operand, the data layout is column-major over N.
            // B[k, col] where col = lane_row (0..15), k_group = lane_kgrp (0..3)
            // thread holds 32 consecutive K elements for one column (token).
            int8_vec b_reg;
            {
                int kv_byte_base = lane_row * QK_DIM + k_offset + lane_kgrp * 32;
                const unsigned char* kv_src = smem_kv + kv_byte_base;
                if (k_offset + lane_kgrp * 32 + 31 < QK_DIM) {
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        unsigned int packed = 0;
                        packed |= (unsigned int)kv_src[r * 4 + 0];
                        packed |= (unsigned int)kv_src[r * 4 + 1] << 8;
                        packed |= (unsigned int)kv_src[r * 4 + 2] << 16;
                        packed |= (unsigned int)kv_src[r * 4 + 3] << 24;
                        b_reg[r] = (int)packed;
                    }
                } else {
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        unsigned int packed = 0;
                        #pragma unroll
                        for (int b = 0; b < 4; b++) {
                            int dim = k_offset + lane_kgrp * 32 + r * 4 + b;
                            if (dim < QK_DIM) {
                                packed |= (unsigned int)smem_kv[lane_row * QK_DIM + dim] << (b * 8);
                            }
                        }
                        b_reg[r] = (int)packed;
                    }
                }
            }

            // Execute MFMA: C[16, 16] += A[16, 128] * B[128, 16]
#ifdef USE_MFMA_BUILTIN
            qk_acc = __builtin_amdgcn_mfma_f32_16x16x128_f8f6f4(
                a_reg, b_reg, qk_acc,
                0,  // cbsz=0: FP8 E4M3 for A
                0   // blgp=0: FP8 E4M3 for B
            );
#else
            qk_acc = mfma_f32_16x16x128_fp8_asm(a_reg, b_reg, qk_acc);
#endif
        }

        // --- Extract QK scores to shared memory ---
        // After MFMA: thread tid holds C[lane_row, lane_kgrp*4 : lane_kgrp*4+4]
        // = scores[head=lane_row, tokens=lane_kgrp*4..lane_kgrp*4+3]
        // Need to store as smem_scores[head][token]
        //
        // Apply combined scale: q_scale * kv_scale * sm_scale
        // The MFMA computes: sum(dequant(Q_fp8[h,k]) * dequant(K_fp8[t,k]))
        // Q_fp8 was quantized as: Q_fp8 = fp8(Q_f32 / q_scale_val)
        // So dequant(Q_fp8) ~= Q_f32 / q_scale_val
        // K_fp8 is the original FP8 data, dequant(K_fp8) ~= K_original / kv_scale? No.
        // K_fp8 IS the stored KV cache. dequant(K_fp8) gives the FP8 numeric value.
        // The actual K value is: K_actual = dequant(K_fp8) * kv_scale.
        // So MFMA result = sum((Q_f32/q_scale) * dequant(K_fp8))
        // True score = sum(Q_f32 * K_actual) = sum(Q_f32 * dequant(K_fp8) * kv_scale)
        // Therefore: true_score = MFMA_result * q_scale_val * kv_scale
        // Final attention score = true_score * sm_scale
        float combined_scale = q_scale_val * kv_scale * sm_scale;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int token_idx = lane_kgrp * 4 + i;
            int head_idx = lane_row;
            smem_scores[head_idx * BLOCK_N + token_idx] = qk_acc[i] * combined_scale;
        }

        __syncthreads();

        // --- Online softmax + V accumulation (scalar, per head) ---
        // All 64 threads participate. Each thread handles 8 V dims across all 16 heads.
        #pragma unroll 1
        for (int h = 0; h < NUM_HEADS; h++) {
            // Find max score for this head across tokens_this_iter tokens
            float tile_max = -1e30f;
            for (int t = 0; t < tokens_this_iter; t++) {
                tile_max = fmaxf(tile_max, smem_scores[h * BLOCK_N + t]);
            }

            float new_max = fmaxf(head_max[h], tile_max);

            // Rescale existing accumulator
            float rescale = expf(head_max[h] - new_max);
            #pragma unroll
            for (int d = 0; d < V_DIMS_PER_THREAD; d++) {
                v_acc[h][d] *= rescale;
            }
            head_sum_exp[h] *= rescale;
            head_max[h] = new_max;

            // Accumulate V weighted by softmax
            for (int t = 0; t < tokens_this_iter; t++) {
                float w = expf(smem_scores[h * BLOCK_N + t] - new_max);
                head_sum_exp[h] += w;

                // Read V from shared memory (already loaded for QK phase)
                const unsigned char* kv_smem_row = &smem_kv[t * QK_DIM];

                #pragma unroll
                for (int d = 0; d < V_DIMS_PER_THREAD; d++) {
                    int v_dim = v_base + d;
                    if (v_dim < V_DIM) {
                        v_acc[h][d] += w * lut[kv_smem_row[v_dim]];
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

        #pragma unroll
        for (int d = 0; d < V_DIMS_PER_THREAD; d++) {
            int v_dim = v_base + d;
            if (v_dim < V_DIM) {
                p_acc[v_dim] = v_acc[h][d];
            }
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
    // Each thread handles 2 V dims (256 threads * 2 = 512)
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

    // Normalize
    // Note: kv_scale is already applied in the splitk kernel via combined_scale.
    // The V values from the LUT are already in dequantized FP8 scale.
    // We need to apply kv_scale to V values here.
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

    // Launch split-K MFMA kernel: 1 wave per (batch, split)
    dim3 grid_splitk(batch_size * num_kv_splits, 1);
    dim3 block_splitk(WARP_SIZE);  // 64 threads

    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    hipLaunchKernelGGL(
        mla_decode_mfma_splitk,
        grid_splitk, block_splitk, smem_bytes, stream,
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

    hipLaunchKernelGGL(
        mla_reduce,
        grid_reduce, block_reduce, 0, stream,
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
            name="mla_hip_phase4a",
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
        print("[HIP Phase 4a MFMA] Compilation SUCCESS")
    except Exception as e:
        _hip_available = False
        print(f"[HIP Phase 4a MFMA] Compilation FAILED: {e}")
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
            print("[HIP Phase 4a] MFMA compilation failed, using AITER fallback")
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
