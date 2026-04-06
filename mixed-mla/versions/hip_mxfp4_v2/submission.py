"""
MXFP4 v2: Multi-wave Scaled MFMA QK + Scalar FP8 PV for MLA decode on gfx950.

Combines proven pieces:
  1. __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 with cbsz=4 blgp=4 (MXFP4)
  2. Phase 4a-v2 multi-wave architecture (256 threads, 4 waves x 16 tokens each)
  3. Scalar FP8 V dequant via LUT (proven working from Phase 4a-v2)

QK Phase: MXFP4 scaled MFMA (5 calls per 16-token tile)
  Q: from dynamic_mxfp4_quant(q) — fp4x2 packed + e8m0 scales
  K: from kv_data["mxfp4"] — fp4x2 packed + e8m0 scales
  Hardware applies block scales automatically via scale operands.

PV Phase: Scalar FP8 dequant + accumulate
  V: from kv_data["fp8"] — FP8 with scalar kv_scale
  All 256 threads cooperate, each handles 2 V dims.

Grid: (batch_size * num_kv_splits, 1)
Block: 256 threads = 4 waves (wave64)

gfx950 ONLY.
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["GPU_MAX_HW_QUEUES"] = "2"
os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"

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

# MXFP4 layout constants
PACKED_QK: int = 288       # 576 / 2 packed bytes
NUM_SCALES: int = 18       # 576 / 32 scale blocks

# Split-K config per (batch_size, kv_seq_len)
KV_SPLITS_MAP = {
    (4, 1024): 4,
    (4, 8192): 16,
    (32, 1024): 4,
    (32, 8192): 16,
    (64, 1024): 8,
    (64, 8192): 16,
    (256, 1024): 8,
    (256, 8192): 16,
}

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

// MXFP4: 576 dims => 288 packed bytes (2 values per byte)
// MXFP4 block_size = 32, so 576/32 = 18 scale blocks per row
#define PACKED_QK    288
#define NUM_SCALE_BLOCKS 18

// MFMA K=128 FP4 elements per instruction
// Each thread holds 4 x int32 = 16 bytes = 32 FP4 elements per kgroup
#define MFMA_K       128
#define NUM_QK_MFMA  5      // ceil(576/128) = 5 MFMA calls for full QK dim
#define BYTES_PER_MFMA_K  64  // 128 FP4 elements = 64 bytes

#define FP8_LUT_SIZE 256
#define REDUCE_BLOCK 256

// V dims per thread: 512 / 256 = 2
#define V_DIMS_PER_THREAD 2

// ---------------------------------------------------------------
// MFMA type definitions for MXFP4
// ---------------------------------------------------------------
// For FP4 (cbsz=4): srcA/srcB are 4 x int32 (16 bytes = 32 FP4 elements per kgroup)
// The builtin takes 8 x int32 but only reads first 4 for FP4.
#define USE_SCALED_MFMA_BUILTIN 1
typedef int    int4_vec   __attribute__((ext_vector_type(4)));
typedef int    int8_vec   __attribute__((ext_vector_type(8)));
typedef float  float4_vec __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------
// Scaled MFMA wrapper for MXFP4
//
// v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz=4 blgp=4:
//   C[16x16] += scale_a * A[16x128] * scale_b * B[128x16]  (all in MXFP4/FP4)
//
// The scale operands are int32 VGPRs holding packed e8m0 scales.
// op_sel_hi:[0,0,0] means: hardware reads byte 0 of each scale VGPR.
// ---------------------------------------------------------------

#if defined(USE_SCALED_MFMA_BUILTIN) && USE_SCALED_MFMA_BUILTIN
__device__ __forceinline__ float4_vec mfma_scale_fp4(
    int4_vec a, int4_vec b, float4_vec c, int sa, int sb
) {
    // Pad to 8 x int32 as required by builtin signature
    int8_vec a_pad = {a[0], a[1], a[2], a[3], 0, 0, 0, 0};
    int8_vec b_pad = {b[0], b[1], b[2], b[3], 0, 0, 0, 0};
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_pad, b_pad, c,
        4,   // cbsz: 4 = FP4 for A
        4,   // blgp: 4 = FP4 for B
        0,   // opsel for scale_a
        sa,  // scale_a: packed 4x e8m0
        0,   // opsel for scale_b
        sb   // scale_b: packed 4x e8m0
    );
}
#else
// Inline assembly fallback
__device__ __forceinline__ float4_vec mfma_scale_fp4(
    int4_vec a, int4_vec b, float4_vec c, int sa, int sb
) {
    float4_vec result = c;
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4 op_sel_hi:[0,0,0] cbsz:4 blgp:4"
        : "+v"(result)
        : "v"(a), "v"(b), "v"(sa), "v"(sb)
    );
    return result;
}
#endif

// ---------------------------------------------------------------
// FP8 E4M3FNUZ -> FP32 LUT builder (for V dequant)
// ---------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3fnuz_to_fp32(unsigned char val) {
    if (val == 0 || val == 0x80) return 0.0f;
    int sign = (val >> 7) & 1;
    int exp_bits = (val >> 3) & 0xF;
    int mant_bits = val & 0x7;
    float mantissa, result;
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
// Warp-level reductions (AMD wavefront=64)
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// ---------------------------------------------------------------
// Shared memory layout (byte offsets):
//   LUT:       256 floats = 1024 bytes         (offset 0)
//   Q FP4:     16 heads * 288 bytes = 4608 B   (offset 1024)
//   Q scales:  16 heads * 20 bytes = 320 B     (offset 5632, padded scale rows)
//   Scores:    4 waves * 16 heads * 16 tokens * 4 = 16384 B (offset ~5968, aligned)
//   Total: ~22.4 KB
// ---------------------------------------------------------------
#define SMEM_LUT_OFFSET       0
#define SMEM_LUT_SIZE_BYTES   (FP8_LUT_SIZE * 4)                              // 1024

#define SMEM_Q_FP4_OFFSET     SMEM_LUT_SIZE_BYTES                             // 1024
#define SMEM_Q_FP4_SIZE       (NUM_HEADS * PACKED_QK)                          // 16 * 288 = 4608

#define SMEM_Q_SCALE_OFFSET   (SMEM_Q_FP4_OFFSET + SMEM_Q_FP4_SIZE)          // 5632
// Pad scale rows to 20 bytes (18 rounded up to multiple of 4) for alignment
#define SCALE_ROW_STRIDE      20
#define SMEM_Q_SCALE_SIZE     (NUM_HEADS * SCALE_ROW_STRIDE)                   // 16 * 20 = 320

#define SMEM_SCORES_OFFSET_RAW (SMEM_Q_SCALE_OFFSET + SMEM_Q_SCALE_SIZE)     // 5952
#define SMEM_SCORES_OFFSET    ((SMEM_SCORES_OFFSET_RAW + 15) & ~15)           // aligned to 16
#define SMEM_SCORES_SIZE      (NUM_WARPS * NUM_HEADS * BLOCK_N * 4)            // 4*16*16*4 = 16384

#define TOTAL_SMEM_BYTES      (SMEM_SCORES_OFFSET + SMEM_SCORES_SIZE + 16)    // ~22.4 KB


// ---------------------------------------------------------------
// MXFP4 v2: Multi-wave Scaled MFMA QK + Scalar FP8 PV Split-K Kernel
//
// Grid:  (batch_size * num_kv_splits, 1)
// Block: 256 threads = 4 waves
//
// Each wave handles BLOCK_N=16 tokens via 5 scaled MFMAs for QK.
// All 256 threads cooperate for online softmax + V accumulation.
// ---------------------------------------------------------------

__global__ __launch_bounds__(256, 2)
void mla_mxfp4_v2_splitk(
    const unsigned char* __restrict__ Q_fp4,      // [batch*16, 288] packed FP4
    const unsigned char* __restrict__ Q_scale,     // [batch*16, >=18] e8m0 scales
    const unsigned char* __restrict__ K_fp4,       // [total_kv, 288] packed FP4
    const unsigned char* __restrict__ K_scale,     // [total_kv, >=18] e8m0 scales
    const unsigned char* __restrict__ V_fp8,       // [total_kv, 576] FP8 (first 512 used)
    float* __restrict__ partial_acc,               // [batch * splits * 16, V_DIM]
    float* __restrict__ partial_max,               // [batch * splits * 16]
    float* __restrict__ partial_sum_exp,           // [batch * splits * 16]
    const int* __restrict__ kv_indptr,             // [batch+1]
    float sm_scale,
    float kv_scale,                                // FP8 V scale factor
    int num_kv_splits,
    int q_scale_stride,                            // stride for Q scale rows
    int k_scale_stride                             // stride for K scale rows
) {
    int batch_split = blockIdx.x;
    int batch = batch_split / num_kv_splits;
    int split = batch_split % num_kv_splits;

    int tid = threadIdx.x;              // 0..255
    int warp_id = tid / WARP_SIZE;      // 0..3
    int lane = tid % WARP_SIZE;         // 0..63

    // MFMA lane mapping
    int lane_row = lane % 16;           // 0..15: head index (A) / token index (B) / C col
    int lane_kgrp = lane / 16;          // 0..3: K-group, each covers 32 FP4 elements

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
    unsigned char* smem_q_fp4 = (unsigned char*)(smem_raw + SMEM_Q_FP4_OFFSET);
    unsigned char* smem_q_scale = (unsigned char*)(smem_raw + SMEM_Q_SCALE_OFFSET);
    float* smem_scores = (float*)(smem_raw + SMEM_SCORES_OFFSET);

    // Step 1: Build FP8 LUT for V dequant (256 threads, 1 entry each)
    if (tid < FP8_LUT_SIZE) {
        lut[tid] = fp8_e4m3fnuz_to_fp32((unsigned char)tid);
    }

    // Step 2: Load Q FP4 data and scales into shared memory
    // Q_fp4: [batch*16, 288], Q_scale: [batch*16, q_scale_stride]
    const unsigned char* q_fp4_base = Q_fp4 + (long long)batch * NUM_HEADS * PACKED_QK;
    const unsigned char* q_scale_base = Q_scale + (long long)batch * NUM_HEADS * q_scale_stride;

    // Cooperative load of Q FP4 (16 * 288 = 4608 bytes)
    for (int i = tid; i < NUM_HEADS * PACKED_QK; i += BLOCK_SIZE) {
        smem_q_fp4[i] = q_fp4_base[i];
    }

    // Cooperative load of Q scales (16 * 18 = 288 bytes, stored with stride)
    for (int i = tid; i < NUM_HEADS * NUM_SCALE_BLOCKS; i += BLOCK_SIZE) {
        int head = i / NUM_SCALE_BLOCKS;
        int sb = i % NUM_SCALE_BLOCKS;
        smem_q_scale[head * SCALE_ROW_STRIDE + sb] = q_scale_base[head * q_scale_stride + sb];
    }

    __syncthreads();

    // Per-head online softmax state and V accumulators
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

    int v_idx0 = tid * V_DIMS_PER_THREAD;
    int v_idx1 = tid * V_DIMS_PER_THREAD + 1;

    // ===================================================================
    // Main loop: process TOKENS_PER_ITER=64 KV tokens per iteration
    // (4 waves x 16 tokens/wave)
    // ===================================================================
    for (int t_base = 0; t_base < num_tokens; t_base += TOKENS_PER_ITER) {
        int tokens_this_iter = min(TOKENS_PER_ITER, num_tokens - t_base);

        // Each wave handles 16 tokens independently
        int wave_token_base = t_base + warp_id * BLOCK_N;
        int wave_tokens = min(BLOCK_N, max(0, num_tokens - (int)(t_base + warp_id * BLOCK_N)));

        // --- QK Phase: 5 scaled MFMAs per wave ---
        float4_vec qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

        if (wave_tokens > 0) {
            #pragma unroll
            for (int mfma_idx = 0; mfma_idx < NUM_QK_MFMA; mfma_idx++) {
                // Each MFMA covers K=128 FP4 elements = 64 packed bytes
                int fp4_elem_offset = mfma_idx * MFMA_K;  // FP4 element offset
                int byte_offset = fp4_elem_offset / 2;     // packed byte offset

                // Thread layout for FP4 MFMA:
                // lane_row (0..15) = head index for A, token index for B
                // lane_kgrp (0..3) = K-group, each covers 32 FP4 = 16 bytes = 4 int32

                // Load A operand (Q): smem_q_fp4[lane_row * 288 + byte_offset + lane_kgrp*16 .. +16]
                int4_vec a_reg;
                {
                    int q_byte_base = lane_row * PACKED_QK + byte_offset + lane_kgrp * 16;
                    if (byte_offset + lane_kgrp * 16 + 15 < PACKED_QK) {
                        // Full load — within bounds
                        const unsigned char* q_src = smem_q_fp4 + q_byte_base;
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            unsigned int packed = 0;
                            packed |= (unsigned int)q_src[r * 4 + 0];
                            packed |= (unsigned int)q_src[r * 4 + 1] << 8;
                            packed |= (unsigned int)q_src[r * 4 + 2] << 16;
                            packed |= (unsigned int)q_src[r * 4 + 3] << 24;
                            a_reg[r] = (int)packed;
                        }
                    } else {
                        // Partial — near boundary (last MFMA where 576/2=288 doesn't divide evenly)
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            unsigned int packed = 0;
                            #pragma unroll
                            for (int b = 0; b < 4; b++) {
                                int byte_idx = byte_offset + lane_kgrp * 16 + r * 4 + b;
                                if (byte_idx < PACKED_QK) {
                                    packed |= (unsigned int)smem_q_fp4[lane_row * PACKED_QK + byte_idx] << (b * 8);
                                }
                            }
                            a_reg[r] = (int)packed;
                        }
                    }
                }

                // Load B operand (K): from global K_fp4
                // Thread lane needs K[token = wave_token_base + lane_row, packed bytes at byte_offset + lane_kgrp*16..+16]
                int4_vec b_reg;
                {
                    int token_global = kv_start + wave_token_base + lane_row;
                    int k_byte_base = byte_offset + lane_kgrp * 16;
                    if (lane_row < wave_tokens && k_byte_base + 15 < PACKED_QK) {
                        const unsigned char* k_src = K_fp4 + (long long)token_global * PACKED_QK + k_byte_base;
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            unsigned int packed = 0;
                            packed |= (unsigned int)k_src[r * 4 + 0];
                            packed |= (unsigned int)k_src[r * 4 + 1] << 8;
                            packed |= (unsigned int)k_src[r * 4 + 2] << 16;
                            packed |= (unsigned int)k_src[r * 4 + 3] << 24;
                            b_reg[r] = (int)packed;
                        }
                    } else if (lane_row < wave_tokens) {
                        // Partial load near boundary
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            unsigned int packed = 0;
                            #pragma unroll
                            for (int b2 = 0; b2 < 4; b2++) {
                                int byte_idx = k_byte_base + r * 4 + b2;
                                if (byte_idx < PACKED_QK) {
                                    packed |= (unsigned int)K_fp4[(long long)token_global * PACKED_QK + byte_idx] << (b2 * 8);
                                }
                            }
                            b_reg[r] = (int)packed;
                        }
                    } else {
                        // Zero pad for out-of-range tokens
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            b_reg[r] = 0;
                        }
                    }
                }

                // Build scale operands (sa, sb): ONE e8m0 byte per thread
                //
                // Scale matrix Ax has shape [M=16, K/32=4] — one scale per (row, K-group).
                // Thread at (lane_row, lane_kgrp) holds Ax[lane_row, lane_kgrp].
                // With op_sel=0, hardware reads byte 0 of the int32 scale VGPR.
                //
                // This MFMA covers FP4 elements [mfma_idx*128 .. mfma_idx*128+127]
                // With block_size=32, that's scale blocks [mfma_idx*4 + 0..3].
                // Thread with lane_kgrp=g needs scale block mfma_idx*4 + g.
                int scale_block_idx = mfma_idx * 4 + lane_kgrp;

                // sa: Q scale for head=lane_row, kgroup=lane_kgrp
                int sa;
                {
                    unsigned char sv = 0;
                    if (scale_block_idx < NUM_SCALE_BLOCKS) {
                        sv = smem_q_scale[lane_row * SCALE_ROW_STRIDE + scale_block_idx];
                    }
                    sa = (int)sv;  // byte 0 = sv, bytes 1-3 = 0
                }

                // sb: K scale for token=wave_token_base + lane_row, kgroup=lane_kgrp
                int sb;
                {
                    unsigned char sv = 0;
                    int token_global = kv_start + wave_token_base + lane_row;
                    if (lane_row < wave_tokens && scale_block_idx < NUM_SCALE_BLOCKS) {
                        sv = K_scale[(long long)token_global * k_scale_stride + scale_block_idx];
                    }
                    sb = (int)sv;  // byte 0 = sv, bytes 1-3 = 0
                }

                // Execute scaled MFMA
                qk_acc = mfma_scale_fp4(a_reg, b_reg, qk_acc, sa, sb);
            }

            // Extract QK scores to shared memory
            // qk_acc[i] = scores[head = lane_kgrp*4+i, token = lane_row]
            // Note: scaled MFMA already incorporates block scales, so we only
            // need sm_scale here (no separate q_scale * kv_scale like FP8 path)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int head_idx = lane_kgrp * 4 + i;
                int token_idx = lane_row;
                float score_val = (token_idx < wave_tokens) ? (qk_acc[i] * sm_scale) : -1e30f;
                smem_scores[warp_id * NUM_HEADS * BLOCK_N + head_idx * BLOCK_N + token_idx] = score_val;
            }
        } else {
            // Wave has no tokens — fill with -inf
            for (int i = lane; i < NUM_HEADS * BLOCK_N; i += WARP_SIZE) {
                smem_scores[warp_id * NUM_HEADS * BLOCK_N + i] = -1e30f;
            }
        }

        __syncthreads();

        // --- Online softmax + V accumulation (all 256 threads) ---
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
            // V comes from FP8 buffer, dequant via LUT
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                int w_base = t_base + w * BLOCK_N;
                int w_tokens = min(BLOCK_N, max(0, tokens_this_iter - w * BLOCK_N));
                float* wave_scores = smem_scores + w * NUM_HEADS * BLOCK_N + h * BLOCK_N;

                for (int t = 0; t < w_tokens; t++) {
                    float w_val = expf(wave_scores[t] - new_max);
                    head_sum_exp[h] += w_val;

                    int kv_idx = kv_start + w_base + t;
                    // V is from FP8 buffer, layout [total_kv, 576], first 512 dims used
                    const unsigned char* v_row = V_fp8 + (long long)kv_idx * QK_DIM;

                    if (v_idx0 < V_DIM) {
                        v_acc[h][0] += w_val * lut[v_row[v_idx0]];
                    }
                    if (v_idx1 < V_DIM) {
                        v_acc[h][1] += w_val * lut[v_row[v_idx1]];
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

    float global_max = -1e30f;
    for (int s = 0; s < num_kv_splits; s++) {
        int idx = (batch * num_kv_splits + s) * NUM_HEADS + head;
        global_max = fmaxf(global_max, partial_max[idx]);
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

        global_sum_exp += se * rescale;

        const float* p_acc = partial_acc + (long long)idx * V_DIM;
        if (v_idx0 < V_DIM) {
            final_v0 += p_acc[v_idx0] * rescale;
        }
        if (v_idx1 < V_DIM) {
            final_v1 += p_acc[v_idx1] * rescale;
        }
    }

    // Normalize and apply kv_scale to V (FP8 V needs kv_scale)
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
torch::Tensor mla_mxfp4_v2_forward(
    torch::Tensor Q_fp4,        // [batch*16, 288] uint8 packed FP4
    torch::Tensor Q_scale,      // [batch*16, >=18] uint8 e8m0 scales
    torch::Tensor K_fp4,        // [total_kv, 288] uint8 packed FP4
    torch::Tensor K_scale,      // [total_kv, >=18] uint8 e8m0 scales
    torch::Tensor V_fp8,        // [total_kv, 576] uint8 FP8
    torch::Tensor kv_indptr,    // [batch+1] int32
    float sm_scale,
    float kv_scale,
    int batch_size,
    int num_kv_splits
) {
    const int v_dim = V_DIM;
    const int num_heads = NUM_HEADS;

    // Ensure contiguous
    Q_fp4 = Q_fp4.contiguous();
    Q_scale = Q_scale.contiguous();
    K_fp4 = K_fp4.contiguous();
    K_scale = K_scale.contiguous();
    V_fp8 = V_fp8.contiguous();

    int q_scale_stride = Q_scale.size(1);
    int k_scale_stride = K_scale.size(1);

    // Allocate partial results
    int num_partials = batch_size * num_kv_splits * num_heads;
    auto partial_acc = torch::zeros({num_partials, v_dim},
                                     torch::dtype(torch::kFloat32).device(Q_fp4.device()));
    auto partial_max = torch::full({num_partials}, -1e30f,
                                    torch::dtype(torch::kFloat32).device(Q_fp4.device()));
    auto partial_sum_exp = torch::zeros({num_partials},
                                         torch::dtype(torch::kFloat32).device(Q_fp4.device()));

    int smem_bytes = TOTAL_SMEM_BYTES;

    // Launch split-K MFMA kernel
    dim3 grid_splitk(batch_size * num_kv_splits, 1);
    dim3 block_splitk(BLOCK_SIZE);

    mla_mxfp4_v2_splitk<<<grid_splitk, block_splitk, smem_bytes>>>(
        (const unsigned char*)Q_fp4.data_ptr<int8_t>(),
        (const unsigned char*)Q_scale.data_ptr<int8_t>(),
        (const unsigned char*)K_fp4.data_ptr<int8_t>(),
        (const unsigned char*)K_scale.data_ptr<int8_t>(),
        (const unsigned char*)V_fp8.data_ptr<int8_t>(),
        partial_acc.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        partial_sum_exp.data_ptr<float>(),
        kv_indptr.data_ptr<int>(),
        sm_scale,
        kv_scale,
        num_kv_splits,
        q_scale_stride,
        k_scale_stride
    );

    // Allocate output
    int total_heads = batch_size * num_heads;
    auto output = torch::zeros({total_heads, v_dim},
                                torch::dtype(torch::kFloat32).device(Q_fp4.device()));

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

torch::Tensor mla_mxfp4_v2_forward(
    torch::Tensor Q_fp4,
    torch::Tensor Q_scale,
    torch::Tensor K_fp4,
    torch::Tensor K_scale,
    torch::Tensor V_fp8,
    torch::Tensor kv_indptr,
    float sm_scale,
    float kv_scale,
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
            name="mla_mxfp4_v2",
            cpp_sources=cpp_source,
            cuda_sources=hip_source,
            functions=["mla_mxfp4_v2_forward"],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "-mcpu=gfx950",
                "-mllvm", "-amdgpu-early-inline-all=true",
                "-mllvm", "-amdgpu-function-calls=false",
            ],
        )
        _hip_available = True
        print("[HIP MXFP4 v2] Compilation SUCCESS (scaled MFMA builtin)")
    except Exception as e:
        _hip_available = False
        print(f"[HIP MXFP4 v2] Compilation FAILED: {e}")
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
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant
    _aiter_available = True
    FP8_DTYPE = aiter_dtypes.fp8
except ImportError:
    FP8_DTYPE = torch.float8_e4m3fnuz
    try:
        from aiter.utility.fp4_utils import dynamic_mxfp4_quant
    except ImportError:
        dynamic_mxfp4_quant = None

_meta_cache = {}
_output_cache = {}


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
# HIP MXFP4 PATH
# ===================================================================

def _hip_mxfp4_path(q, kv_data, kv_indptr, config):
    """Primary path: HIP MXFP4 scaled MFMA kernel."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    num_kv_splits = KV_SPLITS_MAP.get((batch_size, kv_seq_len), 16)

    # Get MXFP4 KV data
    kv_fp4, kv_fp4_scale = kv_data["mxfp4"]

    # Quantize Q to MXFP4
    q_2d = q.view(-1, QK_HEAD_DIM)
    q_packed_raw, q_scale_raw = dynamic_mxfp4_quant(q_2d)
    q_packed = q_packed_raw.view(torch.uint8).contiguous()
    q_scale = q_scale_raw.view(torch.uint8).contiguous()

    # Reshape K data: kv_fp4 may be (total_kv, 1, 288) or (total_kv, 288)
    k_fp4 = kv_fp4.reshape(-1, PACKED_QK).contiguous()
    k_fp4_bytes = k_fp4.view(torch.uint8) if k_fp4.dtype != torch.uint8 else k_fp4

    # K scales
    k_scale_raw = kv_fp4_scale.view(torch.uint8) if kv_fp4_scale.dtype != torch.uint8 else kv_fp4_scale
    total_kv = k_fp4_bytes.shape[0]
    k_scale = k_scale_raw.reshape(total_kv, -1).contiguous()

    # Ensure Q data is properly 2D
    q_packed = q_packed.reshape(-1, PACKED_QK)
    q_scale = q_scale.reshape(q_2d.shape[0], -1).contiguous()

    # Get FP8 V data: kv_fp8 is (total_kv, 1, 576) FP8, reshape to (total_kv, 576)
    kv_fp8, kv_fp8_scale = kv_data["fp8"]
    v_fp8_bytes = kv_fp8.reshape(-1, QK_HEAD_DIM).contiguous().view(torch.int8)
    kv_scale_val = kv_fp8_scale.item()

    output = _hip_module.mla_mxfp4_v2_forward(
        q_packed.view(torch.int8),
        q_scale.view(torch.int8),
        k_fp4_bytes.view(torch.int8),
        k_scale.view(torch.int8),
        v_fp8_bytes,
        kv_indptr,
        SM_SCALE,
        kv_scale_val,
        batch_size,
        num_kv_splits,
    )

    return output


# ===================================================================
# ENTRY POINT
# ===================================================================

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    if _hip_available:
        try:
            return _hip_mxfp4_path(q, kv_data, kv_indptr, config)
        except Exception as e:
            print(f"[HIP MXFP4 v2] Runtime error, falling back to AITER: {e}")

    if _aiter_available:
        return _aiter_fallback(data)

    raise RuntimeError("Neither HIP MXFP4 v2 kernel nor AITER available")
