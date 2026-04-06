"""
MXFP4 v5: v3b accuracy fix + v4 LDS V loading optimization.

Based on v4 (FP8 Q x MXFP4 K Scaled MFMA QK + BF16 MFMA PV + LDS V).

Combines:
  - v3b: Corrected QK score extraction (head_idx = lane_row, token_idx = lane_kgrp*4+i)
  - v4: FP8->BF16 lookup table (256 entries) built in shared memory at kernel start
  - v4: V data loaded cooperatively into LDS with coalesced global reads
  - v4: PV MFMA reads V operands from LDS instead of individual global memory reads

QK Phase: FP8 Q x MXFP4 K scaled MFMA (5 calls per 16-token tile, 4 waves independent)
PV Phase: BF16 MFMA (each wave handles 128 V dims, 32 tokens per MFMA call)

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
PACKED_QK: int = 288       # 576 / 2 packed bytes for FP4
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

// FP8 Q: 576 bytes per head (1 byte per element)
// MXFP4 K: 288 packed bytes per token (2 FP4 values per byte)
#define Q_FP8_BYTES  576
#define PACKED_QK    288
#define NUM_SCALE_BLOCKS 18   // 576/32 = 18

// MFMA K=128 elements per instruction
// For FP8 A (cbsz=0): 8 x int32 = 32 bytes = 32 FP8 elements per kgroup, 4 kgroups = 128
// For FP4 B (blgp=4): 4 x int32 = 16 bytes = 32 FP4 elements per kgroup, 4 kgroups = 128
#define MFMA_K       128
#define NUM_QK_MFMA  5      // ceil(576/128) = 5 MFMA calls
#define FP8_BYTES_PER_MFMA_K  128  // 128 FP8 elements = 128 bytes
#define FP4_BYTES_PER_MFMA_K  64   // 128 FP4 elements = 64 bytes

// BF16 MFMA for PV: v_mfma_f32_16x16x32_bf16
#define PV_MFMA_K    32     // tokens per BF16 MFMA call
#define PV_MFMA_N    16     // V dims per MFMA call

#define REDUCE_BLOCK 256

// FP8 LUT: 256 entries * 2 bytes (bf16) = 512 bytes
#define FP8_LUT_SIZE  512

// V tile in LDS: per-wave buffer, 64 tokens * 16 v_dims * 2 bytes (bf16) = 2048 bytes each
// 4 waves * 2048 = 8192 bytes total
// Each wave independently loads its own V chunk data using 64 threads
#define V_TILE_TOKENS  TOKENS_PER_ITER  // 64
#define V_TILE_DIMS    PV_MFMA_N        // 16
#define V_TILE_PER_WAVE  (V_TILE_TOKENS * V_TILE_DIMS * 2)  // 2048 bytes per wave
#define V_TILE_TOTAL     (NUM_WARPS * V_TILE_PER_WAVE)       // 8192 bytes

// ---------------------------------------------------------------
// Vector type definitions
// ---------------------------------------------------------------
typedef int    int4_vec   __attribute__((ext_vector_type(4)));
typedef int    int8_vec   __attribute__((ext_vector_type(8)));
typedef float  float4_vec __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------
// Scaled MFMA wrapper: FP8 Q x MXFP4 K
// ---------------------------------------------------------------
__device__ __forceinline__ float4_vec mfma_scale_fp8_fp4(
    int8_vec a, int4_vec b, float4_vec c, int sa, int sb
) {
    int8_vec b_pad = {b[0], b[1], b[2], b[3], 0, 0, 0, 0};
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b_pad, c,
        0,   // cbsz: 0 = FP8 for A
        4,   // blgp: 4 = FP4 for B
        0,   // opsel for scale_a
        sa,  // scale_a (e8m0 = 127 for 1.0)
        0,   // opsel for scale_b
        sb   // scale_b (K block scale)
    );
}

// ---------------------------------------------------------------
// BF16 MFMA wrapper for PV
// v_mfma_f32_16x16x32_bf16: C[16x16] += A[16x32] * B[32x16]
// ---------------------------------------------------------------
__device__ __forceinline__ float4_vec mfma_bf16_16x16x32(
    int4_vec a, int4_vec b, float4_vec c
) {
    float4_vec result = c;
    asm volatile(
        "v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0"
        : "+v"(result)
        : "v"(a), "v"(b)
    );
    return result;
}

// ---------------------------------------------------------------
// FP8 E4M3FNUZ -> FP32 conversion (used only for LUT building)
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
// FP32 -> BF16 (truncate) as unsigned short
// ---------------------------------------------------------------
__device__ __forceinline__ unsigned short fp32_to_bf16_bits(float val) {
    union { float f; unsigned int u; } cvt;
    cvt.f = val;
    return (unsigned short)(cvt.u >> 16);
}

// ---------------------------------------------------------------
// Shared memory layout:
//   Q FP8:      16 heads * 576 bytes = 9216 B         (offset 0)
//   Scores:     4 waves * 16 heads * 16 tokens * 4 = 16384 B
//   P_bf16:     NUM_HEADS * TOKENS_PER_ITER * 2 = 2048 B
//   FP8 LUT:    256 * 2 bytes = 512 B (bf16 LUT)
//   V tile:     4 waves * 64 tokens * 16 dims * 2 bytes = 8192 B
//   Total: ~36 KB
// ---------------------------------------------------------------
#define SMEM_Q_FP8_OFFSET     0
#define SMEM_Q_FP8_SIZE       (NUM_HEADS * Q_FP8_BYTES)                        // 9216

#define SMEM_SCORES_OFFSET_RAW (SMEM_Q_FP8_OFFSET + SMEM_Q_FP8_SIZE)         // 9216
#define SMEM_SCORES_OFFSET    ((SMEM_SCORES_OFFSET_RAW + 15) & ~15)           // aligned
#define SMEM_SCORES_SIZE      (NUM_WARPS * NUM_HEADS * BLOCK_N * 4)            // 16384

#define SMEM_P_BF16_OFFSET_RAW (SMEM_SCORES_OFFSET + SMEM_SCORES_SIZE)
#define SMEM_P_BF16_OFFSET    ((SMEM_P_BF16_OFFSET_RAW + 15) & ~15)
#define SMEM_P_BF16_SIZE      (NUM_HEADS * TOKENS_PER_ITER * 2)               // 2048

#define SMEM_FP8_LUT_OFFSET_RAW (SMEM_P_BF16_OFFSET + SMEM_P_BF16_SIZE)
#define SMEM_FP8_LUT_OFFSET   ((SMEM_FP8_LUT_OFFSET_RAW + 15) & ~15)
#define SMEM_FP8_LUT_SIZE     FP8_LUT_SIZE                                    // 512

#define SMEM_V_TILE_OFFSET_RAW (SMEM_FP8_LUT_OFFSET + SMEM_FP8_LUT_SIZE)
#define SMEM_V_TILE_OFFSET    ((SMEM_V_TILE_OFFSET_RAW + 15) & ~15)
#define SMEM_V_TILE_SIZE      V_TILE_TOTAL                                    // 8192

#define TOTAL_SMEM_BYTES      (SMEM_V_TILE_OFFSET + SMEM_V_TILE_SIZE + 16)   // ~36 KB


// ---------------------------------------------------------------
// MXFP4 v5: Optimized V loading via LDS + FP8 LUT
// ---------------------------------------------------------------

__global__ __launch_bounds__(256, 2)
void mla_mxfp4_v5_splitk(
    const unsigned char* __restrict__ Q_fp8,       // [batch*16, 576] FP8 Q data
    const unsigned char* __restrict__ K_fp4,       // [total_kv, 288] packed FP4
    const unsigned char* __restrict__ K_scale,     // [total_kv, >=18] e8m0 scales
    const unsigned char* __restrict__ V_fp8,       // [total_kv, 576] FP8 (first 512 used)
    float* __restrict__ partial_acc,               // [batch * splits * 16, V_DIM]
    float* __restrict__ partial_max,               // [batch * splits * 16]
    float* __restrict__ partial_sum_exp,           // [batch * splits * 16]
    const int* __restrict__ kv_indptr,             // [batch+1]
    float sm_scale,                                // SM_SCALE * q_fp8_scale
    float kv_scale,                                // FP8 V scale factor
    int num_kv_splits,
    int k_scale_stride
) {
    int batch_split = blockIdx.x;
    int batch = batch_split / num_kv_splits;
    int split = batch_split % num_kv_splits;

    int tid = threadIdx.x;              // 0..255
    int warp_id = tid / WARP_SIZE;      // 0..3
    int lane = tid % WARP_SIZE;         // 0..63

    int lane_row = lane % 16;           // 0..15
    int lane_kgrp = lane / 16;          // 0..3

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
    unsigned char* smem_q_fp8 = (unsigned char*)(smem_raw + SMEM_Q_FP8_OFFSET);
    float* smem_scores = (float*)(smem_raw + SMEM_SCORES_OFFSET);
    unsigned short* smem_p_bf16 = (unsigned short*)(smem_raw + SMEM_P_BF16_OFFSET);
    unsigned short* smem_fp8_lut = (unsigned short*)(smem_raw + SMEM_FP8_LUT_OFFSET);
    unsigned short* smem_v_tile = (unsigned short*)(smem_raw + SMEM_V_TILE_OFFSET);

    // ---------------------------------------------------------------
    // Build FP8 -> BF16 lookup table in shared memory
    // 256 entries, each thread handles 1 entry (256 threads, 256 entries)
    // ---------------------------------------------------------------
    {
        unsigned char fp8_val = (unsigned char)tid;
        float fval = fp8_e4m3fnuz_to_fp32(fp8_val);
        smem_fp8_lut[tid] = fp32_to_bf16_bits(fval);
    }

    // Load Q FP8 data into shared memory (16 heads * 576 bytes = 9216 bytes)
    const unsigned char* q_fp8_base = Q_fp8 + (long long)batch * NUM_HEADS * Q_FP8_BYTES;
    for (int i = tid; i < NUM_HEADS * Q_FP8_BYTES; i += BLOCK_SIZE) {
        smem_q_fp8[i] = q_fp8_base[i];
    }

    __syncthreads();

    // Per-head online softmax state
    float head_max[NUM_HEADS];
    float head_sum_exp[NUM_HEADS];

    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        head_max[h] = -1e30f;
        head_sum_exp[h] = 0.0f;
    }

    // PV output accumulators
    // Wave w handles V dim chunks [w*8 .. w*8+7] (128 V dims per wave, 512 total).
    // Per thread: 8 chunks * 4 floats = 32 floats.
    float v_acc[8][4];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        v_acc[c][0] = 0.0f; v_acc[c][1] = 0.0f;
        v_acc[c][2] = 0.0f; v_acc[c][3] = 0.0f;
    }

    int my_head = lane_row;  // head this thread accumulates for PV

    // ===================================================================
    // Main loop: process 64 KV tokens per iteration (4 waves x 16)
    // ===================================================================
    for (int t_base = 0; t_base < num_tokens; t_base += TOKENS_PER_ITER) {
        int tokens_this_iter = min(TOKENS_PER_ITER, num_tokens - t_base);

        // ---------------------------------------------------------------
        // Phase 1: QK via FP8 x MXFP4 Scaled MFMA
        // ---------------------------------------------------------------
        int wave_token_base = t_base + warp_id * BLOCK_N;
        int wave_tokens = min(BLOCK_N, max(0, num_tokens - (int)(t_base + warp_id * BLOCK_N)));

        float4_vec qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

        if (wave_tokens > 0) {
            #pragma unroll
            for (int mfma_idx = 0; mfma_idx < NUM_QK_MFMA; mfma_idx++) {
                int elem_offset = mfma_idx * MFMA_K;  // element offset into QK_DIM

                // Load A operand (Q FP8): 8 x int32 = 32 FP8 bytes per thread
                int8_vec a_reg;
                {
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        int byte_idx = elem_offset + lane_kgrp * 32 + r * 4;
                        if (byte_idx + 3 < QK_DIM) {
                            const unsigned char* q_src = smem_q_fp8 + lane_row * Q_FP8_BYTES + byte_idx;
                            unsigned int packed = 0;
                            packed |= (unsigned int)q_src[0];
                            packed |= (unsigned int)q_src[1] << 8;
                            packed |= (unsigned int)q_src[2] << 16;
                            packed |= (unsigned int)q_src[3] << 24;
                            a_reg[r] = (int)packed;
                        } else {
                            unsigned int packed = 0;
                            for (int b = 0; b < 4; b++) {
                                int bi = byte_idx + b;
                                if (bi < QK_DIM) {
                                    packed |= (unsigned int)smem_q_fp8[lane_row * Q_FP8_BYTES + bi] << (b * 8);
                                }
                            }
                            a_reg[r] = (int)packed;
                        }
                    }
                }

                // Load B operand (K MXFP4): 4 x int32 = 16 bytes = 32 FP4 elements per kgroup
                int fp4_byte_offset = elem_offset / 2;
                int4_vec b_reg;
                {
                    int token_global = kv_start + wave_token_base + lane_row;
                    int k_byte_base = fp4_byte_offset + lane_kgrp * 16;
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
                        #pragma unroll
                        for (int r = 0; r < 4; r++) b_reg[r] = 0;
                    }
                }

                // Scale operands
                int sa = 127;
                int scale_block_idx = mfma_idx * 4 + lane_kgrp;
                int sb;
                {
                    unsigned char sv = 0;
                    int token_global = kv_start + wave_token_base + lane_row;
                    if (lane_row < wave_tokens && scale_block_idx < NUM_SCALE_BLOCKS) {
                        sv = K_scale[(long long)token_global * k_scale_stride + scale_block_idx];
                    }
                    sb = (int)sv;
                }

                qk_acc = mfma_scale_fp8_fp4(a_reg, b_reg, qk_acc, sa, sb);
            }

            // Extract QK scores to shared memory
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int head_idx = lane_row;
                int token_idx = lane_kgrp * 4 + i;
                float score_val = (token_idx < wave_tokens) ? (qk_acc[i] * sm_scale) : -1e30f;
                smem_scores[warp_id * NUM_HEADS * BLOCK_N + head_idx * BLOCK_N + token_idx] = score_val;
            }
        } else {
            for (int i = lane; i < NUM_HEADS * BLOCK_N; i += WARP_SIZE) {
                smem_scores[warp_id * NUM_HEADS * BLOCK_N + i] = -1e30f;
            }
        }

        __syncthreads();

        // ---------------------------------------------------------------
        // Phase 2: Online Softmax + Store P as bf16
        // ---------------------------------------------------------------

        #pragma unroll 1
        for (int h = 0; h < NUM_HEADS; h++) {
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
            float rescale = expf(head_max[h] - new_max);

            if (h == my_head) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    v_acc[c][0] *= rescale;
                    v_acc[c][1] *= rescale;
                    v_acc[c][2] *= rescale;
                    v_acc[c][3] *= rescale;
                }
            }

            head_sum_exp[h] *= rescale;
            head_max[h] = new_max;

            float local_sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                int w_tokens = min(BLOCK_N, max(0, tokens_this_iter - w * BLOCK_N));
                float* wave_scores = smem_scores + w * NUM_HEADS * BLOCK_N + h * BLOCK_N;
                int token_base_in_iter = w * BLOCK_N;
                for (int t = 0; t < w_tokens; t++) {
                    float w_val = expf(wave_scores[t] - new_max);
                    local_sum += w_val;
                    smem_p_bf16[h * TOKENS_PER_ITER + token_base_in_iter + t] = fp32_to_bf16_bits(w_val);
                }
                for (int t = w_tokens; t < BLOCK_N; t++) {
                    smem_p_bf16[h * TOKENS_PER_ITER + token_base_in_iter + t] = 0;
                }
            }
            head_sum_exp[h] += local_sum;
        }

        __syncthreads();

        // ---------------------------------------------------------------
        // Phase 3: PV via BF16 MFMA with LDS-based V loading
        //
        // Each wave handles 8 V-dim chunks of 16 = 128 V dims.
        // Wave w handles V dims [w*128 .. w*128+127].
        //
        // Each wave has its own V tile buffer in LDS (2048 bytes each).
        // For each V-dim chunk:
        //   1. Wave's 64 threads cooperatively load V fp8 data for
        //      64 tokens x 16 dims = 1024 bytes, convert via LUT to bf16
        //   2. Wave-level sync (__builtin_amdgcn_s_barrier)
        //   3. MFMA reads V operands from wave's LDS buffer
        //
        // V tile layout per wave: [token][v_dim] as bf16, 64*16 entries
        // ---------------------------------------------------------------

        // Each wave gets its own V tile buffer
        unsigned short* my_v_tile = smem_v_tile + warp_id * (V_TILE_PER_WAVE / 2);

        int v_chunk_start = warp_id * 8;

        #pragma unroll 1
        for (int vc = 0; vc < 8; vc++) {
            int v_chunk_idx = v_chunk_start + vc;
            int v_dim_base = v_chunk_idx * PV_MFMA_N;

            // ---------------------------------------------------------------
            // Wave-local V loading into its LDS buffer
            // 64 threads per wave load 64 tokens * 16 V_dims = 1024 FP8 bytes
            // Each thread loads 16 bytes (16 FP8 values), converts via LUT
            // 64 threads * 16 = 1024 bytes = exact coverage
            // ---------------------------------------------------------------
            {
                // Each of 64 lanes handles one token's 16 V dims
                // lane = token index within the 64-token iteration
                int t_local = lane;  // 0..63

                if (t_local < tokens_this_iter && v_dim_base < V_DIM) {
                    int kv_idx = kv_start + t_base + t_local;
                    long long v_base_addr = (long long)kv_idx * QK_DIM + v_dim_base;
                    const unsigned char* v_ptr = V_fp8 + v_base_addr;

                    // Load 16 FP8 bytes as 4 x uint32 (vectorized)
                    // All 16 V dims fit since V_DIM=512 and chunks are 16-aligned
                    unsigned int w0 = *reinterpret_cast<const unsigned int*>(v_ptr);
                    unsigned int w1 = *reinterpret_cast<const unsigned int*>(v_ptr + 4);
                    unsigned int w2 = *reinterpret_cast<const unsigned int*>(v_ptr + 8);
                    unsigned int w3 = *reinterpret_cast<const unsigned int*>(v_ptr + 12);

                    // Convert via LUT and store to LDS
                    int tile_base = t_local * V_TILE_DIMS;

                    // Unrolled: 16 LUT lookups + stores
                    my_v_tile[tile_base +  0] = smem_fp8_lut[(w0 >>  0) & 0xFF];
                    my_v_tile[tile_base +  1] = smem_fp8_lut[(w0 >>  8) & 0xFF];
                    my_v_tile[tile_base +  2] = smem_fp8_lut[(w0 >> 16) & 0xFF];
                    my_v_tile[tile_base +  3] = smem_fp8_lut[(w0 >> 24) & 0xFF];
                    my_v_tile[tile_base +  4] = smem_fp8_lut[(w1 >>  0) & 0xFF];
                    my_v_tile[tile_base +  5] = smem_fp8_lut[(w1 >>  8) & 0xFF];
                    my_v_tile[tile_base +  6] = smem_fp8_lut[(w1 >> 16) & 0xFF];
                    my_v_tile[tile_base +  7] = smem_fp8_lut[(w1 >> 24) & 0xFF];
                    my_v_tile[tile_base +  8] = smem_fp8_lut[(w2 >>  0) & 0xFF];
                    my_v_tile[tile_base +  9] = smem_fp8_lut[(w2 >>  8) & 0xFF];
                    my_v_tile[tile_base + 10] = smem_fp8_lut[(w2 >> 16) & 0xFF];
                    my_v_tile[tile_base + 11] = smem_fp8_lut[(w2 >> 24) & 0xFF];
                    my_v_tile[tile_base + 12] = smem_fp8_lut[(w3 >>  0) & 0xFF];
                    my_v_tile[tile_base + 13] = smem_fp8_lut[(w3 >>  8) & 0xFF];
                    my_v_tile[tile_base + 14] = smem_fp8_lut[(w3 >> 16) & 0xFF];
                    my_v_tile[tile_base + 15] = smem_fp8_lut[(w3 >> 24) & 0xFF];
                } else {
                    // Zero-fill for out-of-range tokens
                    int tile_base = t_local * V_TILE_DIMS;
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        my_v_tile[tile_base + d] = 0;
                    }
                }
            }

            // Wave-level barrier to ensure V tile is fully loaded
            // All 64 threads in this wave must complete their stores before MFMA reads
            __syncthreads();

            // ---------------------------------------------------------------
            // PV MFMA: read V operands from wave's LDS buffer
            // ---------------------------------------------------------------
            float4_vec pv_acc = {v_acc[vc][0], v_acc[vc][1], v_acc[vc][2], v_acc[vc][3]};

            for (int t_chunk = 0; t_chunk < tokens_this_iter; t_chunk += PV_MFMA_K) {
                // Load A (P weights for this head/token group) from smem_p_bf16
                int4_vec a_reg;
                {
                    int head = lane_row;
                    int t_start = t_chunk + lane_kgrp * 8;
                    unsigned short vals[8];
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int t_idx = t_start + k;
                        if (t_idx < tokens_this_iter) {
                            vals[k] = smem_p_bf16[head * TOKENS_PER_ITER + t_idx];
                        } else {
                            vals[k] = 0;
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        a_reg[r] = (int)((unsigned int)vals[r * 2] | ((unsigned int)vals[r * 2 + 1] << 16));
                    }
                }

                // Load B (V data from wave's LDS buffer, already bf16!)
                int4_vec b_reg;
                {
                    int v_dim_in_chunk = lane_row;  // 0..15 within this 16-dim chunk
                    int t_start = t_chunk + lane_kgrp * 8;
                    unsigned short vals[8];
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        int t_idx = t_start + k;
                        if (t_idx < tokens_this_iter) {
                            vals[k] = my_v_tile[t_idx * V_TILE_DIMS + v_dim_in_chunk];
                        } else {
                            vals[k] = 0;
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        b_reg[r] = (int)((unsigned int)vals[r * 2] | ((unsigned int)vals[r * 2 + 1] << 16));
                    }
                }

                pv_acc = mfma_bf16_16x16x32(a_reg, b_reg, pv_acc);
            }

            v_acc[vc][0] = pv_acc[0];
            v_acc[vc][1] = pv_acc[1];
            v_acc[vc][2] = pv_acc[2];
            v_acc[vc][3] = pv_acc[3];

            __syncthreads();
        }

        __syncthreads();
    }

    // ---------------------------------------------------------------
    // Write partial results
    // ---------------------------------------------------------------

    for (int h = 0; h < NUM_HEADS; h++) {
        int partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + h;
        if (tid == 0) {
            partial_max[partial_idx] = head_max[h];
            partial_sum_exp[partial_idx] = head_sum_exp[h];
        }
    }

    // Write V accumulator
    {
        int my_head_partial_idx = (batch * num_kv_splits + split) * NUM_HEADS + my_head;
        float* p_acc = partial_acc + (long long)my_head_partial_idx * V_DIM;

        #pragma unroll
        for (int vc = 0; vc < 8; vc++) {
            int v_dim_base = (warp_id * 8 + vc) * PV_MFMA_N;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int v_dim = v_dim_base + lane_kgrp * 4 + i;
                if (v_dim < V_DIM) {
                    p_acc[v_dim] = v_acc[vc][i];
                }
            }
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
    int v_idx0 = tid * 2;
    int v_idx1 = tid * 2 + 1;
    float final_v0 = 0.0f;
    float final_v1 = 0.0f;

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
torch::Tensor mla_mxfp4_v5_forward(
    torch::Tensor Q_fp8,
    torch::Tensor K_fp4,
    torch::Tensor K_scale,
    torch::Tensor V_fp8,
    torch::Tensor kv_indptr,
    float sm_scale,
    float kv_scale,
    int batch_size,
    int num_kv_splits
) {
    const int v_dim = V_DIM;
    const int num_heads = NUM_HEADS;

    Q_fp8 = Q_fp8.contiguous();
    K_fp4 = K_fp4.contiguous();
    K_scale = K_scale.contiguous();
    V_fp8 = V_fp8.contiguous();

    int k_scale_stride = K_scale.size(1);

    int num_partials = batch_size * num_kv_splits * num_heads;
    auto partial_acc = torch::zeros({num_partials, v_dim},
                                     torch::dtype(torch::kFloat32).device(Q_fp8.device()));
    auto partial_max = torch::full({num_partials}, -1e30f,
                                    torch::dtype(torch::kFloat32).device(Q_fp8.device()));
    auto partial_sum_exp = torch::zeros({num_partials},
                                         torch::dtype(torch::kFloat32).device(Q_fp8.device()));

    int smem_bytes = TOTAL_SMEM_BYTES;

    dim3 grid_splitk(batch_size * num_kv_splits, 1);
    dim3 block_splitk(BLOCK_SIZE);

    mla_mxfp4_v5_splitk<<<grid_splitk, block_splitk, smem_bytes>>>(
        (const unsigned char*)Q_fp8.data_ptr<int8_t>(),
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
        k_scale_stride
    );

    int total_heads = batch_size * num_heads;
    auto output = torch::zeros({total_heads, v_dim},
                                torch::dtype(torch::kFloat32).device(Q_fp8.device()));

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

torch::Tensor mla_mxfp4_v5_forward(
    torch::Tensor Q_fp8,
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
            name="mla_mxfp4_v5",
            cpp_sources=cpp_source,
            cuda_sources=hip_source,
            functions=["mla_mxfp4_v5_forward"],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "-mcpu=gfx950",
                "-mllvm", "-amdgpu-early-inline-all=true",
                "-mllvm", "-amdgpu-function-calls=false",
            ],
        )
        _hip_available = True
        print("[HIP MXFP4 v5] Compilation SUCCESS (FP8 LUT + LDS V loading)")
    except Exception as e:
        _hip_available = False
        print(f"[HIP MXFP4 v5] Compilation FAILED: {e}")
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

try:
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant
except ImportError:
    dynamic_mxfp4_quant = None

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
# FP8 Q quantization helper
# ===================================================================

def _quantize_q_fp8(q_2d):
    """Quantize Q to FP8 E4M3FNUZ, return (q_fp8_bytes, scale_float)."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = q_2d.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    q_fp8 = (q_2d / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return q_fp8, scale.item()


# ===================================================================
# HIP MXFP4 v5 PATH
# ===================================================================

def _hip_mxfp4_path(q, kv_data, kv_indptr, config):
    """Primary path: HIP MXFP4 v5 kernel (FP8 QK + LDS-optimized BF16 PV)."""
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    num_kv_splits = KV_SPLITS_MAP.get((batch_size, kv_seq_len), 16)

    # Get MXFP4 KV data (for K)
    kv_fp4, kv_fp4_scale = kv_data["mxfp4"]

    # Quantize Q to FP8
    q_2d = q.view(-1, QK_HEAD_DIM)
    q_fp8, q_fp8_scale = _quantize_q_fp8(q_2d)
    q_fp8_bytes = q_fp8.view(torch.uint8).contiguous().reshape(-1, QK_HEAD_DIM)

    # Effective sm_scale includes Q FP8 scale: SM_SCALE * q_fp8_scale
    effective_sm_scale = SM_SCALE * q_fp8_scale

    # Reshape K data
    k_fp4 = kv_fp4.reshape(-1, PACKED_QK).contiguous()
    k_fp4_bytes = k_fp4.view(torch.uint8) if k_fp4.dtype != torch.uint8 else k_fp4

    # K scales
    k_scale_raw = kv_fp4_scale.view(torch.uint8) if kv_fp4_scale.dtype != torch.uint8 else kv_fp4_scale
    total_kv = k_fp4_bytes.shape[0]
    k_scale = k_scale_raw.reshape(total_kv, -1).contiguous()

    # Get FP8 V data
    kv_fp8, kv_fp8_scale = kv_data["fp8"]
    v_fp8_bytes = kv_fp8.reshape(-1, QK_HEAD_DIM).contiguous().view(torch.int8)
    kv_scale_val = kv_fp8_scale.item()

    output = _hip_module.mla_mxfp4_v5_forward(
        q_fp8_bytes.view(torch.int8),
        k_fp4_bytes.view(torch.int8),
        k_scale.view(torch.int8),
        v_fp8_bytes,
        kv_indptr,
        effective_sm_scale,
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
            print(f"[HIP MXFP4 v5] Runtime error, falling back to AITER: {e}")

    if _aiter_available:
        return _aiter_fallback(data)

    raise RuntimeError("Neither HIP MXFP4 v5 kernel nor AITER available")
