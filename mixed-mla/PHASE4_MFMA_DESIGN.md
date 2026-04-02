# Phase 4 Design: MXFP4 MFMA Hardware MLA Decode Kernel

## Executive Summary

Replace Phase 3.5's scalar FP8 LUT-based dequantization with hardware MFMA
instructions (`v_mfma_f32_16x16x128_f8f6f4`) that natively consume FP8/FP4 data
and produce FP32 accumulators. This eliminates the per-element dequant bottleneck
that causes 5-35x slowdown on large configs.

**Target instruction**: `v_mfma_f32_16x16x128_f8f6f4` (unscaled FP8) or
`v_mfma_scale_f32_16x16x128_f8f6f4` (MXFP4 with E8M0 block scales).

---

## 1. Instruction Reference

### 1.1 Unscaled MFMA (FP8)

```
v_mfma_f32_16x16x128_f8f6f4  dst[4], srcA[8], srcB[8], srcC[4]
```

- **A matrix**: M=16, K=128 (FP8 E4M3, 1 byte each) = 2048 bytes total
- **B matrix**: K=128, N=16 (FP8 E4M3) = 2048 bytes total
- **C accumulator**: M=16, N=16 (FP32) = 1024 bytes total
- **Per thread**: A = 8 x int32 (32 bytes), B = 8 x int32 (32 bytes), C = 4 x float32
- **Latency**: 32 cycles (when either operand is FP8)
- **HIP builtin**: `__builtin_amdgcn_mfma_f32_16x16x128_f8f6f4(a, b, c, cbsz, blgp)`

### 1.2 Scaled MFMA (MXFP4 / MXFP8)

```
v_mfma_scale_f32_16x16x128_f8f6f4  dst[4], srcA[8], srcB[8], srcC[4],
                                     cbsz, abid, blgp,
                                     opsel_a, scale_a, opsel_b, scale_b
```

- **cbsz**: A format (0=FP8_E4M3, 2=FP6_E3M2, 3=FP6_E2M3, 4=FP4_E2M1)
- **blgp**: B format (same encoding)
- **scale_a, scale_b**: E8M0 values (actual_scale = 2^(value - 127))
- Each scale covers 32 elements of the K dimension
- For K=128: 4 scales per row/column

**LLVM intrinsic** (from llvm-project PR #116723):
```c
// Returns v4f32, inputs are v8i32
float4 __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    int8 a,        // A matrix: 8 x int32 per thread
    int8 b,        // B matrix: 8 x int32 per thread
    float4 c,      // Accumulator: 4 x float32 per thread
    int cbsz,      // A format selector (4 = FP4_E2M1)
    int abid,      // A block ID (usually 0)
    int blgp,      // B format selector (4 = FP4_E2M1)
    int opsel_a,   // Scale selector for A (2-bit)
    int scale_a,   // E8M0 scale value for A
    int opsel_b,   // Scale selector for B (2-bit)
    int scale_b    // E8M0 scale value for B
);
```

### 1.3 Format Selection Table

| cbsz/blgp | Format     | Bits/elem | K elems per 8xi32 (32 bytes) | Scales needed (per 32 elems) |
|------------|------------|-----------|------------------------------|------------------------------|
| 0          | FP8 E4M3   | 8         | 128                          | 4                            |
| 4          | FP4 E2M1   | 4         | 256                          | 8                            |

**Important**: With FP4 (cbsz=4), the same 8xi32 registers hold 256 FP4 elements
(not 128). The hardware interprets the bit pattern differently. For our K=128
case with FP8 data, cbsz=0 / blgp=0.

---

## 2. Thread-to-Element Register Mapping (16x16x128 FP8)

### 2.1 Derivation from AITER Assembly

From the disassembly of `gfx950_a8w8_decode_ps.asm`, the MFMA instruction uses
**AGPR** (accumulator registers) for A and B operands:

```asm
; QK Phase - 5 MFMAs for 5*128=640 dims (576 used, 64 padding)
; A = Q data (loaded via ds_read_b128 into AGPRs a[40:75])
; B = K data (loaded via ds_read_b64_tr_b8 into AGPRs a[0:39] and a[120:183])

v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[40:47], a[0:7], 0      ; MFMA 0: K[0:127]
v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[48:55], a[8:15], v[38:41] ; MFMA 1: K[128:255]
v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[56:63], a[16:23], v[38:41]; MFMA 2: K[256:383]
v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[64:71], a[24:31], v[38:41]; MFMA 3: K[384:511]
v_mfma_f32_16x16x128_f8f6f4 v[38:41], a[72:79], a[32:39], v[38:41]; MFMA 4: K[512:639]
; Result in v[38:41] = QK scores for 16 heads x 16 tokens (first token group)
; Second token group uses a[80:119] and accumulates to v[42:45]
```

### 2.2 Thread Mapping Formulas

For wave64 (64 threads), `v_mfma_f32_16x16x128_f8f6f4`:

**Matrix A[M=16, K=128] (row-major FP8)**:
```
thread_id = lane_id  (0..63)
row = lane_id % 16   (which of the 16 M rows)
k_group = lane_id / 16  (0..3, each group covers 32 K elements)
// thread holds A[row, k_group*32 : k_group*32 + 32] packed in 8 x int32
// That's 32 FP8 bytes = 8 x 4-byte registers
```

**Matrix B[K=128, N=16] (col-major FP8)**:
```
thread_id = lane_id  (0..63)
col = lane_id % 16   (which of the 16 N columns)
k_group = lane_id / 16  (0..3, each group covers 32 K elements)
// thread holds B[k_group*32 : k_group*32 + 32, col] packed in 8 x int32
```

**Matrix C[M=16, N=16] (FP32 accumulator)**:
```
thread_id = lane_id  (0..63)
row = lane_id % 16
col_group = lane_id / 16  (0..3, each covers 4 N columns)
// thread holds C[row, col_group*4 : col_group*4 + 4] as 4 x float32
```

### 2.3 Register Budget Summary

| Operand | Regs/thread | Type   | Total across wave |
|---------|-------------|--------|-------------------|
| A       | 8 x int32   | VGPR/AGPR | 512 regs        |
| B       | 8 x int32   | VGPR/AGPR | 512 regs        |
| C       | 4 x float32 | VGPR      | 256 regs         |

---

## 3. Data Loading: ds_read_b64_tr_b8 (Transposed LDS Read)

### 3.1 Instruction Behavior

```asm
ds_read_b64_tr_b8 a[dst:dst+1], v[addr]  ; offset:imm
```

**HIP builtin**: `__builtin_amdgcn_ds_read_tr8_b64_v2i32(const __local int* ptr)`

This instruction reads 64 bits (8 bytes) from LDS per thread, but with a
**transposed access pattern** across the wave. Instead of each thread reading
consecutive bytes, threads cooperatively read a 2D tile and redistribute:

- 64 threads read from LDS with stride pattern
- The result is transposed so that data is arranged for MFMA B-operand layout
- Each thread gets 2 x int32 (8 bytes = 8 FP8 elements)
- Two ds_read_b64_tr_b8 calls fill one pair of AGPR registers

### 3.2 How AITER Uses It

From the assembly, K/V data is loaded from global memory into LDS via
`buffer_load_dwordx4 ... lds` (direct global-to-LDS), then read into AGPRs via
`ds_read_b64_tr_b8` with various offsets. This handles the B matrix (K tokens).

The pattern loads 16 KV tokens at a time, each with 576 bytes (FP8), into LDS.
The transposed reads extract the data in MFMA-compatible layout.

**Loading pattern for B operand (K data, 16 tokens x 128 dims per MFMA)**:
```
; For each group of 128 K-dims, load data for 16 tokens transposed:
; 64 threads x 2 AGPRs per read = 128 bytes per read
; Need 16 tokens x 128 bytes / (64 threads x 8 bytes/read) = 4 reads per MFMA
; But pattern shows 8 reads for B, suggesting 2 groups of 16 tokens

ds_read_b64_tr_b8 a[120:121], v10                ; KV tokens 0-15, K[0:7]
ds_read_b64_tr_b8 a[122:123], v11                ; KV tokens 0-15, K[8:15]
ds_read_b64_tr_b8 a[124:125], v10 offset:36864   ; KV tokens 0-15, K[16:23]
ds_read_b64_tr_b8 a[126:127], v11 offset:36864   ; KV tokens 0-15, K[24:31]
; ... 4 more reads for remaining K elements within this 128-dim tile
```

### 3.3 Q Data Loading (A operand)

Q data (16 heads x 576 dims = 9216 bytes) is loaded from LDS using regular
`ds_read_b128` (not transposed), because Q is the same for all KV tokens:

```asm
ds_read_b128 a[40:43], v21           ; Q[heads, K[0:15]]
ds_read_b128 a[44:47], v21 offset:1024  ; Q[heads, K[16:31]]
; ... 9 reads total covering 576 dims in groups of 128
; (5 MFMA tiles x 128 = 640, padding last 64)
```

Each `ds_read_b128` reads 16 bytes per thread. With 64 threads, that is 1024
bytes per instruction. 9 reads = 9216 bytes = 16 heads x 576 dims.

---

## 4. MLA Decode Kernel Design

### 4.1 Problem Dimensions

```
Q:  [batch, 16 heads, 576 dims]      -- FP32 (from host)
KV: [total_tokens, 576 dims]         -- FP8 E4M3 (quantized)
Output: [batch, 16 heads, 512 dims]  -- FP32

QK = Q[16, 576] x K[576, BLOCK_N]^T  -> scores[16, BLOCK_N]
P  = softmax(scores * sm_scale)       -> weights[16, BLOCK_N]
PV = P[16, BLOCK_N] x V[BLOCK_N, 512] -> output[16, 512]
```

### 4.2 MFMA Tiling Strategy (Following AITER Pattern)

**QK Phase**: Q[16, 576] x K[576, N]^T

Each `v_mfma_f32_16x16x128_f8f6f4` computes: A[16,128] x B[128,16] -> C[16,16]

- M=16 (heads) -- matches MFMA M dimension exactly
- K=576: need ceil(576/128) = 5 MFMA calls, accumulating
- N=16 KV tokens per MFMA column: BLOCK_N=16 per output tile

For BLOCK_N=32 (2 groups of 16 tokens):
- 2 x 5 = 10 MFMAs per BLOCK_N=32 tile
- Output: 2 x v[4] = 8 float32 accumulators per thread

**PV Phase**: P[16, BLOCK_N] x V[BLOCK_N, 512]

After softmax, P values are FP32 -> convert to FP8 for MFMA input.
AITER does exactly this (line 644-647 in ASM analysis):
```asm
v_cvt_pk_fp8_f32 v38, v38, v39          ; Convert 2 FP32 scores to FP8
v_cvt_pk_fp8_f32 v38, v40, v41 op_sel   ; Pack 4 more
v_permlane16_swap_b32_e32 v38, v39       ; Redistribute across lanes
```

Then uses 8 MFMAs per V-dimension group (128 V dims at a time):
```
P[16, 128] x V[128, 16] -> acc[16, 16]
```
- 512/16 = 32 output V-dim tiles
- But P only has BLOCK_N columns per iteration, not 128
- Need to accumulate across multiple KV blocks
- When BLOCK_N=16: pad P to 128 (fill with 0), or loop over KV blocks of 128

**AITER's actual approach** (from ASM): Process 16 tokens at a time.
For PV, accumulate across all 16-token blocks. Each block contributes:
- P[16, 16] padded to P[16, 128] (mostly zeros) -- wasteful
- OR: use a smaller MFMA? No, 16x16x128 is the only f8f6f4 16-wide MFMA.
- OR: pack multiple blocks of 16 tokens into the 128 K dimension.

**Key insight from AITER**: AITER processes **128 KV tokens in the K-dimension
of PV MFMA** by batching 8 blocks of 16 tokens together. This is why the V
accumulation uses `a[120:183]` (64 AGPRs = 8 groups of 8 AGPRs for V tiles):

```asm
; PV Phase: 8 MFMAs for 8 x 16 = 128 V dims at a time
v_mfma_f32_16x16x128_f8f6f4 v[74:77],  a[120:127], v[38:45], v[74:77]
v_mfma_f32_16x16x128_f8f6f4 v[78:81],  a[128:135], v[38:45], v[78:81]
v_mfma_f32_16x16x128_f8f6f4 v[82:85],  a[136:143], v[38:45], v[82:85]
v_mfma_f32_16x16x128_f8f6f4 v[86:89],  a[144:151], v[38:45], v[86:89]
v_mfma_f32_16x16x128_f8f6f4 v[90:93],  a[152:159], v[38:45], v[90:93]
v_mfma_f32_16x16x128_f8f6f4 v[94:97],  a[160:167], v[38:45], v[94:97]
v_mfma_f32_16x16x128_f8f6f4 v[98:101], a[168:175], v[38:45], v[98:101]
v_mfma_f32_16x16x128_f8f6f4 v[102:105],a[176:183], v[38:45], v[102:105]
```

Here, `v[38:45]` = 8 floats = softmax weights for 16 tokens (packed as FP8),
and `a[120:183]` = V data for those tokens across 8 x 16 = 128 V dims.
This gives 8 output tiles of [16, 16] = 128 V dims per pass.

512 V dims / 128 per pass = 4 passes over V dims.
But wait -- 8 MFMAs x 16 N = 128 V dims. And 512/128 = 4 passes.
That means 32 total MFMAs for PV per 16 KV tokens.

### 4.3 Full Pipeline

```
For each split of KV tokens:
  For each BLOCK_N=16 KV tokens:
    1. Load Q[16, 576] into LDS (once per split, reuse)
    2. Load K[16_tokens, 576] into LDS via buffer_load ... lds
    3. QK Phase: 5 MFMAs -> scores[16 heads, 16 tokens]
    4. Online softmax: max, exp, normalize
    5. Convert P scores to FP8 (v_cvt_pk_fp8_f32)
    6. Load V[16_tokens, 512] into LDS  (or reuse from KV load)
    7. PV Phase: 32 MFMAs -> partial_output[16, 512]
    8. Accumulate with running output using online softmax rescaling
```

### 4.4 Wavefront Organization

Following AITER's pattern (1 threadgroup, 4 waves = 256 threads):

- **Wave 0-1**: Compute waves (execute MFMAs)
- **Wave 2-3**: Memory waves (load data from global to LDS)
- Ping-pong between compute and memory using barriers

Actually, AITER uses a more nuanced scheme per the gfx950 blog: within a
single wave, odd and even cycles alternate between MFMA execution and memory
operations. The 16-cycle MFMA latency allows interleaving loads.

---

## 5. Implementation Approach

### 5.1 Phase 4a: FP8 MFMA via HIP C++ Builtins (Recommended First Step)

Use the existing FP8 KV cache data (no MXFP4 conversion needed) with the
unscaled MFMA instruction. This is the simplest path:

```cpp
#include <hip/hip_runtime.h>

// Type aliases matching LLVM intrinsics
using int8_vec = int __attribute__((ext_vector_type(8)));
using float4_vec = float __attribute__((ext_vector_type(4)));

// The builtin (available in ROCm 6.x+ with gfx950 target)
extern "C" float4_vec __builtin_amdgcn_mfma_f32_16x16x128_f8f6f4(
    int8_vec a, int8_vec b, float4_vec c,
    int cbsz, int blgp
);

__global__ void mla_decode_mfma(
    const float* __restrict__ Q,        // [batch, 16, 576] FP32
    const uint8_t* __restrict__ KV,     // [total_tokens, 576] FP8
    float* __restrict__ output,         // [batch, 16, 512] FP32
    const int* __restrict__ kv_indptr,
    float sm_scale, float kv_scale
) {
    // Shared memory for Q (FP8) and KV tiles
    __shared__ uint8_t smem_q[16 * 576];     // 9216 bytes
    __shared__ uint8_t smem_kv[16 * 576];    // 9216 bytes per tile
    __shared__ float smem_scores[16 * 16];   // softmax workspace

    int lane = threadIdx.x % 64;
    int warp = threadIdx.x / 64;

    // Step 1: Convert Q from FP32 to FP8 and store in LDS
    // (Q is FP32 input, MFMA needs FP8)
    // Use v_cvt_pk_fp8_f32 or software conversion
    load_q_as_fp8(Q, smem_q, ...);
    __syncthreads();

    // Step 2: For each tile of 16 KV tokens
    for (int t = 0; t < num_tokens; t += 16) {
        // Load KV[16, 576] into LDS
        load_kv_tile(KV, smem_kv, t, ...);
        __syncthreads();

        // Step 3: QK phase - 5 MFMAs
        float4_vec qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < 5; k++) {
            int8_vec a_q = load_a_from_lds(smem_q, k * 128);
            int8_vec b_k = load_b_from_lds_transposed(smem_kv, k * 128);
            qk_acc = __builtin_amdgcn_mfma_f32_16x16x128_f8f6f4(
                a_q, b_k, qk_acc, 0, 0);  // cbsz=0 (FP8), blgp=0 (FP8)
        }

        // Step 4: Online softmax on qk_acc
        // Each thread has 4 scores for (lane%16) head, 4 consecutive tokens
        online_softmax(qk_acc, &running_max, &running_sum, ...);

        // Step 5: Convert softmax weights to FP8
        // Pack into registers for PV MFMA

        // Step 6: PV phase - 32 MFMAs (4 passes of 8 MFMAs)
        for (int vg = 0; vg < 4; vg++) {
            for (int vm = 0; vm < 8; vm++) {
                int8_vec a_p = softmax_weights_fp8;
                int8_vec b_v = load_v_from_lds(smem_kv, vg*128 + vm*16);
                pv_acc[vg*8+vm] = __builtin_amdgcn_mfma_f32_16x16x128_f8f6f4(
                    a_p, b_v, pv_acc[vg*8+vm], 0, 0);
            }
        }
    }
}
```

### 5.2 Phase 4b: MXFP4 Scaled MFMA (Stretch Goal)

If KV cache is stored in MXFP4 format (as in our Triton kernels), use the
scaled variant:

```cpp
extern "C" float4_vec __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    int8_vec a, int8_vec b, float4_vec c,
    int cbsz, int abid, int blgp,
    int opsel_a, int scale_a,
    int opsel_b, int scale_b
);

// For MXFP4 KV data:
// cbsz=4 (FP4 for A/Q), blgp=4 (FP4 for B/K)
// Each scale covers 32 elements
// K=128 needs 4 scales per MFMA call
// scale_a and scale_b are E8M0 values loaded from scale tensors
```

### 5.3 Inline Assembly Fallback

If the compiler builtins are not available in our ROCm version, use inline asm:

```cpp
// Unscaled MFMA
__device__ __forceinline__ float4_vec mfma_f32_16x16x128_f8f6f4(
    int8_vec a, int8_vec b, float4_vec c
) {
    float4_vec result;
    asm volatile(
        "v_mfma_f32_16x16x128_f8f6f4 %0, %1, %2, %3"
        : "=v"(result)
        : "a"(a), "a"(b), "v"(c)  // Note: a,b in AGPRs, c in VGPRs
    );
    return result;
}

// Transposed LDS read
__device__ __forceinline__ int2 ds_read_tr8_b64(const __local void* ptr) {
    int2 result;
    asm volatile(
        "ds_read_b64_tr_b8 %0, %1"
        : "=a"(result)
        : "v"(ptr)
    );
    return result;
}
```

---

## 6. Critical Data Layout Details

### 6.1 LDS Layout for K Data (B operand)

To use `ds_read_b64_tr_b8` for loading K data, the LDS must be laid out so
the transposed read produces MFMA-compatible registers.

**LDS layout for 16 KV tokens x 128 K-dims (one MFMA tile)**:
```
LDS[token][k_dim] -- row-major, 16 rows x 128 cols = 2048 bytes
```

The `ds_read_b64_tr_b8` reads from 64 lanes simultaneously with transposition:
- Lane `i` reads 8 bytes starting at `base + (i%16)*128 + (i/16)*8`
  (this is speculative -- exact formula depends on ISA doc)
- The transposition ensures each lane gets the correct K elements for its
  MFMA B-operand slot

**From the AITER ASM**, the K data is loaded with offset patterns:
- Base offsets: 0, 36864 (= 0x9000 = 36 KB offset for second LDS bank)
- Step offsets: 16, 1024, 1040
- This suggests a complex LDS layout optimized for bank-conflict avoidance

### 6.2 LDS Layout for Q Data (A operand)

Q data uses regular `ds_read_b128` (not transposed). From AITER:
```
ds_read_b128 a[40:43], v21              ; 16 bytes, offset 0
ds_read_b128 a[44:47], v21 offset:1024  ; +1024
ds_read_b128 a[48:51], v21 offset:2048  ; +2048
...
ds_read_b128 a[72:75], v21 offset:8192  ; +8192
```

9 reads x 16 bytes/read x 64 threads = 9216 bytes = Q[16 heads, 576 dims] in
FP8 (after conversion from FP32).

The 1024-byte stride suggests Q is laid out as:
```
LDS_Q[head_group][k_chunk] with stride 1024 between k_chunks
```

### 6.3 Softmax Score Distribution

After QK MFMAs, scores are in v[38:41] (4 floats per thread):
```
thread i holds: scores[i%16, (i/16)*4 : (i/16)*4+4]
= scores[head=i%16, tokens=(i/16)*4..(i/16)*4+3]
```

For online softmax:
- Need max across all 16 tokens for each head
- Each head's scores are spread across 4 threads (i/16 = 0..3)
- Need cross-lane reduction: `__shfl_xor` or `ds_permute`
- AITER uses `ds_write_b32` + `ds_read_b32` + `v_max3_f32` for this

### 6.4 Softmax-to-FP8 Conversion for PV Phase

After softmax, AITER converts weights to FP8 for use as MFMA A operand:
```asm
v_cvt_pk_fp8_f32 v38, v38, v39          ; 2 FP32 -> 2 FP8 (packed in low 16 bits)
v_cvt_pk_fp8_f32 v38, v40, v41 op_sel   ; 2 more FP8 in high 16 bits
v_cvt_pk_fp8_f32 v39, v42, v43          ; Next register
v_cvt_pk_fp8_f32 v39, v44, v45 op_sel
v_permlane16_swap_b32_e32 v38, v39       ; Redistribute for MFMA layout
```

Then `v[38:45]` is used as the A operand (softmax weights) in PV MFMAs.

---

## 7. Performance Estimation

### 7.1 MFMA Throughput

Per CU on MI355X (gfx950):
- 4 MFMA units (one per SIMD)
- `v_mfma_f32_16x16x128_f8f6f4`: 32 cycles, 16x16x128 = 32768 FP8 MACs
- Peak: 32768 / 32 = 1024 MACs/cycle per MFMA unit
- 4 units x 1024 = 4096 MACs/cycle per CU

### 7.2 Per-Token Computation

For one batch item, 16 heads, processing 16 KV tokens:
- **QK**: 5 MFMAs x 2 token groups = 10 MFMAs -> 10 x 32 = 320 cycles
- **Softmax**: ~50 cycles (exp, reduction, conversion)
- **PV**: 32 MFMAs x 1 pass = 32 MFMAs -> 32 x 32 = 1024 cycles
- **Total compute**: ~1400 cycles for 16 tokens

For kv_len=8192: 8192/16 = 512 iterations -> 512 x 1400 = 716,800 cycles
At ~1.7 GHz: ~420 us

But with proper pipelining (overlap memory and compute):
- Memory latency hidden behind MFMA execution
- Estimated 2-3x speedup from pipelining -> ~150-210 us

### 7.3 Comparison

| Config | Phase 3.5 HIP | AITER | Phase 4 (estimated) |
|--------|---------------|-------|---------------------|
| bs=4, kv=1024 | 0.104ms | 0.385ms | ~0.08ms |
| bs=4, kv=8192 | slow | 0.5ms | ~0.2ms |
| bs=32, kv=8192 | very slow | 2.5ms | ~1.5ms |
| bs=64, kv=8192 | very slow | 4.5ms | ~3.0ms |

The improvement comes from:
1. No scalar dequant loop (128 FP8 ops replaced by 1 MFMA)
2. Hardware transposed LDS reads (no manual transpose)
3. Proper use of AGPR file (doubles effective register space)

---

## 8. Implementation Plan

### Phase 4a: Minimal MFMA Kernel (1-2 days)

**Goal**: Get a single MFMA instruction working in HIP C++ via load_inline.

1. Write a simple test kernel that:
   - Loads two 16x128 FP8 matrices from global memory
   - Calls `__builtin_amdgcn_mfma_f32_16x16x128_f8f6f4`
   - Writes 16x16 FP32 result to global memory
2. Verify correctness against CPU reference
3. If builtin fails, try inline assembly approach

**Risk**: Compiler may not support the builtin via torch's load_inline.
**Mitigation**: Use `__asm__` volatile inline assembly.

### Phase 4b: QK MFMA Integration (1-2 days)

**Goal**: Replace scalar QK dot product with 5 MFMAs.

1. Convert Q from FP32 to FP8 in shared memory
2. Load KV tile into LDS
3. Execute 5 MFMAs for QK scores
4. Extract and validate scores match Phase 3.5

**Risk**: Getting the LDS layout right for `ds_read_b64_tr_b8`.
**Mitigation**: Start with regular `ds_read_b128` (non-transposed),
manually arrange data. Optimize with tr reads later.

### Phase 4c: Softmax + PV MFMA (1-2 days)

**Goal**: Full end-to-end MFMA kernel.

1. Online softmax on MFMA output scores
2. Convert softmax weights to FP8
3. 32 MFMAs for PV accumulation
4. Output writeback

**Risk**: Softmax score distribution across lanes is tricky.
**Mitigation**: Use LDS for cross-lane reduction (proven in AITER).

### Phase 4d: Memory Optimization (1-2 days)

**Goal**: Match or approach AITER performance.

1. Double-buffered LDS loads
2. Ping-pong between compute and memory waves
3. `ds_read_b64_tr_b8` for transposed K loads
4. Direct global-to-LDS buffer loads
5. AGPR usage for A/B operands

**Risk**: Inline ASM complexity, hard to debug.
**Mitigation**: Incremental optimization, benchmark each step.

---

## 9. Risk Assessment

### High Risk
- **Register layout correctness**: The thread-to-element mapping for 16x16x128
  is not well-documented. Must verify empirically with test kernels.
- **Compiler support**: `__builtin_amdgcn_mfma_f32_16x16x128_f8f6f4` may
  require specific ROCm/compiler version. The MI355X evaluation nodes may
  have an older toolchain.
- **LDS transposed reads**: `ds_read_b64_tr_b8` behavior is only partially
  documented. May need to reverse-engineer from AITER disassembly.

### Medium Risk
- **Softmax between phases**: Converting FP32 softmax weights back to FP8
  and redistributing across lanes adds complexity.
- **Numerical accuracy**: FP8 MFMA accumulation across 5 tiles may lose
  precision vs FP32 scalar computation.
- **AGPR vs VGPR**: MFMA A/B operands may need to be in AGPRs. HIP C++
  builtins may not support AGPR allocation directly -- may need inline asm.

### Low Risk
- **Split-K reduction**: Can reuse existing Phase 3.5 reduction kernel.
- **Memory bandwidth**: KV data is already FP8, no additional quantization.
- **Integration**: Can slot into existing routing framework.

---

## 10. Key Takeaways from AITER ASM Analysis

1. **AITER uses `v_mfma_f32_16x16x128_f8f6f4` (unscaled FP8)**, NOT the
   `_scale` variant. The KV cache is plain FP8 E4M3, not MXFP4.

2. **AITER processes 16 tokens at a time** with BLOCK_N=16 matching MFMA N dim.

3. **QK uses 5 MFMAs** (576 dims / 128 = 4.5, rounded to 5 with padding).

4. **PV uses 8 MFMAs per V-dim group** (128 V dims), 4 groups for 512 V dims
   = 32 MFMAs total. This processes 16 KV tokens' worth of V data per MFMA
   set (K dimension of MFMA = 128, but actual token count = 16; the remaining
   112 slots in the K dimension are zero-padded or from subsequent tokens).

5. **Online softmax is done between QK and PV phases** with FP32 arithmetic,
   then weights are converted to FP8 via `v_cvt_pk_fp8_f32` and redistributed
   via `v_permlane16_swap_b32`.

6. **Data is loaded into AGPRs** (accumulator registers, a[]) not VGPRs (v[]).
   The MFMA can source from either, but AGPRs effectively double register
   capacity since they don't compete with VGPR allocation.

7. **Buffer load direct-to-LDS** (`buffer_load_dwordx4 ... lds`) is used
   extensively, bypassing VGPRs entirely for KV data movement.

---

## 11. Recommended Approach: Start with Unscaled FP8

Given the analysis, Phase 4 should use **unscaled FP8 MFMA** (matching AITER's
approach) rather than MXFP4 scaled MFMA, because:

1. KV cache is already in FP8 format -- no conversion needed
2. Unscaled instruction is simpler (fewer parameters, no scale management)
3. AITER proves this approach achieves high performance on MI355X
4. Can always add MXFP4 scaled variant later as Phase 4b

The primary goal is to replace the **scalar FP8 dequant loop** with a **single
MFMA instruction per 128 K-elements**, which is a 128x reduction in instruction
count for the inner loop.
