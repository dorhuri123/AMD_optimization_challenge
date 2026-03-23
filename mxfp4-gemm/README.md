# Challenge 3: MXFP4 GEMM

**Owner:** Person B
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)

---

## What Is MXFP4?

MXFP4 (Microscaling FP4) is a block quantization format where:
- Each value uses **4 bits**
- Every **32 values** share one FP8 scaling factor (the "microscale")
- The MI355X has **native hardware support** for MXFP4 matrix cores

This enables up to **4x higher peak throughput vs FP16** on MI355X — but only if the kernel correctly feeds the matrix cores fast enough to keep them saturated.

GEMM accounts for ~62% of end-to-end compute in large transformer models (e.g., Llama2-70B), making this challenge the highest compute-impact optimization.

## MXFP4 Format Details

```
Block layout (32 elements):
  [e0, e1, ..., e31]  — each 4-bit FP4 value
  [scale]             — one FP8 exponent shared across the block

FP4 value range:   ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}  (e2m1 format)
Scale (FP8 e8m0):  power-of-2 shared exponent

Actual value = e_i * 2^scale
```

## Files

| File | Purpose |
|------|---------|
| `reference.py` | Baseline from gpu-mode/reference-kernels — do not modify |
| `solution.py` | Our optimized GEMM kernel — edit this |
| `benchmark.py` | Performance comparison across matrix shapes |

## Getting the Official Baseline

```bash
cp $REF_KERNELS_DIR/AMD/mxfp4-gemm/reference.py ./reference.py
cp $REF_KERNELS_DIR/AMD/mxfp4-gemm/task.py ./task.py
cp $REF_KERNELS_DIR/AMD/mxfp4-gemm/task.yml ./task.yml
```

## Running

```bash
cd mxfp4-gemm
python reference.py
python solution.py
python benchmark.py
```

## Submission

```bash
popcorn submit solution.py --problem mxfp4-gemm
```

## Optimization Spaces (in priority order)

### 1. Native Matrix Core Usage
MI355X matrix cores natively process MXFP4 — the key is ensuring we are actually using them.
- Use `rocprof` to verify matrix core utilization (look for `SQ_INSTS_VALU_MFMA_*` counters)
- If utilization is low, the kernel is not feeding data fast enough → add double-buffering

### 2. Double-Buffering (Async Prefetch)
Overlap HBM tile loading with matrix core compute:
```python
# Triton: use tl.async_copy + pipeline stages
# Target: while computing tile K, prefetch tile K+1 from HBM
```

### 3. Tile Size Tuning for MI355X
MI355X has different matrix core tile dimensions vs MI300X. Profile and tune:
```python
# Key Triton autotuning parameters:
BLOCK_M = [16, 32, 64, 128]
BLOCK_N = [16, 32, 64, 128]
BLOCK_K = [32, 64, 128]        # must be multiple of MXFP4 block size (32)
num_stages = [2, 3, 4]         # pipeline depth
num_warps  = [4, 8]
```

### 4. Scale Factor Prefetch
The FP8 scale factors (one per 32 elements) must be loaded alongside the data.
- Pack scale loads with data loads to avoid separate HBM accesses
- Keep scales in shared memory across the K-loop

### 5. Stream-K Decomposition
Standard data-parallel GEMM has load-imbalance for non-square shapes.
Stream-K assigns work in K-dimension strips for better warp utilization:
- Particularly important for tall-skinny (large M, small N) and wide (small M, large N) matrices
- Reference: [Stream-K paper](https://arxiv.org/abs/2301.03598)

### 6. Persistent Kernels
For many small GEMMs (MoE context), kernel launch overhead adds up.
A persistent kernel stays alive and processes multiple tiles in sequence:
```python
# One kernel launch handles all tiles via a work queue
```

### 7. Mixed Precision (MXFP4 weights + MXFP6 activations)
AMD research shows MXFP4 weights + MXFP6 activations maintain accuracy better than pure MXFP4.
Check if the task spec allows this variant.

## Key Performance Targets

| Shape (M×N×K) | Expected TFLOPS | Notes |
|---|---|---|
| 4096×4096×4096 | ~1200+ TFLOPS | Large square, compute-bound |
| 128×4096×4096 | ~300+ TFLOPS | Small M, memory-bound |
| 4096×128×4096 | ~300+ TFLOPS | Small N, memory-bound |

MI355X peak MXFP4 compute: ~2457 TFLOPS (matrix cores)

## Useful Starting Point: CK Templates

Composable Kernel (CK) has pre-tuned MXFP4 GEMM templates. Study these before writing from scratch:

```bash
git clone https://github.com/ROCm/composable_kernel
# Look in: composable_kernel/include/ck_tile/ops/gemm/
# And:     composable_kernel/example/ck_tile/14_mxfp4_gemm/
```

## Profiling

```bash
# Check matrix core utilization
rocprof --stats --counter SQ_INSTS_VALU_MFMA_I32_16X16X32_FP8 python solution.py

# Full HW analysis
omniperf profile --name gemm_run -- python solution.py
omniperf analyze -p workloads/gemm_run/ --dispatch 0

# Roofline: are we compute-bound or memory-bound?
# If TFLOPS << peak AND memory BW utilization is high → memory-bound → add prefetching
# If TFLOPS << peak AND memory BW low → compute kernel not using matrix cores properly
```

## References

- [MXFP4 quantization on AMD MI355X](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [CK MXFP4 GEMM](https://github.com/ROCm/composable_kernel)
- [Triton for ROCm](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-guide/triton.html)
- [MI355X architecture specs](https://www.amd.com/en/products/accelerators/instinct/mi300/mi355x.html)
