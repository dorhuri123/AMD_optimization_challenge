# mxfp4-mm: MXFP4 Matrix Multiply (A4W4)

**Owner:** Person B
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)
**Official spec:** [amd_202602/mxfp4-mm](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/mxfp4-mm)
*(No separate README in the official repo — spec is in `task.py` and `task.yml`)*

---

## What This Challenge Is

Implement a fast **A4W4 GEMM**: given bf16 activations A and pre-quantized MXFP4 weights B, quantize A online to MXFP4 and run the GEMM on MI355X.

This is the building block for the MoE challenge — the same kernel drives both Stage 1 and Stage 2 of every expert.

## Task Interface

```python
# Input tuple: (A, B, B_q, B_shuffle, B_scale_sh)
A          # [m, k]          bfloat16   — activations (to be quantized online)
B          # [n, k]          bfloat16   — original weights (for reference/dequant check only)
B_q        # [n, k//2]       fp4x2      — B quantized to MXFP4, raw layout
B_shuffle  # [n, k//2]       fp4x2      — B_q after shuffle_weight((16,16)) for CK
B_scale_sh # [...]           e8m0       — block scales, e8m0_shuffled (from quant(B, shuffle=True))

# Output: C [m, n] bfloat16
```

What your `custom_kernel` must do:

1. Quantize A to MXFP4 (per_1x32 dynamic block quantization)
2. Run `gemm_a4w4(A_q, B_shuffle, xscale=A_scale, wscale=B_scale_sh, bpreshuffle=True)`

Constraints:

- M, N divisible by 64
- K divisible by 64 (scale group 32 × fp4 pack 2)
- Output dtype: bfloat16

## MXFP4 Format

| Property | Value |
| --- | --- |
| FP4 format | E2M1 — values `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, max=6.0 |
| Scale format | E8M0 — exponent-only (power-of-2) |
| Block size | 32 values per scale |
| Packing | 2 FP4 values per byte (fp4x2): low nibble=even index, high nibble=odd index |
| Weight shuffle | `shuffle_weight(B_q, layout=(16,16))` — tile-coalesced for CK GEMM instructions |
| Scale shuffle | `fp4_utils.e8m0_shuffle(scale)` — reorders scales to match shuffled weight layout |

## Files

| File | Purpose |
| --- | --- |
| `reference.py` | AITER `gemm_a4w4` baseline — do not modify |
| `task.py` | `input_t`, `output_t`, `TestSpec` definitions |
| `task.yml` | 6 benchmark configs (leaderboard uses geometric mean) |
| `submission.py` | **Edit this** — `custom_kernel(data)` implementation |
| `benchmark.py` | reference vs submission latency + TFLOPS table |

## Running

```bash
cd mxfp4-mm
python reference.py    # verify baseline
python benchmark.py    # reference vs your submission
popcorn submit submission.py --problem mxfp4-mm
```

## Accuracy Target

Validated against `run_torch_fp4_mm` (dequant + bf16 matmul) with `rtol=1e-02, atol=1e-02`.

## Benchmark Configs (from task.yml)

Shapes reflect actual DeepSeek-R1 weight dimensions (tokens × hidden):

| m | n | k | Notes |
| --- | --- | --- | --- |
| 4 | 4096 | 7168 | decode, single token |
| 16 | 4096 | 7168 | small decode batch |
| 64 | 4096 | 7168 | |
| 128 | 4096 | 7168 | |
| 256 | 4096 | 7168 | |
| 256 | 7168 | 4096 | down-proj shape |

Small M is the hard case — matrix cores are underutilized when M is tiny, and memory bandwidth dominates.

---

## Optimization Opportunities

The reference is `aiter.gemm_a4w4(bpreshuffle=True)`. The bottleneck moves depending on M:

- **Small M (4–64):** memory-bandwidth bound — A is tiny, B weights dominate HBM reads
- **Large M (256+):** compute-bound — feed matrix cores fast enough

### 1. Fuse A Quantization into the GEMM Prologue (highest impact)

The reference quantizes A in a **separate pass** (write fp4x2 + scale to HBM, then read back in GEMM). Fusing saves one full activation write + read:

```text
Current:  A (bf16) → [quant kernel] → A_q (fp4x2) [HBM] → [gemm_a4w4]
Fused:    A (bf16) → [quant in GEMM prologue, stays in registers] → [gemm_a4w4]
```

This is the same pattern as the MoE activation fusion and the core insight of vLLM PR #36297.

### 2. Tile Size Tuning for MI355X

MI355X matrix cores have different optimal tile dimensions vs MI300X. The CK kernel has fixed tile sizes — a Triton kernel lets you autotune:

```python
# Key parameters (BLOCK_K must be multiple of 32 for MXFP4 block alignment):
BLOCK_M = [16, 32, 64, 128]
BLOCK_N = [64, 128, 256]
BLOCK_K = [32, 64, 128]
num_stages = [2, 3, 4]    # pipeline depth for double/triple buffering
num_warps  = [4, 8]
```

### 3. Double-Buffering (Async Prefetch)

For larger M, overlap HBM B-weight tile loading with matrix core compute on the previous tile:

```python
# Triton: num_stages=3 or 4 enables the compiler to emit async copy + barrier
# Target: while computing tile K_i, prefetch B tile K_{i+1} from HBM
```

Particularly effective for the large-M configs (m=256) where compute time is long enough to hide the prefetch latency.

### 4. Scale Factor Load Coalescing

Each E8M0 scale covers 32 elements. Scale loads must be coalesced with the corresponding fp4x2 data loads to avoid a second HBM round-trip. Ensure:

- Scale tensor is accessed with the same stride pattern as the weight tensor
- Scale values are kept in shared memory across the K-loop iterations within a tile

### 5. Small-M Kernel Path

For m=4–16, matrix cores are often idle waiting for data. A separate small-M kernel path can:

- Use a wider BLOCK_N to amortize B-weight loads across more output columns
- Reduce grid size to match the actual number of active CUs
- Skip async copy (prefetch doesn't help when compute finishes before the next tile arrives)

### 6. Stream-K Decomposition

For non-square shapes (m=4, n=4096, k=7168), standard data-parallel GEMM wastes CUs — most blocks finish early while a few large-K blocks are still running. Stream-K distributes work in K-strips for uniform CU load.

---

## Profiling

```bash
# Matrix core utilization — are we actually using FP4 instructions?
rocprof --stats --counter SQ_INSTS_VALU_MFMA_F32_16X16X32_FP8 python benchmark.py

# Full roofline analysis
omniperf profile --name mm -- python benchmark.py
omniperf analyze -p workloads/mm/
# Check: VALU utilization vs theoretical peak, HBM BW utilization

# Quick roofline sanity
# MI355X: ~2457 TFLOPS FP4, ~6.5 TB/s HBM3e
# For m=256, n=4096, k=7168: arithmetic intensity = 2*256*4096*7168 / (256*7168 + 4096*7168) bytes
# ~ 2*256*4096*7168 / (7168*(256+4096)) bytes = ~476 FLOP/byte → compute-bound
# For m=4: ~7.4 FLOP/byte → memory-bound
```

## References

- [MXFP4 quantization on AMD MI355X](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [aiter gemm_a4w4 tests](https://github.com/ROCm/aiter/blob/main/op_tests/test_gemm_a4w4.py)
- [CK MXFP4 GEMM templates](https://github.com/ROCm/composable_kernel)
- [Triton for ROCm](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-guide/triton.html)
