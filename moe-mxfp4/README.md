# moe-mxfp4: MXFP4 Mixture-of-Experts Fused Kernel

**Owner:** Person B
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)
**Official spec:** [amd_202602/moe-mxfp4](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/moe-mxfp4)

---

## What This Challenge Is

Implement a DeepSeek-R1 style MXFP4 MoE fused kernel that beats AITER's `fused_moe` reference on MI355X.

The kernel fuses a **2-stage pipeline** across all tokens and experts:

1. **Stage 1:** MXFP4 GEMM (gate + up projection) + SwiGLU activation
2. **Stage 2:** MXFP4 GEMM (down projection) + weighted reduction across top-k experts

## DeepSeek-R1 MoE Architecture

| Parameter | Value | Notes |
| --- | --- | --- |
| hidden_size | 7168 | Model hidden dimension |
| moe_intermediate_size | 2048 | Per-expert intermediate dim (full, EP-off splits to 256) |
| n_routed_experts | 256 | Routed experts per GPU (EP-off) or 32 (EP=8) |
| n_shared_experts | 1 | Always selected, weight = 1.0 |
| top_k (routed) | 8 | Routed experts per token |
| total_top_k | **9** | 8 routed + 1 shared |
| E_total | **257** | 256 + 1 shared (EP-off) or 33 (EP=8) |

## Kernel Flow

```
For each token i and each assigned expert j:

(1) Quant: hidden_states → MXFP4  (aiter per-1x32 dynamic, block_size=32)

(2) Stage 1 GEMM + SwiGLU:
    gate = x_i @ W_gate_j.T          [d_hidden → d_expert]
    up   = x_i @ W_up_j.T            [d_hidden → d_expert]
    h    = SiLU(gate) * up            ← SwiGLU activation
    (W_gate and W_up fused as one a4w4 GEMM)

(3) Stage 2 GEMM:
    expert_out = h @ W_down_j.T       [d_expert → d_hidden]

(4) Weighted reduction:
    output_i += w_ij * expert_out     ← accumulate across top_9 experts
```

All GEMMs are **a4w4** (MXFP4 activations × MXFP4 weights, per-1x32 block scaling).

## Input Tuple (12 elements)

```python
(
  hidden_states,                  # [M, d_hidden]                            bf16
  gate_up_weight,                 # [E, 2*d_expert_pad, d_hidden_pad//2]     fp4x2  (raw)
  down_weight,                    # [E, d_hidden_pad, d_expert_pad//2]       fp4x2  (raw)
  gate_up_weight_scale,           # [E, 2*d_expert_pad, d_hidden_pad//32]    e8m0   (raw)
  down_weight_scale,              # [E, d_hidden_pad, d_expert_pad//32]      e8m0   (raw)
  gate_up_weight_shuffled,        # [E, 2*d_expert_pad, d_hidden_pad//2]     fp4x2  (pre-shuffled 16x16)
  down_weight_shuffled,           # [E, d_hidden_pad, d_expert_pad//2]       fp4x2  (pre-shuffled 16x16)
  gate_up_weight_scale_shuffled,  # [padded, flat]                           e8m0   (shuffled)
  down_weight_scale_shuffled,     # [padded, flat]                           e8m0   (shuffled)
  topk_weights,                   # [M, 9]                                   float32
  topk_ids,                       # [M, 9]  cols 0-7: routed, col 8: shared  int32
  config,                         # dict: d_hidden, d_expert, pads, expert counts
)
```

## MXFP4 Format

| Property | Value |
| --- | --- |
| FP4 format | E2M1 — values `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, max=6.0 |
| Scale format | E8M0 — exponent-only (power-of-2) |
| Block size | 32 elements per scale |
| Packing | 2 FP4 values per byte (fp4x2): low nibble=even, high nibble=odd |
| Padding | Dims padded to 256-alignment for CK kernel |

## Files

| File | Purpose |
| --- | --- |
| `reference.py` | AITER `fused_moe` baseline — do not modify |
| `task.py` | `input_t`, `output_t`, `TestSpec` definitions |
| `task.yml` | 6 benchmark configs (leaderboard uses geometric mean) |
| `submission.py` | **Edit this** — `custom_kernel(data)` implementation |
| `benchmark.py` | reference vs submission latency table |

## Running

```bash
cd moe-mxfp4
python reference.py    # verify baseline
python benchmark.py    # reference vs your submission
popcorn submit submission.py --problem moe-mxfp4
```

## Accuracy Target

Validated against AITER reference with `rtol=1e-2, atol=1e-2`.

## Reference Performance (AITER `fused_moe` on MI355X)

| bs | E | d_expert | top_k | time (μs) |
| --- | --- | --- | --- | --- |
| 4 | 257 | 256 | 9 | 46.9 |
| 64 | 257 | 256 | 9 | 187.7 |
| 256 | 257 | 256 | 9 | 245.7 |
| 64 | 33 | 2048 | 9 | 220.6 |
| 256 | 33 | 2048 | 9 | 276.4 |
| 1024 | 33 | 2048 | 9 | 572.2 |

Ranking = geometric mean latency across all 6 configs.

---

## Optimization Opportunities

The following are taken directly from the [official README](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/moe-mxfp4) plus our own analysis. The AITER CK `fused_moe` kernel is already well-optimized — beating it requires at least one of these.

### 1. Activation Quantization Fusion (highest impact)

The reference quantizes activations to MXFP4 in a **separate kernel** before Stage 1 GEMM. This writes quantized activations to HBM, then reads them back for the GEMM.

Fusing dynamic MXFP4 quantization into the Stage 1 GEMM prologue saves one full activation-tensor write + read:

```text
Current:  hidden (bf16) → [quant kernel] → activations (fp4x2) [HBM] → [GEMM]
Fused:    hidden (bf16) → [quant in GEMM prologue, stays in registers] → [GEMM]
```

### 2. Inter-Stage Fusion (Stage 1 + Stage 2 in one kernel)

The reference runs Stage 1 and Stage 2 as separate kernel launches. The intermediate buffer (`h = SwiGLU(gate_up_output)`) is written to HBM between stages.

Fusing both stages eliminates this intermediate buffer:

```text
Current:  [Stage1 kernel] → h [HBM] → [Stage2 kernel] → output [HBM]
Fused:    [single kernel: Stage1 → SwiGLU → Stage2 → weighted reduce]
```

This is the most ambitious fusion — requires enough shared memory or register space to hold the intermediate activation tile.

### 3. Shared Expert Fusion

The shared expert (index 256) is **always selected for every token** with weight=1.0. It never needs routing — it's just a dense GEMM over all M tokens.

The reference treats it identically to routed experts (included in the top-k loop). A custom kernel can:

- Compute the shared expert as a standard dense a4w4 GEMM (no routing overhead, better utilization)
- Fuse its output directly into the weighted reduction accumulator

### 4. Custom Expert Dispatch / Wave Scheduling

With E=257 experts but only 9 active per token, most expert slots receive zero tokens. The CK kernel uses a fixed tile strategy that may launch empty wavefronts for inactive experts.

A compact dispatch strategy:

- Only launch blocks for experts that actually receive tokens (`tokens_per_expert[e] > 0`)
- Particularly impactful at small batch sizes (bs=4, 64) where few experts are hot

### 5. Split-K for Large M (EP-on config)

For the EP-on benchmarks (bs=1024, E=33, d_expert=2048), each expert receives ~280 tokens on average. The GEMMs are large enough that split-K parallelism within a single expert reduces latency by giving multiple CU groups work on the same expert's GEMM.

### 6. Custom Tiling / Scheduling for Small Batch

At bs=4 with E=257 experts, each expert sees ~0.14 tokens on average (mostly 0, occasionally 1). The CK kernel's tile sizes are tuned for larger M. A custom schedule that batches multiple near-empty experts into a single block can improve occupancy.

---

## Profiling

```bash
# Where is the time for each config?
for bs in 4 64 256 1024; do
    echo "=== bs=$bs ===" && python -c "
import torch; from reference import generate_input, ref_kernel
import triton
data = generate_input(dhidden=7168, dexpert=256 if $bs<=256 else 2048,
    nroutedexperts=256 if $bs<=256 else 32, nexpertspertoken=8,
    nsharedexperts=1, bs=$bs, seed=0)
t = triton.testing.do_bench(lambda: ref_kernel(data))
print(f'bs=$bs: {t:.3f} ms')
"
done

# Deep HW analysis
omniperf profile --name moe -- python benchmark.py
omniperf analyze -p workloads/moe/
```

## References

- [Official moe-mxfp4 README](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/moe-mxfp4)
- [ROCm/aiter fused_moe](https://github.com/ROCm/aiter)
- [MXFP4 quantization on AMD](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [CK GroupedGemm](https://github.com/ROCm/composable_kernel)
