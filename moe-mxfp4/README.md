# Challenge 1: MXFP4 MoE (Mixture of Experts)

**Owner:** Person B (builds on MXFP4 GEMM work)
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)

---

## What Is MoE?

Mixture of Experts (MoE) replaces a single FFN layer with N expert FFNs, routing each token to the top-K experts. This enables models like DeepSeek-V3 (671B total, ~37B active) and gpt-oss (120B total, 4 active experts per token) to run efficiently.

The challenge: optimize the MoE layer with **MXFP4-quantized expert weights** on AMD MI355X.

**Target model: gpt-oss**
- 128 experts total (or 32 for 20B variant)
- Each token routes to **top-4 experts**
- No shared expert (unlike DeepSeek's architecture)
- Expert weights stored in MXFP4

## MoE Forward Pass

```
Input tokens: [batch * seq_len, hidden_dim]
                    │
                    ▼
            Router (linear layer)
                    │
                    ▼
           topK scores + expert indices
                    │
                    ▼
        Sort/scatter tokens to experts
                    │
              ┌─────┴─────┐
          Expert 0    Expert 1  ...  Expert N
          GEMM1+Act   GEMM1+Act      GEMM1+Act   ← MXFP4 weights
          GEMM2       GEMM2          GEMM2
              └─────┬─────┘
                    │
              Gather + weighted sum
                    │
                    ▼
                   Output
```

**The bottleneck shifts by batch size:**
- Small batch (decode): routing/sorting overhead dominates
- Large batch (prefill): GEMM throughput dominates

## Files

| File | Purpose |
|------|---------|
| `reference.py` | Baseline from gpu-mode/reference-kernels — do not modify |
| `solution.py` | Our optimized MoE kernel — edit this |
| `benchmark.py` | Performance comparison across batch sizes |

## Getting the Official Baseline

```bash
cp $REF_KERNELS_DIR/AMD/mxfp4-moe/reference.py ./reference.py
cp $REF_KERNELS_DIR/AMD/mxfp4-moe/task.py ./task.py
cp $REF_KERNELS_DIR/AMD/mxfp4-moe/task.yml ./task.yml
```

## Running

```bash
cd mxfp4-moe
python reference.py
python solution.py
python benchmark.py
```

## Submission

```bash
popcorn submit solution.py --problem mxfp4-moe
```

## Optimization Spaces (in priority order)

### 1. Grouped GEMM for Variable Expert Batch Sizes
Each expert sees a different number of tokens per step. A naïve loop over experts wastes GPU occupancy.
- Use a single "grouped GEMM" kernel with variable-size problem descriptors
- CK provides `DeviceGroupedGemm` template — study it first
- Triton alternative: `tl.constexpr`-based static loop unrolling won't work; use dynamic dispatch

```python
# Pseudo-structure for grouped GEMM:
# problems = [(M_0, N, K, A_ptr_0, B_ptr_0, C_ptr_0), ..., (M_127, N, K, ...)]
# One kernel launch processes all problems
```

### 2. Fused TopK Sort + Expert Dispatch
The routing pipeline: `router_logits → softmax → topK → sort by expert_id → scatter`.
The sort step is O(batch × K × log(N_experts)) and becomes the bottleneck at long contexts.

Optimization directions:
- Replace sort with **counting sort** (N_experts=128 → O(batch × K + N_experts))
- Fuse the scatter/gather into the GEMM kernel's index computation
- Use shared memory for the expert token count histogram

### 3. Online MXFP4 Activation Quantization
If the task requires MXFP4 activations (not just weights):
- Quantize activations inline in the dispatch kernel — avoid a separate quantization pass
- Each block of 32 activation values: compute max, derive FP8 scale, write FP4 values

### 4. Expert Weight Caching in Shared Memory
For small batches where the same expert is selected repeatedly:
- Pre-load frequently used expert weights into shared memory
- Amortize HBM load cost across multiple tokens routed to the same expert

### 5. Dynamic Grid Sizing
Naïve approach: launch a fixed grid and check `if expert_id == my_expert` — wastes threads.
Better: launch exactly `sum(tokens_per_expert_i > 0)` blocks, each assigned to one expert.

### 6. Kernel Fusion Budget Target
Production systems target **3–4 kernel launches** per MoE forward pass:
1. Router + topK
2. Sort/scatter
3. Grouped GEMM (GEMM1 + activation)
4. Grouped GEMM (GEMM2) + weighted sum

Current vLLM uses 7 kernels — halving this halves launch overhead on large-batch decode.

## Key Profiling Questions

1. **Where is the time?** At decode batch sizes (1–16), is it routing or GEMM?
2. **Expert load balance?** Measure token count per expert — uneven distribution hurts utilization.
3. **Memory access pattern?** MXFP4 weights are 4-bit but need 8-bit scale fetches — are these coalesced?

```bash
# Profile at different batch sizes
for bs in 1 8 32 128 512 2048; do
    echo "=== batch=$bs ===" && python benchmark.py --batch_size $bs
done

# HW counters
rocprof --stats --counter SQ_INSTS_VALU_MFMA_I32_16X16X32_FP8 python solution.py
omniperf profile --name moe_run -- python solution.py
```

## Relationship to MXFP4 GEMM Challenge

The MoE challenge is MXFP4 GEMM + routing logic. Complete the GEMM challenge first:
- The grouped GEMM kernel here reuses the MXFP4 GEMM tile structure
- Tile sizes, prefetching, and scale-factor handling are identical
- Only difference: variable M per expert instead of fixed M

## References

- [AITER FusedMoE operators](https://github.com/ROCm/aiter)
- [vLLM MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/)
- [CK GroupedGemm](https://github.com/ROCm/composable_kernel)
- [MXFP4 MoE on AMD blog](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
