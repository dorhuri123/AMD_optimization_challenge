# Challenge 2: MLA Decode

**Owner:** Person A
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)

---

## What Is MLA?

Multi-Head Latent Attention (MLA) is the attention mechanism used in DeepSeek-V3/R1 and Kimi K2.5. Unlike standard MHA, MLA compresses the KV cache through low-rank projections:

- Standard MHA KV cache: ~8K dimensions per head
- MLA KV cache: **576 dimensions** (14x memory reduction)
- This compression is what enables running 600B+ models on a single node

The decode phase is the dominant bottleneck: for an output sequence of length 1K, you run 1K decode iterations.

## Architecture Overview

```
Input hidden state
       │
       ├──► q_proj → q_nope + q_rope (via RoPE)
       │
       └──► kv_proj → compressed_kv (576 dims, stored in KV cache)
                          │
                          ├──► k_nope  (absorbed into query)
                          └──► k_rope  (via RoPE)

Attention:
  scores = softmax((q_nope · k_nope^T + q_rope · k_rope^T) / sqrt(d))
  context = scores · v_proj(compressed_kv)

Output:
  out = v_up_proj(context)   ← this is where PR #36297 applies
        │
        └──► FP8 quantize → next layer input
```

## Our Key Optimization (PR #36297)

The original flow for the output projection:

```
context (bf16) → v_up_proj BMM → intermediate (bf16) [write to HBM]
                                                        ↓
                                              FP8 quantize [read + write]
```

Fused kernel (PR #36297):

```
context (bf16) → [fused BMM + FP8 quant] → output (fp8)  [one write]
                  intermediate stays in registers
```

**Result on MI300X: 5.2–6.6x speedup** on this kernel.
On MI355X (CDNA3, newer HBM3e) we expect at least the same.

## Files

| File | Purpose |
|------|---------|
| `reference.py` | Baseline from gpu-mode/reference-kernels — do not modify |
| `solution.py` | Our optimized kernel — edit this |
| `benchmark.py` | Performance comparison: reference vs solution |

## Getting the Official Baseline

```bash
# Copy from the cloned reference-kernels repo
cp $REF_KERNELS_DIR/AMD/mla-decode/reference.py ./reference.py
cp $REF_KERNELS_DIR/AMD/mla-decode/task.py ./task.py
cp $REF_KERNELS_DIR/AMD/mla-decode/task.yml ./task.yml
```

## Running

```bash
cd mla-decode

# Verify baseline correctness
python reference.py

# Run our solution
python solution.py

# Compare performance
python benchmark.py
```

## Submission

```bash
popcorn submit solution.py --problem mla-decode
```

## Optimization Spaces (in priority order)

### 1. Fused BMM + FP8 Output Projection (PR #36297) ← start here
Port the Triton fused kernel from vLLM PR #36297 into `solution.py`.
- File in vLLM: `vllm/kernels/triton/ops/bmm_fp8_quant.py`
- Need to retune autotuned configs for MI355X (different L2 cache / compute ratio vs MI300X)
- ROCm FP8: `torch.float8_e4m3fnuz` (max=240, not NVIDIA's 448)

### 2. KV Cache Memory Layout
- AITER uses paged KV layout — verify the benchmark uses the same layout
- Contiguous layout may be faster for short sequences; paged for long contexts
- Coalesced HBM access patterns for MI355X's HBM3e bandwidth

### 3. Fused RoPE + Attention
- RoPE is currently applied before attention in a separate pass
- Can fuse `q_rope` application into the attention kernel to save one HBM round-trip

### 4. `concat_mla_k` Kernel
- k_nope + k_rope concatenation is currently a separate kernel (FlashInfer's `concat_mla_k`)
- Investigate fusing this into the attention kernel or pre-computing during prefill

### 5. Full Decode Fusion
- Ultimate goal: one kernel covers attention + output projection + FP8 quant
- Saves multiple kernel launch overheads and HBM round-trips

### 6. AITER Assembly Kernel Tuning
- AITER's `mla_decode_fwd` is a hand-tuned assembly kernel providing up to 17x speedup
- Study its tile shapes and see if MI355X needs different block sizes

## Key Parameters (from AITER `mla_decode_fwd`)

```python
# Required inputs
q           # [batch, num_heads, kv_lora_rank + qk_rope_dim]
kv_buffer   # [num_pages, page_size, num_heads_kv, qk_head_dim]
o           # [batch, num_heads, kv_lora_rank]   output buffer
qo_indptr   # [batch + 1]
kv_indptr   # [batch + 1]
kv_indices  # [kv_indptr[-1]]
kv_last_page_lens  # [batch]
max_seqlen_q

# Optional
sm_scale    # default: 1/sqrt(qk_head_dim)
num_kv_splits  # auto-determined
```

## Useful AITER Snippets

```python
import aiter
from aiter import mla_decode_fwd

# Enable AITER in vLLM
import os
os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["VLLM_ROCM_USE_AITER_MLA"] = "1"

# Flash attention for prefill
from aiter import flash_attn_varlen_func
```

## Profiling

```bash
# Kernel-level timing
rocprof --stats python solution.py

# Deep HW analysis (identify memory vs compute bottleneck)
omniperf profile --name mla_run -- python solution.py
omniperf analyze -p workloads/mla_run/

# Quick bandwidth estimate
python -c "
import torch
# MI355X theoretical peak: ~6.5 TB/s HBM3e
# If your kernel is below ~70% of this, it's memory-bound and can be improved
"
```

## References

- [AITER MLA decode tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)
- [AITER-enabled MLA blog](https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html)
- [vLLM ROCm attention backends](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)
- [vLLM PR #36297](https://github.com/vllm-project/vllm/pull/36297) — our fused BMM+FP8 kernel
