# mixed-mla: MLA Decode with Mixed Quantization

**Owner:** Person A
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)
**Official spec:** [amd_202602/mixed-mla](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/mixed-mla)

---

## What This Challenge Is

Implement a fast MLA (Multi-Head Latent Attention) **decode kernel** for DeepSeek R1 on MI355X.

The task gives you the KV cache in **three formats simultaneously** (bf16, fp8, mxfp4) — you choose which to use. The reference already uses fp8 Q + fp8 KV via AITER's a8w8 kernel. To win, you need to go further.

## DeepSeek R1 MLA Architecture

```
Absorbed query q: (total_q, 16 heads, 576 dims)
                              ↕  q_nope (512 dims): Q@K^T with kv_buffer (absorbed path)
                              ↕  q_rope  (64 dims): RoPE scores

KV cache: (total_kv, 1 KV head, 576 dims)   ← 1 shared KV head for all 16 query heads
  ├─ full 576 dims used as keys  (for score computation)
  └─ first 512 dims used as values (for output, = kv_lora_rank)

Output: (total_q, 16 heads, 512 dims) bfloat16
```

Key numbers (DeepSeek R1):

| Parameter | Value |
| --- | --- |
| num_heads (Q) | 16 |
| num_kv_heads | **1** (MQA — all 16 Q heads share the same KV) |
| kv_lora_rank | 512 (value dim) |
| qk_rope_dim | 64 |
| qk_head_dim | 576 (512 + 64) |
| sm_scale | 1/√576 |

## The Three KV Cache Formats (pick one or mix)

```python
kv_data["bf16"]   # (total_kv, 1, 576)     bfloat16        — baseline, ~6.5 TB/s BW
kv_data["fp8"]    # (kv_fp8, scale)         fp8 + scalar    — 2x BW savings vs bf16
kv_data["mxfp4"]  # (kv_fp4x2, scale_e8m0) fp4x2 + e8m0   — 4x BW savings vs bf16
```

Reference uses: fp8 KV + fp8 Q → **1.4–2.3x faster than bf16** on MI355X.

## Files

| File | Purpose |
| --- | --- |
| `reference.py` | AITER a8w8 baseline — do not modify |
| `task.py` | `input_t`, `output_t`, `TestSpec` definitions |
| `task.yml` | 8 benchmark configs (leaderboard uses geometric mean) |
| `submission.py` | **Edit this** — `custom_kernel(data)` implementation |
| `benchmark.py` | reference vs submission latency table |

## Running

```bash
cd mixed-mla
python reference.py    # verify baseline
python benchmark.py    # reference vs your submission
popcorn submit submission.py --problem mixed-mla
```

## Accuracy Target

Submissions are validated against the fp8 reference with `rtol=2e-02, atol=8e-03`.

---

## Optimization Opportunities

The reference is the AITER a8w8 kernel (fp8 Q + fp8 KV). Beat it by reducing memory bandwidth or fusing ops.

### 1. Use MXFP4 KV Cache (biggest lever, try first)

The task hands you `kv_data["mxfp4"]` for free. MXFP4 is 2× denser than fp8 and 4× denser than bf16.

Current `submission.py` has this path commented out — **the first thing to do is enable it** and verify AITER's MXFP4 MLA decode API works on MI355X:

```python
# In submission.py — uncomment and test:
kv_buffer, kv_scale = kv_data["mxfp4"]
q_fp8, q_scale = _quant_fp8_dynamic(q)
aiter.mla_decode_fwd(q_fp8, kv_buffer, ..., kv_scale=kv_scale, q_scale=q_scale)
```

Expected gain: additional ~1.4x over the fp8 KV path.

### 2. Fuse MXFP4 Dequantization into the Attention Kernel

The AITER reference dequantizes KV before the attention softmax. A custom Triton kernel can fuse the dequant directly into the attention score computation:

```
kv_buffer (fp4x2) → [load tile] → [dequant in registers] → [score += q @ k^T]
```
No intermediate fp16/fp32 materialization → saves one full KV-buffer read at higher precision.

### 3. Exploit the MQA Pattern (1 KV head, 16 Q heads)

Because `num_kv_heads = 1`, every KV tile is reused 16 times (once per Q head). A custom kernel can:
- Load each KV tile into shared memory **once**
- Compute scores for all 16 Q heads against that tile before evicting
- 16× reduction in KV HBM loads vs treating this as 16 independent attention ops

This is the single biggest structural opportunity the reference kernel may not fully exploit.

### 4. Tune `num_kv_splits` for MI355X

AITER's `mla_decode_fwd` auto-determines `num_kv_splits`. This controls how the KV sequence is split across CUs. Manually sweeping this for MI355X's CU count and HBM3e bandwidth profile can yield 10–20% gains.

```python
# Try different values:
for splits in [1, 2, 4, 8, 16]:
    aiter.mla_decode_fwd(..., num_kv_splits=splits)
```

### 5. FP8 Query Quantization Fusion

The current path quantizes `q` to fp8 before calling the kernel (two ops: quantize + attention). A custom kernel can fuse query quantization into the attention prologue — the fp8 scale computation touches q once, and the attention uses it immediately.

### 6. vLLM PR #36297 — Relevance

PR #36297 (fused BMM + FP8 for `v_up_proj`) applies to the layer **after** this challenge's output. Specifically:

- This challenge outputs: `context (total_q, 16, 512)` bf16
- PR #36297 fuses: `context @ W_v_up` + FP8 quant → the next layer's fp8 input

If the evaluation harness measures end-to-end including the output projection, PR #36297 is directly applicable. Check `task.yml` to confirm the output boundary.

---

## Benchmark Configs (from task.yml)

| bs | kv_len | Notes |
| --- | --- | --- |
| 4 | 512 | Small batch, short context |
| 4 | 2048 | Small batch, medium context |
| 16 | 512 | |
| 16 | 2048 | |
| 64 | 512 | |
| 64 | 2048 | |
| 128 | 4096 | Larger batch, long context |
| 256 | 8192 | Large batch, max context |

Ranking = geometric mean latency across all 8.

## Profiling

```bash
# Which path is hot — KV load or attention compute?
omniperf profile --name mla -- python benchmark.py
omniperf analyze -p workloads/mla/

# BW utilization (MI355X peak: ~6.5 TB/s HBM3e)
rocprof --stats python benchmark.py
# Look for: FETCH_SIZE / duration → compare to 6500 GB/s

# Sweep num_kv_splits
python -c "
import aiter, torch
# ... time mla_decode_fwd with num_kv_splits in [1,2,4,8,16]
"
```

## References

- [AITER MLA decode tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)
- [AITER MLA blog (MI300X)](https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html)
- [vLLM ROCm attention backends](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)
- [vLLM PR #36297](https://github.com/vllm-project/vllm/pull/36297) — fused BMM+FP8 (relevant for output projection)
