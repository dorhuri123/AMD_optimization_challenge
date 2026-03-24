# mixed-mla: MLA Decode with Mixed Quantization

**Owner:** Person A
**Deadline:** April 6, 2026
**Hardware:** AMD Instinct MI355X (CDNA3)
**Official spec:** [amd_202602/mixed-mla](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/mixed-mla)

---

## What This Challenge Is

Implement a fast MLA (Multi-Head Latent Attention) **decode kernel** for DeepSeek R1 on MI355X.

The task gives you the KV cache in **three formats simultaneously** (bf16, fp8, mxfp4) — you choose which to use. The reference uses fp8 Q + fp8 KV via AITER's a8w8 persistent kernel. To win, you need to go further.

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
| v_head_dim | 512 (= kv_lora_rank) |
| sm_scale | 1/√576 |
| q_seq_len | 1 (decode only) |

## The Three KV Cache Formats (pick one or mix)

```python
kv_data["bf16"]   # (total_kv, 1, 576)     bfloat16        — baseline
kv_data["fp8"]    # (kv_fp8, scale)         fp8 + scalar    — 2x BW savings vs bf16
kv_data["mxfp4"]  # (kv_fp4x2, scale_e8m0) fp4x2 + e8m0   — 4x BW savings vs bf16
```

## Reference Kernel Variants (AITER)

| Q_DTYPE | KV_DTYPE | AITER kernel | Description |
| --- | --- | --- | --- |
| `"fp8"` **(default)** | `"fp8"` **(default)** | `mla_a8w8_qh16_...` | fp8 Q + fp8 KV — fastest |
| `"bf16"` | `"fp8"` | `mla_a16w8_qh16_...` | bf16 Q + fp8 KV |
| `"bf16"` | `"bf16"` | `mla_a16w16_qh16_...` | bf16 Q + bf16 KV — highest precision |

**Note**: fp8 Q + bf16 KV is not valid (no a8w16 kernel). fp8 Q + mxfp4 KV **does not exist in AITER**.

### Reference Latency (MI355X)

| Case | a8w8 (μs) | a16w16 (μs) | a8w8 speedup |
| --- | --- | --- | --- |
| bs=4, kv=1k | ~118 | ~162 | 1.4x |
| bs=4, kv=8k | ~113 | ~177 | 1.6x |
| bs=64, kv=8k | ~171 | ~353 | 2.1x |
| bs=256, kv=8k | ~349 | ~814 | 2.3x |

## Files

| File | Purpose |
| --- | --- |
| `reference.py` | AITER a8w8 baseline — do not modify |
| `task.py` | `input_t`, `output_t`, `TestSpec` definitions |
| `task.yml` | 8 benchmark configs (leaderboard = geometric mean) |
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

Checked against the a8w8 reference with `rtol=2e-02, atol=8e-03`.

All approaches (a8w8, a16w8, mxfp4 dequant→torch) are well within tolerance.

## Benchmark Configs (from task.yml)

| batch_size | q_seq_len | kv_seq_len | Notes |
| --- | --- | --- | --- |
| 4 | 1 | 1024 | Small batch, short ctx — launch overhead dominates |
| 4 | 1 | 8192 | Small batch, long ctx — BW-bound |
| 32 | 1 | 1024 | |
| 32 | 1 | 8192 | |
| 64 | 1 | 1024 | |
| 64 | 1 | 8192 | |
| 256 | 1 | 1024 | Large batch, short ctx |
| 256 | 1 | 8192 | Large batch, long ctx — highest total KV traffic |

Ranking = geometric mean latency across all 8.

---

## Optimization Opportunities — Feasibility Analysis

### What the reference does

The a8w8 persistent kernel:

1. Quantizes Q from bf16 → fp8 (per-tensor dynamic, separate kernel)
2. Loads fp8 KV from HBM (pre-quantized by `generate_input`)
3. Runs MLA decode: Q@K^T → softmax → ·V, with `NUM_KV_SPLITS=32`
4. Outputs bf16 (total_q, 16, 512)

The persistent kernel uses hand-tuned ASM (assembly) and is **highly optimized**. Beating it requires either:

- Reducing total HBM traffic (use MXFP4 KV = 2x less than fp8)
- Eliminating overhead (skip Q quantization, better split tuning)
- Both

### Opt 1: `num_kv_splits` tuning — LOW EFFORT, 10-20% potential

**Status: PLAUSIBLE — try first**

Reference hardcodes `NUM_KV_SPLITS = 32`. This controls how the KV sequence is partitioned across CUs. The optimal value depends on batch_size × kv_seq_len × CU_count.

For small configs (bs=4, kv=1k → only 4K tokens total), 32 splits means each split handles ~125 tokens — may be too fine-grained. For large configs (bs=256, kv=8k → 2M tokens), 32 may be too few.

```python
# Sweep per benchmark config:
optimal_splits = {}
for bs, kvlen in configs:
    for splits in [4, 8, 16, 32, 48, 64]:
        t = bench(bs, kvlen, num_kv_splits=splits)
        # pick best
```

### Opt 2: Skip Q quantization (use a16w8: bf16 Q + fp8 KV) — LOW EFFORT, SMALL

**Status: PLAUSIBLE — easy to test**

The fp8 Q quantization (`quantize_fp8`) is a separate kernel launch that:

- Reads all of Q (bf16) from HBM
- Computes amax (reduction)
- Writes Q (fp8) back to HBM

For small batch sizes (bs=4, Q is only 4×16×576 = 36KB), the kernel launch overhead may exceed the BW savings from fp8 Q. Using `a16w8` skips this entirely.

But for large bs (256 × 16 × 576 × 2 bytes = 4.7MB Q), fp8 Q halves the Q load inside the attention kernel. Trade-off is config-dependent.

**Approach: use a16w8 for small batch, a8w8 for large batch.**

### Opt 3: Custom Triton kernel with MXFP4 KV — HIGH EFFORT, HIGHEST REWARD

**Status: PLAUSIBLE but requires writing a full attention kernel**

**AITER does NOT support MXFP4 KV in `mla_decode_fwd`.** The kernel only accepts bf16, fp8, or uint8 (3BUFFER) KV buffers. To use the provided `kv_data["mxfp4"]`, you must write a custom kernel.

The payoff: MXFP4 KV is **2x less HBM traffic** than fp8 KV (4 bits vs 8 bits per value, plus E8M0 scales are amortized over 32 elements). For the memory-bound large configs, this directly translates to ~2x speedup on KV load time.

The challenge: the reference ASM kernel is already at ~70-80% of HBM bandwidth. A Triton kernel will have lower instruction-level efficiency. The net win depends on whether the 2x BW reduction overcomes the ~20-30% Triton overhead.

**Two sub-strategies:**

**A) Dequant-in-register (fp8 Q × dequant(mxfp4 KV) → fp32 accumulator):**

```python
@triton.jit
def mla_decode_mxfp4_kernel(...):
    # Load Q tile (fp8 or bf16) — small, fits in registers
    # For each KV tile along sequence:
    #   Load fp4x2 KV tile from HBM (half the bytes of fp8)
    #   Load E8M0 scales (1 byte per 32 elements)
    #   Dequant: unpack fp4x2 → 2×fp4 → multiply by scale → bf16
    #   score += Q_tile @ K_tile^T  (in fp32)
    # softmax(scores)
    # For each KV tile (value path, first 512 dims only):
    #   Same dequant flow
    #   output += softmax_score × V_tile
```

**B) Full fp4×fp4 compute (quantize Q to mxfp4 too):**

Uses MI355X native MXFP4 MFMA instructions for Q@K^T. Higher throughput but more quantization noise. Accuracy may be tight given `rtol=2e-02, atol=8e-03`.

### Opt 4: Exploit MQA pattern (1 KV head, 16 Q heads) — MEDIUM EFFORT, MEDIUM REWARD

**Status: LIKELY ALREADY EXPLOITED by AITER**

AITER's persistent kernel knows `nhead_kv=1` and `nhead=16`. The ASM kernel almost certainly loads each KV tile once and broadcasts to all 16 Q heads. Verify with `rocprof` — if KV HBM traffic is 1/16th of what 16 independent heads would read, it's already exploited.

If NOT exploited: writing a custom kernel that loads KV into LDS once and computes 16 Q-head scores would give up to 16x reduction in KV traffic. This would be a huge win.

**Verdict: verify before investing effort.**

### Opt 5: 3BUFFER layout (FP8 nope + FP32 scales + BF16 rope) — MEDIUM EFFORT, SMALL REWARD

**Status: UNLIKELY TO HELP for speed**

3BUFFER uses per-channel quantized FP8 KV (4 FP32 scales per 512-dim nope buffer) with separate BF16 rope storage. This is for **accuracy**, not speed — total bytes per KV token is actually more than plain fp8 (512 fp8 + 4×4 fp32 + 64×2 bf16 = 640+16+128 = 784 bytes vs 576 bytes for fp8).

**Skip unless accuracy is a problem.**

### Opt 6: vLLM PR #36297 relevance

**Status: NOT APPLICABLE to this challenge**

PR #36297 fuses BMM + FP8 quant for `v_up_proj` — the layer **after** this challenge's output boundary. The challenge measures only the attention kernel (output: bf16 tensor). The `v_up_proj` step is not part of the measured path.

---

## Implementation Plan

| Priority | Optimization | Effort | Expected gain | When |
| --- | --- | --- | --- | --- |
| P0 | num_kv_splits sweep | 1 hour | 10-20% | Day 1 |
| P0 | Adaptive Q dtype (a16w8 for small bs) | 1 hour | 5-10% small bs | Day 1 |
| P1 | Custom Triton MXFP4 KV decode | 2-3 days | up to 2x on large configs | Day 2-4 |
| P2 | Verify MQA exploitation in AITER | 2 hours | 0% if already done | Day 1 |

## Profiling

```bash
# Where is the time — Q quant vs attention vs reduce?
rocprof --stats python benchmark.py

# HBM bandwidth utilization
omniperf profile --name mla -- python benchmark.py
omniperf analyze -p workloads/mla/

# KV traffic check (is MQA exploited?)
# If total_kv_bytes_read ≈ total_kv × 576 × dtype_size → MQA is exploited
# If total_kv_bytes_read ≈ 16 × total_kv × 576 × dtype_size → MQA NOT exploited
rocprof --stats --counter FETCH_SIZE python benchmark.py
```

## References

- [Official mixed-mla README](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602/mixed-mla)
- [AITER mla.py source](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)
- [AITER MLA decode tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)
- [vLLM ROCm attention backends](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)
