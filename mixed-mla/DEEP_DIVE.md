# Mixed-MLA Deep Dive: Everything You Need to Know

A thorough explanation of the problem, the GPU concepts behind it, and the reasoning behind every optimization path. Written for someone with vLLM familiarity but new to low-level GPU kernel optimization.

---

## Table of Contents

1. [The Big Picture: What Are We Optimizing?](#1-the-big-picture)
2. [Attention Recap: From Standard to MLA](#2-attention-recap)
3. [The Decode Phase: Why It's Special](#3-the-decode-phase)
4. [GPU Memory Hierarchy: The Key to Everything](#4-gpu-memory-hierarchy)
5. [Quantization Formats: FP8, MXFP4, E8M0](#5-quantization-formats)
6. [The Reference Kernel: What AITER Does](#6-the-reference-kernel)
7. [The Challenge Interface: What We Get and What We Return](#7-the-challenge-interface)
8. [Optimization Paths: The Thinking Behind Each Move](#8-optimization-paths)
9. [The Leaderboard Landscape](#9-the-leaderboard-landscape)
10. [Glossary](#10-glossary)

---

## 1. The Big Picture

### What is this challenge?

We're given the **attention decode kernel** from DeepSeek R1 (a 671B parameter LLM) and asked to make it run faster on an AMD MI355X GPU.

The attention kernel is the innermost loop of LLM inference. Every single token the model generates requires one full pass through this kernel. If the model generates 1000 tokens, this kernel runs 1000 times. Shaving even 10 microseconds off it translates to 10 milliseconds saved per response — and at scale (millions of requests), that's enormous.

### Why AMD?

AMD's MI355X (CDNA3 architecture) is the latest competitor to NVIDIA's H100/H200. It has:

- **HBM3e memory**: ~6.5 TB/s bandwidth (faster than H100's 3.35 TB/s)
- **Native MXFP4 support**: hardware matrix cores that can multiply 4-bit values directly
- **304 CUs** (Compute Units): AMD's equivalent of NVIDIA SMs

The challenge is specifically about exploiting MI355X's strengths — particularly its massive memory bandwidth and native low-precision support.

### What does "E2E Model Speedrun" mean?

"End-to-end" means the competition covers the full inference pipeline, not just one isolated operation. Phase 1 has three challenges that together represent the main bottlenecks in DeepSeek R1 inference:

1. **Mixed-MLA** (this one): The attention layer
2. **MXFP4-MM**: The matrix multiplications (linear layers)
3. **MOE-MXFP4**: The Mixture-of-Experts routing + computation

These three operations account for ~90%+ of inference time.

---

## 2. Attention Recap: From Standard to MLA

### Standard Multi-Head Attention (MHA)

In a standard transformer, attention works like this:

```
Input: hidden_state [batch, seq_len, d_model]

For each attention head h:
  Q_h = hidden_state @ W_Q_h    → [batch, seq_len, d_head]     "What am I looking for?"
  K_h = hidden_state @ W_K_h    → [batch, seq_len, d_head]     "What do I contain?"
  V_h = hidden_state @ W_V_h    → [batch, seq_len, d_head]     "What information do I carry?"

  scores = Q_h @ K_h^T / sqrt(d_head)     → [batch, seq_len, seq_len]
  weights = softmax(scores)
  output_h = weights @ V_h                 → [batch, seq_len, d_head]

output = concat(output_1, ..., output_H) @ W_O
```

The **KV cache** stores K and V tensors from all previous tokens so we don't recompute them. For a model with 128 heads, d_head=128, and a 4K context:

```
KV cache per layer = 2 (K+V) × 4096 (tokens) × 128 (heads) × 128 (dim) × 2 (bf16 bytes)
                   = 256 MB per layer
For 60 layers      = 15.4 GB  ← this is why KV cache is a bottleneck
```

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

The insight: do all query heads really need their own K and V? Turns out, sharing K/V across query heads barely hurts quality but dramatically reduces KV cache size.

- **MQA** (Multi-Query): ALL query heads share 1 K head and 1 V head → 128x less KV cache
- **GQA** (Grouped-Query): Groups of query heads share K/V → 8-32x less KV cache

DeepSeek R1 uses **MQA with 1 KV head shared across 16 query heads**. This is crucial for our optimization — we'll come back to it.

### Multi-Head Latent Attention (MLA) — DeepSeek's Innovation

MLA goes further than MQA. Instead of storing separate K and V vectors, it stores a single **compressed latent vector** per token:

```
Standard MHA KV cache per token: K (128 dims) + V (128 dims) = 256 dims per head
                                 × 128 heads = 32,768 values per token

MLA KV cache per token:          ONE compressed vector of 576 dims
                                 That's it. 576 values per token.
                                 → 57x less memory than standard MHA
```

How does this work? Through **absorption**. During model training, the projection matrices (W_Q, W_K, W_V) are factored into low-rank decompositions. At inference time, we can "absorb" the K and V projection matrices into the Q projection, so that:

```
Instead of:
  K = kv_cache @ W_K           ← separate K projection
  scores = Q @ K^T

We compute (absorbed form):
  Q_absorbed = Q @ W_K^T       ← absorb W_K into Q (done once before attention)
  scores = Q_absorbed @ kv_cache^T   ← directly use the compressed kv_cache
```

The compressed `kv_cache` has 576 dimensions:
- **First 512 dims** (`kv_lora_rank`): The "latent" representation — used for both key matching AND value extraction
- **Last 64 dims** (`qk_rope_dim`): RoPE (Rotary Position Embedding) component — only used for key matching

This means:
- For **score computation** (Q @ K^T): we use all 576 dims
- For **value extraction** (weights @ V): we use only the first 512 dims

### Why "Mixed" MLA?

The "mixed" refers to **mixed quantization** of the KV cache. The challenge provides the same KV cache in three formats simultaneously:

- **bf16**: Full precision, 576 × 2 bytes = 1,152 bytes per token
- **fp8**: Half precision, 576 × 1 byte = 576 bytes per token
- **mxfp4**: Quarter precision, 576 × 0.5 bytes + scales = ~300 bytes per token

You choose which format to use. Lower precision = less memory to read = faster kernel (if you can handle the dequantization efficiently).

---

## 3. The Decode Phase: Why It's Special

LLM inference has two phases:

### Prefill Phase
Process the entire input prompt at once. Q has many tokens, KV has many tokens.
- **Compute-bound**: Lots of matrix multiplications, GPU compute units are busy
- Shape: Q is [batch, prompt_len, dim] — could be hundreds of tokens

### Decode Phase (this challenge)
Generate one token at a time, autoregressively. Q has **1 token**, KV has **all previous tokens**.
- **Memory-bound**: Q is tiny, but we must read the ENTIRE KV cache from memory
- Shape: Q is [batch, **1**, dim] — always 1 token per sequence

```
Decode attention arithmetic intensity:

  For bs=64, kv_len=8192, dim=576:
    Q tensor:   64 × 1 × 16 × 576 × 2 bytes  =  1.1 MB  (tiny!)
    KV cache:   64 × 8192 × 1 × 576 × 2 bytes = 603 MB   (huge!)
    Compute:    64 × 16 × 8192 × 576 FLOPs     = 4.8 GFLOP

    Arithmetic intensity = 4.8 GFLOP / 604 MB ≈ 8 FLOP/byte

  MI355X can do ~2400 TFLOPS but only read ~6.5 TB/s.
  At 8 FLOP/byte, the memory system is the bottleneck, not compute.
  The GPU's compute units are idle most of the time, waiting for data.
```

**This is the fundamental insight**: decode attention is **memory-bandwidth bound**. The winning strategy is to **reduce the bytes we read from HBM**, not to do more compute per second.

This is why quantization matters so much:
- bf16 KV: read 1,152 bytes per KV token
- fp8 KV: read 576 bytes per KV token (2x less → up to 2x faster)
- mxfp4 KV: read ~300 bytes per KV token (4x less → up to 4x faster)

---

## 4. GPU Memory Hierarchy: The Key to Everything

Understanding the GPU memory hierarchy is essential for kernel optimization. Think of it like a pyramid:

```
                    ┌──────────┐
                    │ Registers│  ~20 TB/s  (fastest, smallest)
                    │  per CU  │  256 KB per CU
                    ├──────────┤
                    │   LDS    │  ~12 TB/s  (shared within workgroup)
                    │ (Shared) │  64 KB per CU
                    ├──────────┤
                    │ L2 Cache │  ~6 TB/s
                    │          │  256 MB total
                    ├──────────┤
                    │   HBM    │  ~6.5 TB/s (slowest, largest)
                    │ (Global) │  288 GB total
                    └──────────┘
```

### HBM (High Bandwidth Memory)
The main GPU memory. All tensors live here. Reading from HBM is the primary bottleneck for decode attention. MI355X has HBM3e at ~6.5 TB/s — incredibly fast by CPU standards, but still the bottleneck when you need to read 600 MB of KV cache per decode step.

### LDS (Local Data Share) — AMD's name for Shared Memory
A small, fast scratchpad shared by all threads in a workgroup (block). 64 KB per CU. This is where we can load a KV tile once and let all 16 query heads read from it — the MQA optimization.

### Registers
The fastest storage. Each thread has its own registers. Ideal for the current Q vector and partial accumulations. The key to "dequant in registers" — we load compressed fp4x2 data from HBM, unpack it into registers as bf16, use it for computation, and never write the decompressed version back to memory.

### L2 Cache
Sits between the CUs and HBM. Automatically caches recently accessed data. Important for small tensors like Q (which is read by every CU) — if Q fits in L2, it's effectively free to access.

### What "memory-bound" really means

A kernel is **memory-bound** when:

```
Time to read data from HBM > Time to compute on that data
```

For decode attention:
```
Time to read KV cache = 604 MB / 6.5 TB/s = 93 μs
Time to compute        = 4.8 GFLOP / 2400 TFLOPS = 2 μs

The GPU finishes computing in 2 μs but waits 93 μs for data.
The compute units are idle 98% of the time!
```

This is why reducing memory reads (via quantization) is the dominant optimization strategy.

---

## 5. Quantization Formats: FP8, MXFP4, E8M0

### Why Quantize?

Quantization reduces the number of bits per value. Fewer bits = fewer bytes to read from HBM = faster kernel. The trade-off is accuracy — but for attention scores (which go through softmax anyway), slight inaccuracies are tolerable.

### BFloat16 (bf16) — The Baseline

```
bf16: 1 sign + 8 exponent + 7 mantissa = 16 bits
Range: ±3.4 × 10^38
Precision: ~3 decimal digits
Size: 2 bytes per value
```

Standard training and inference precision. This is our baseline — no quantization.

### FP8 (e4m3fnuz on AMD, e4m3fn on NVIDIA)

```
fp8: 1 sign + 4 exponent + 3 mantissa = 8 bits
Range: ±240 (AMD) or ±448 (NVIDIA)
Precision: ~1.5 decimal digits
Size: 1 byte per value

Quantization: per-tensor (one scale factor for the entire tensor)
  scale = max(abs(tensor)) / fp8_max
  fp8_val = round(tensor / scale)
  To dequantize: bf16_val = fp8_val × scale
```

**Per-tensor** means one single scale factor for the entire KV cache. Simple and fast to dequantize (just multiply by a scalar), but limited dynamic range — if one value is 100 and another is 0.01, the 0.01 gets quantized to zero.

Why AMD uses `e4m3fnuz` vs NVIDIA's `e4m3fn`: AMD's format has a different NaN/zero encoding. The `uz` means "unsigned zero" — there's no negative zero, giving one extra representable value. Max value is 240 instead of 448.

### MXFP4 (Microscaling FP4) — The Big Opportunity

```
fp4 (E2M1): 1 sign + 2 exponent + 1 mantissa = 4 bits
Representable values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}  (only 8 positive values!)
Size: 0.5 bytes per value (two values packed into one byte as "fp4x2")

Scaling: per-block of 32 elements (one E8M0 scale per 32 values)
  E8M0: 8-bit exponent-only format → value = 2^(byte - 127)
  This is a power-of-2 scale (no mantissa), so multiplication is just an exponent add

Quantization:
  For each block of 32 values:
    scale_exponent = floor(log2(max(abs(block)))) + bias
    scale = 2^(scale_exponent - 127)
    fp4_vals = round(block / scale)   → each clipped to {0, 0.5, 1, ..., 6}
```

**Why is block-32 scaling better than per-tensor?**

Per-tensor FP8: if one token in the KV cache has attention logit 50.0 and another has 0.001, the scale is set by the 50.0, and 0.001 gets quantized to 0.

Block-32 MXFP4: each group of 32 consecutive values gets its own scale. If the 0.001 is in a different block from the 50.0, it gets a much smaller scale and preserves its precision. This **dramatically improves accuracy** despite using only 4 bits.

**How fp4x2 packing works:**

```
One byte stores two FP4 values:
  byte = (high_nibble << 4) | low_nibble
  low_nibble  = even-indexed element (bits 0-3)
  high_nibble = odd-indexed element (bits 4-7)

So a 576-dim vector becomes 288 bytes (576/2) of fp4x2 data
Plus 18 bytes of E8M0 scales (576/32 = 18 blocks × 1 byte each)
Total: 306 bytes vs 576 bytes (fp8) vs 1152 bytes (bf16)
```

**MI355X native MXFP4 support**: The matrix cores (MFMA instructions) can directly consume fp4x2 data and E8M0 scales. This means the hardware does the dequantization for free during matrix multiplication — no software overhead. However, this only works for structured GEMM operations (matrix × matrix), not for the arbitrary memory access patterns in attention.

### The shuffle_weight format

AMD's matrix cores expect data in a specific "shuffled" layout for maximum throughput. The `shuffle_weight((16,16))` function rearranges data into 16×16 tiles that match the hardware's data consumption pattern. This is relevant for the GEMM challenges but less so for attention (which has irregular access patterns).

---

## 6. The Reference Kernel: What AITER Does

### What is AITER?

AITER (AMD Inference Toolkit for Enhanced ROCm) is AMD's library of highly optimized GPU kernels for LLM inference. Think of it as AMD's equivalent of NVIDIA's TensorRT or FlashAttention — hand-written assembly code that squeezes maximum performance from the hardware.

The reference kernel uses AITER's `mla_decode_fwd` function, which is a **persistent-mode** kernel written in **assembly language** (not Triton, not CUDA-C — actual ISA-level hand-tuned code).

### What "persistent mode" means

A normal GPU kernel:
```
Launch kernel → each thread block processes one tile → kernel exits → launch next kernel
```

A persistent kernel:
```
Launch kernel → thread blocks stay alive → fetch work from a queue → process tile
→ fetch next work → process → ... → all work done → kernel exits
```

Benefits:
- **Zero kernel launch overhead** between tiles (launching a kernel takes ~5-10 μs)
- **Better load balancing** — fast thread blocks take more work, slow ones take less
- **Amortized setup cost** — metadata computation happens once

The reference uses persistent mode with these metadata structures:
- `work_meta_data`: tells each CU which batch element and KV range to process
- `work_indptr`, `work_info_set`: work distribution tables
- `reduce_indptr`, `reduce_final_map`, `reduce_partial_map`: how to merge partial results

### The reference kernel's flow

```
1. Prepare metadata (CPU-side, ~0 cost):
   get_mla_metadata_v1() → fills work distribution buffers

2. Quantize Q from bf16 → fp8 (separate kernel launch):
   scale = max(abs(Q)) / 240
   Q_fp8 = round(Q / scale).to(fp8)
   Cost: reads Q (bf16) from HBM, writes Q (fp8) to HBM

3. Run mla_decode_fwd (persistent ASM kernel):
   For each batch element b (persistent scheduling):
     For each KV split s (NUM_KV_SPLITS = 32):
       Load Q[b] from HBM (fp8, ~576 bytes per head × 16 heads)
       For each KV tile in split s:
         Load KV[tile] from HBM (fp8, 576 bytes per token)
         Dequant: KV_bf16 = KV_fp8 × kv_scale (scalar multiply)
         Dequant: Q_bf16 = Q_fp8 × q_scale
         score += Q_bf16 @ KV_bf16^T        (in fp32 accumulator)
       partial_softmax[b][s] = online_softmax(scores)
       partial_output[b][s] = softmax_weights @ V[first 512 dims]

   Reduce across splits:
     output[b] = merge(partial_softmax[b][0..31], partial_output[b][0..31])

4. Output: (total_q, 16, 512) bf16
```

### What is "Split-K" (num_kv_splits)?

For a single query token attending to 8192 KV tokens, one CU would need to:
- Read all 8192 KV tokens sequentially
- Compute 8192 dot products
- Run softmax over all 8192 scores
- Weight-sum 8192 value vectors

This would take forever and waste the other 303 CUs. **Split-K** divides the KV sequence into `num_kv_splits` chunks:

```
KV sequence: [token_0, token_1, ..., token_8191]

Split into 32 chunks of 256 tokens each:
  CU 0: processes tokens 0-255     → partial_softmax_0, partial_output_0
  CU 1: processes tokens 256-511   → partial_softmax_1, partial_output_1
  ...
  CU 31: processes tokens 7936-8191 → partial_softmax_31, partial_output_31

Reduce phase: merge all 32 partial results using the log-sum-exp trick
  (this is mathematically exact — no approximation)
```

The **optimal number of splits** depends on:
- **Too few splits** (e.g., 4): not enough parallelism, CUs idle
- **Too many splits** (e.g., 128): each split is too small, overhead of reduce dominates
- **Sweet spot**: depends on batch_size × kv_seq_len ÷ num_CUs

This is why sweeping `num_kv_splits` is our first optimization.

### Online Softmax

Standard softmax requires two passes over the data:
```
Pass 1: max_val = max(scores)                    ← need to read all scores
Pass 2: output = exp(scores - max_val) / sum(...)  ← read all scores again
```

Online softmax does it in **one pass** using running statistics:
```
For each new score s_i:
  if s_i > running_max:
    # Rescale all previous results
    correction = exp(old_max - s_i)
    running_sum *= correction
    running_output *= correction
    running_max = s_i
  running_sum += exp(s_i - running_max)
  running_output += exp(s_i - running_max) * v_i
```

This is critical because we can process KV tiles in a streaming fashion without storing all scores in memory. The reference kernel uses this internally, and any custom kernel must too.

---

## 7. The Challenge Interface

### What we receive

```python
data = (q, kv_data, qo_indptr, kv_indptr, config)
```

**`q`**: `(total_q, 16, 576)` bfloat16 — The absorbed query vectors.
- `total_q = batch_size × q_seq_len` (and q_seq_len is always 1 for decode)
- 16 heads, each 576-dimensional (512 nope + 64 rope, already absorbed)
- This is small: for bs=256, it's 256 × 16 × 576 × 2 = 4.7 MB

**`kv_data`**: A dict with three versions of the SAME KV cache:
```python
kv_data["bf16"]   # (total_kv, 1, 576) bfloat16
kv_data["fp8"]    # (fp8_tensor, scalar_scale)  — pre-quantized
kv_data["mxfp4"]  # (fp4x2_tensor, e8m0_scales) — pre-quantized
```
- `total_kv = batch_size × kv_seq_len`
- The "1" dimension is `num_kv_heads = 1` (MQA — single shared KV head)
- This is large: for bs=256, kv=8192, bf16: 256 × 8192 × 576 × 2 = 2.4 GB

**`qo_indptr`** and **`kv_indptr`**: `(batch_size + 1,)` int32
These are "CSR-style" index pointers for variable-length sequences:
```python
# If batch has 3 sequences with kv lengths [100, 200, 150]:
kv_indptr = [0, 100, 300, 450]
# Sequence 0's KV tokens: kv_buffer[0:100]
# Sequence 1's KV tokens: kv_buffer[100:300]
# Sequence 2's KV tokens: kv_buffer[300:450]
```
In our challenge, all sequences have the same kv_seq_len, so the indptrs are uniform.

**`config`**: Dict with all the architecture constants.

### What we return

```python
output: (total_q, 16, 512) bfloat16
```

For each query token and each of the 16 heads, a 512-dimensional output vector. Note it's 512 (kv_lora_rank), not 576 — we only use the first 512 dims as values.

### Accuracy requirement

Our output must match the reference (fp8 a8w8 kernel) within:
```
rtol = 2e-02  (2% relative tolerance)
atol = 8e-03  (0.008 absolute tolerance)
```

This is fairly generous. Even MXFP4 (4-bit) achieves max absolute differences of ~8e-4, well within tolerance.

---

## 8. Optimization Paths: The Thinking Behind Each Move

### Path 1: num_kv_splits Tuning

**The thought process:**

The reference hardcodes `NUM_KV_SPLITS = 32`. But is 32 optimal for every configuration?

Consider the extremes:
- **bs=4, kv=1024**: total_kv = 4096 tokens. With 32 splits, each split handles 128 tokens. MI355X has 304 CUs. So only 4 × 32 = 128 CUs are active out of 304 — **58% of the GPU is idle**.
  - Fewer splits (8-16) might let each CU do more work but wouldn't help parallelism
  - More splits (64) would give 256 active CUs but each split handles only 64 tokens

- **bs=256, kv=8192**: total_kv = 2M tokens. With 32 splits per sequence, we have 256 × 32 = 8192 work items for 304 CUs. Each CU processes ~27 work items. This seems well balanced.
  - Fewer splits (16): 4096 work items, each handles 512 tokens — might be more efficient per item
  - More splits (64): 16384 work items — more reduce overhead

The optimal value depends on the balance between:
```
More splits → more parallelism (good) + more reduce overhead (bad)
Fewer splits → less overhead (good) + less parallelism (bad)
```

**Why this is low-hanging fruit:** We don't change any algorithm, just one integer parameter. The sweep takes 5 minutes and could yield 10-20%.

**Expected outcome:** Modest improvement (5-20%), mainly on small-batch configs where the default 32 splits causes under-utilization.

---

### Path 2: Skip Q Quantization for Small Batches (a16w8)

**The thought process:**

The reference does this every call:
```
1. [Kernel launch] Read Q (bf16, 4.7 MB for bs=256) from HBM
2. [Kernel launch] Compute amax (reduction over all Q values)
3. [Kernel launch] Write Q (fp8, 2.35 MB) to HBM
4. [Kernel launch] mla_decode_fwd reads Q (fp8) from HBM
```

Steps 1-3 are the `quantize_fp8` function. For **small batches**:
```
bs=4: Q is 4 × 16 × 576 × 2 = 73 KB
  quantize_fp8 overhead: kernel launch (~5 μs) + read 73 KB + write 36 KB ≈ 5-10 μs
  Saving inside attention: Q is loaded ~32 times (num_kv_splits), saving 73-36=37 KB per load
    = 37 KB × 32 = 1.2 MB less HBM traffic
    At 6.5 TB/s: saves ~0.2 μs

  Cost: 5-10 μs for quantization
  Benefit: 0.2 μs in attention
  → Quantization is a NET LOSS for small batch!
```

For **large batches**:
```
bs=256: Q is 256 × 16 × 576 × 2 = 4.7 MB
  quantize_fp8 overhead: read 4.7 MB + write 2.35 MB ≈ 1.1 μs + launch overhead
  Saving inside attention: 2.35 MB × 32 = 75 MB less HBM traffic ≈ 11.5 μs saved
  → Quantization is a NET WIN for large batch
```

AITER supports `a16w8` (bf16 Q + fp8 KV) which skips Q quantization entirely. The plan:
- Small batch (bs ≤ threshold): use a16w8
- Large batch: use a8w8 (with Q quantization)

**Expected outcome:** 5-10% on small batch configs, neutral on large.

---

### Path 3: Custom Triton Kernel with MXFP4 KV (The Big Win)

**The thought process:**

This is where the real speedup lives. Let me walk through the reasoning step by step.

**Step 1: Why MXFP4?**

The decode kernel is memory-bound. The dominant cost is reading the KV cache:

```
KV bytes per token (total_kv × bytes_per_token):

  bf16:  576 × 2 = 1,152 bytes/token
  fp8:   576 × 1 = 576 bytes/token      (2x less than bf16)
  mxfp4: 288 + 18 = 306 bytes/token     (1.9x less than fp8, 3.8x less than bf16)
         ↑ fp4x2  ↑ E8M0 scales

For bs=256, kv=8192:
  fp8 KV traffic:   256 × 8192 × 576 = 1.2 GB
  mxfp4 KV traffic: 256 × 8192 × 306 = 0.64 GB
  Savings: 0.56 GB less to read → at 6.5 TB/s, that's ~86 μs saved
```

**Step 2: Why can't we just use AITER with MXFP4?**

AITER's `mla_decode_fwd` only accepts bf16, fp8, or uint8 (3BUFFER) KV buffers. There is **no code path** for MXFP4 KV in the MLA decode kernel. AITER has MXFP4 support in its flash attention prefill kernel and in its GEMM kernels, but not in the paged decode attention kernel.

This means we have two options:
1. Dequantize MXFP4 → fp8, then call AITER (but this costs an extra HBM write+read, losing most of the benefit)
2. Write a custom kernel that reads MXFP4 directly and dequants in registers

Option 2 is the only one that makes sense.

**Step 3: Why Triton and not CK/ASM?**

Writing assembly for AMD GPUs is extremely difficult and time-consuming. Composable Kernel (CK) has templates but they're designed for GEMM, not attention. Triton compiles Python-like code to optimized GPU assembly:

```python
@triton.jit
def kernel(ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    data = tl.load(ptr + offsets)
    # ... compute ...
    tl.store(out_ptr + offsets, result)
```

Triton on ROCm generates decent (not perfect) code. The top competitors at ~33 μs are likely using hand-tuned CK or ASM, but Triton can get us to the top 10-15 range (~40-50 μs).

**Step 4: The kernel algorithm**

The custom kernel needs to implement:

```
For each query token q_idx and each batch element:
  kv_start, kv_end = kv_indptr[batch_idx], kv_indptr[batch_idx + 1]

  Initialize: running_max = -inf, running_sum = 0, output = 0

  For each KV tile of BLOCK_KV tokens in [kv_start, kv_end]:
    # ─── Load phase ───
    Load fp4x2 KV tile from HBM (BLOCK_KV × 288 bytes)
    Load E8M0 scales (BLOCK_KV × 18 bytes)

    # ─── Dequant phase (in registers, no HBM write) ───
    For each packed byte: unpack two fp4 values
    For each block of 32: multiply by 2^(scale - 127)
    Now we have bf16 KV tile in registers

    # ─── Score phase ───
    For each of 16 query heads:
      score = dot(Q[head, :576], KV_tile[:, :576])  # score for this KV tile

    # ─── Online softmax update ───
    new_max = max(running_max, max(scores))
    correction = exp(running_max - new_max)
    running_sum = running_sum × correction + sum(exp(scores - new_max))
    running_max = new_max

    # ─── Value accumulation ───
    weights = exp(scores - running_max) / running_sum  (approximate, refined later)
    output += weights × KV_tile[:, :512]  # only first 512 dims for values

  Final output = output / running_sum  (final normalization)
```

**Step 5: MQA — The hidden multiplier**

Because `num_kv_heads = 1`, ALL 16 query heads read the same KV data. A naive kernel would load the KV tile 16 times (once per head). An optimized kernel:

```
Load KV tile into LDS (shared memory) → 1 HBM read
For head in 0..15:
  Load Q[head] from registers
  score[head] = Q[head] @ KV_tile_in_LDS   → 0 HBM reads (from LDS)
```

This gives ~16x reduction in KV HBM traffic (AITER's ASM kernel already does this, but a naive Triton kernel might not).

**Expected outcome:**
- Naive Triton MXFP4 (without MQA optimization): ~60-90 μs (rank 27-40)
- With MQA LDS sharing: ~40-60 μs (rank 10-25)
- With aggressive tuning: ~35-45 μs (rank 3-10)

**The risk:** Triton may not generate efficient enough code for the MXFP4 unpacking and E8M0 scaling. The reference ASM kernel has been hand-tuned for months. Our Triton kernel needs to overcome a ~20-30% "Triton tax" through the 2x bandwidth advantage of MXFP4.

---

### Path 4: Verify MQA Exploitation in AITER

**The thought process:**

Before writing a custom kernel, we should verify that AITER's ASM kernel already exploits MQA. If it loads KV once and broadcasts to 16 heads (likely), then MQA optimization in our Triton kernel just matches the reference — no advantage.

If AITER does NOT exploit MQA (unlikely but worth checking), then even staying with fp8 and fixing this would give a huge speedup.

**How to check:**

```bash
rocprof --stats python benchmark.py
# Look at total FETCH_SIZE (bytes read from HBM)
# Calculate: does it match 1× KV cache size or 16× KV cache size?
```

**Expected outcome:** AITER already exploits MQA. This confirms our custom kernel must do the same to be competitive.

---

### Path 5: Why PR #36297 Doesn't Apply Here

**The thought process:**

Your vLLM PR fuses `v_up_proj` (BMM) + FP8 quantization into one kernel. In the full MLA pipeline:

```
  [This challenge's boundary]
         ↓
  Q → attention(Q, KV) → context (total_q, 16, 512) bf16
                                       ↓
                          context @ W_v_up → intermediate (bf16)  ← PR #36297 fuses this
                                       ↓                            with FP8 quant below
                              FP8 quantize → next layer input     ←
```

The challenge measures ONLY the attention kernel — from Q+KV input to context output. The `v_up_proj` step happens afterward, outside the measured region.

However, **the technique from PR #36297 IS relevant conceptually**: the idea of fusing a computation with its subsequent quantization step to avoid an HBM round-trip. We apply the same principle in reverse:

- PR #36297: fuse **output projection + quantization** (avoid writing bf16, reading bf16, writing fp8)
- Our MXFP4 kernel: fuse **dequantization + attention** (avoid writing bf16 KV, reading bf16 KV)

Same principle, different direction. Your experience writing that PR directly informs how to think about the MXFP4 kernel's memory access patterns.

---

## 9. The Leaderboard Landscape

### Performance tiers (as of March 24, 2026)

```
#1-2:   ~33 μs    — Almost certainly custom ASM/CK kernels with MXFP4 KV
#3-10:  ~38-46 μs — Custom Triton or CK with MXFP4 KV + good tuning
#11-37: ~50-78 μs — Moderate custom work, possibly MXFP4 with less tuning
#38-55: ~89-153 μs — Minor tuning of AITER parameters, or partial custom kernels
#56-75: ~158-200 μs — Essentially the AITER reference with cosmetic changes
#76+:   >200 μs   — Broken or suboptimal implementations
```

### What the top submissions probably do

The #1 (32.7 μs) has 246 submissions — that's iterative tuning of a custom kernel. At that latency, they're reading ~30% less data than the fp8 reference, suggesting MXFP4 KV with tight MQA sharing and tuned split-K.

The gap between #2 (33.0 μs) and #3 (37.7 μs) is 14% — a full tier difference. The top 2 likely have a fundamentally better kernel (possibly CK-based or hand-tuned ASM), while #3+ uses Triton.

### Realistic targets

With dedicated effort over the remaining 13 days:

- **Conservative** (kv_splits + a16w8 only): rank ~50 → not competitive but on the board
- **Moderate** (basic Triton MXFP4): rank ~25-35 → respectable
- **Ambitious** (tuned Triton MXFP4 + MQA): rank ~10-15 → competitive
- **Heroic** (CK/ASM-level optimization): rank ~3-5 → top tier

---

## 10. Glossary

| Term | Meaning |
| --- | --- |
| **AITER** | AMD Inference Toolkit for Enhanced ROCm — AMD's optimized kernel library |
| **a8w8** | 8-bit activations × 8-bit weights (fp8 Q + fp8 KV) |
| **a16w8** | 16-bit activations × 8-bit weights (bf16 Q + fp8 KV) |
| **Absorbed query** | Q vector that has K/V projection matrices pre-multiplied in, so attention can be computed directly against the compressed KV cache |
| **bf16** | bfloat16: 16-bit float with 8-bit exponent (same range as fp32, less precision) |
| **BLOCK_KV** | Number of KV tokens processed per tile in a Triton kernel |
| **CDNA3** | AMD's 3rd-gen Compute DNA architecture (MI355X) |
| **CK** | Composable Kernel — AMD's template library for writing high-performance GPU kernels |
| **CSR** | Compressed Sparse Row — format using indptr arrays for variable-length sequences |
| **CU** | Compute Unit — AMD's equivalent of an NVIDIA SM (Streaming Multiprocessor) |
| **Decode** | Autoregressive token generation: Q has 1 token, KV has all previous tokens |
| **E2M1** | 2-bit exponent, 1-bit mantissa — the FP4 format used in MXFP4 |
| **E8M0** | 8-bit exponent, 0-bit mantissa — exponent-only scale factor format |
| **fp4x2** | Two FP4 values packed into one byte |
| **fp8** | 8-bit floating point (e4m3fnuz on AMD, e4m3fn on NVIDIA) |
| **GQA** | Grouped-Query Attention — groups of Q heads share K/V heads |
| **HBM** | High Bandwidth Memory — the GPU's main memory (~6.5 TB/s on MI355X) |
| **indptr** | Index pointer array for CSR-style variable-length batching |
| **kv_lora_rank** | 512 — the latent dimension of MLA's compressed KV cache |
| **LDS** | Local Data Share — AMD's name for shared memory (64 KB per CU) |
| **MFMA** | Matrix Fused Multiply-Add — AMD's matrix core instructions |
| **MLA** | Multi-Head Latent Attention — DeepSeek's compressed attention mechanism |
| **MQA** | Multi-Query Attention — all Q heads share one KV head |
| **MXFP4** | Microscaling FP4 — 4-bit values with per-32-element block scaling |
| **num_kv_splits** | How many chunks the KV sequence is split into for parallel processing |
| **Online softmax** | Single-pass softmax using running max and sum statistics |
| **Persistent kernel** | GPU kernel that stays resident and fetches work from a queue |
| **Prefill** | Processing the full input prompt at once (compute-bound) |
| **qk_rope_dim** | 64 — the RoPE dimension appended to the latent representation |
| **ROCm** | Radeon Open Compute — AMD's GPU compute platform (equivalent to CUDA) |
| **RoPE** | Rotary Position Embedding — encodes token position via rotation in embedding space |
| **sm_scale** | Softmax scale factor: 1/√(qk_head_dim) = 1/√576 ≈ 0.0417 |
| **Split-K** | Parallelization strategy: split the KV sequence into chunks processed by different CUs |
| **Triton** | OpenAI's GPU programming language — compiles Python-like code to GPU assembly |
| **v_head_dim** | 512 — output dimension per head (= kv_lora_rank, first 512 dims of KV buffer) |
