# Benchmark Results & Optimization Log

**Hardware:** AMD Instinct MI355X (gfx950, 256 CUs)
**Eval:** Torch 2.10.0+rocm7.1, AITER (JIT compiled)
**Submission:** popcorn-cli v1.3.6

---

## moe-mxfp4 (MoE MXFP4 Fused Kernel)

**Leaderboard geomean: ~185 us**

### Best Submission (lean wrapper)

Minimized Python overhead: pre-cached enums, skip unused tuple elements, removed redundant kwargs.

| Config | bs | E | d_expert | Time (us) | Ref Target |
|--------|-----|-----|----------|-----------|------------|
| TP=8 | 16 | 257 | 256 | 130 | 152.7 |
| TP=8 | 128 | 257 | 256 | 209 | 239.0 |
| TP=8 | 512 | 257 | 256 | 245 | 336.5 |
| TP=4 | 16 | 33 | 512 | 89 | 106.2 |
| TP=4 | 128 | 33 | 512 | 126 | 141.1 |
| TP=4 | 512 | 33 | 512 | 209 | 225.0 |
| EP-on | 512 | 33 | 2048 | 340 | 380.4 |

### Experiments Tried

| Approach | Result | Notes |
|----------|--------|-------|
| Clean AITER baseline | 139-352 us | Reference implementation |
| **Lean wrapper** | **130-340 us** | **Best: 3-6% faster** |
| block_size_M override | 138-351 us | Slightly worse than default |
| splitk=2 | No effect | Param ignored by AITER for per_1x32 |
| doweight_stage1=True | All tests FAIL | Incorrect numerical results |
| dispatch_policy=1 | 93-438 us | Much worse for large configs |
| AITER_BYPASS_TUNE_CONFIG=1 | 93-311 us | Worse for tuned E=257 configs |
| AITER_USE_NON_TEMPORAL_LOAD=1 | ~same | Already used by default |
| Direct fused_moe_2stages call | TIMEOUT | Extra JIT builds eat timeout |
| Cached get_2stage_cfgs | TIMEOUT | Same JIT issue |
| 1-stage ASM path | 2/3 tests FAIL | Incorrect for small batch |
| FlyDSL config injection | CRASH | Kernel name format issues |

### Key Findings

- AITER internally uses `fused_dynamic_mxfp4_quant_moe_sort` (fused quant+sort) already
- CK 2-stage pipeline is the default and well-tuned for E=257 shapes
- E=33 shapes use heuristic config (no CSV-tuned entry), potential optimization target
- FlyDSL has fp4x2 stage1/stage2 kernels compiled but default config doesn't select them
- The pipeline is: fused_quant_sort -> CK_stage1 -> CK_stage2 (3 GPU kernels)

---

## mxfp4-mm (MXFP4 Matrix Multiply)

**Leaderboard geomean: ~24 us** (best on board: ~9 us)

### Best Submission (lean triton quant + CK gemm)

| Config | m | n | k | Time (us) | Ref Target |
|--------|-----|------|------|-----------|------------|
| decode | 4 | 2880 | 512 | 19.3 | 8.2 |
| decode | 16 | 2112 | 7168 | 33.7 | 20.9 |
| small | 32 | 4096 | 512 | 19.6 | 9.5 |
| small | 32 | 2880 | 512 | 19.5 | 9.2 |
| medium | 64 | 7168 | 2048 | 24.3 | 12.7 |
| large | 256 | 3072 | 1536 | 23.0 | 12.2 |

### Experiments Tried

| Approach | Result | Notes |
|----------|--------|-------|
| **Lean triton quant + CK gemm** | **19-34 us** | **Best achievable with AITER API** |
| CUDA graph capture | 28-44 us | copy_ overhead exceeds launch savings |
| CUDA graph no clone | 28-35 us | Still slower (A copy cost) |
| gemm_a16wfp4_preshuffle (Triton) | KeyError | Triton can't canonicalize fp4x2 dtype |
| gemm_a16wfp4_ direct call | KeyError | Same Triton dtype issue |
| gemm_a8wfp4_preshuffle | ImportError | Not available in eval AITER version |
| gemm_afp4wfp4_pre_quant | CRASH | Crashes on eval machine |
| HIP fused quant (shuffle=True) | All tests FAIL | Scale layout mismatch with CK gemm |
| fp4_utils quant (shuffle=True) | All tests FAIL | Different shuffle format |
| Skip A scale shuffle | All tests FAIL | CK requires shuffled A scales |
| bf16 matmul (A @ B.T) | All tests FAIL | Max error 28.75 vs tolerance 0.01 |
| Triton AFP4WFP4 preshuffle | TIMEOUT | JIT compile exceeds 12 min |
| Triton AFP4WFP4 no-AOT | CRASH | Same dtype KeyError |
| Pre-allocated buffers | No improvement | AITER allocates internally |

### Key Findings

- A quantization overhead: ~55 us on MI300X (~10 us on MI355X estimated)
- e8m0_shuffle overhead: ~28 us on MI300X (~5 us on MI355X estimated)
- Actual GPU compute: quant ~3 us + shuffle ~5 us + gemm ~11 us = ~19 us total
- Python/CPU overhead dominates: aten::empty, hipModuleLaunchKernel
- Top 9 us submissions likely use a fused kernel that skips separate quant
- Triton path blocked by dtype compatibility (float4_e2m1fn_x2 not in triton._utils)
- CK gemm_a4w4 is the only working GEMM path on eval machine

### Gap Analysis

Our 24 us vs best 9 us = **2.7x gap**. The ~15 us difference is:
- ~10 us: A quantization (dynamic_mxfp4_quant Triton kernel launch)
- ~5 us: e8m0_shuffle (Triton kernel launch)

Top submissions eliminate these by fusing quant into the GEMM prologue.

---

## mixed-mla (MLA Decode)

**Leaderboard geomean: ~175 us**

### Best Submission (fp8 Q + fp8 KV)

| Config | bs | kv_len | Time (us) |
|--------|-----|--------|-----------|
| small | 4 | 512 | 124 |
| small | 4 | 2048 | 133 |
| medium | 16 | 512 | 136 |
| medium | 16 | 2048 | 183 |
| medium | 64 | 512 | 146 |
| medium | 64 | 2048 | 226 |
| large | 128 | 4096 | 177 |
| large | 256 | 8192 | 369 |

### Experiments Tried

| Approach | Result | Notes |
|----------|--------|-------|
| **fp8 Q + fp8 KV baseline** | **124-369 us** | **Best** |
| MXFP4 KV cache | Falls to fp8 fallback | MLA decode doesn't support fp4 KV |
| Tuned NUM_KV_SPLITS | 154-408 us | Fewer splits = worse for all configs |

---

## Environment Notes

- **MI300X (gfx942):** Cannot run MXFP4 GEMM kernels (no native FP4 matrix cores)
- **MI355X (gfx950):** Competition target, only accessible via popcorn remote submission
- **Rate limits:** 4-6 submissions/hour, 1 leaderboard/hour
- **JIT overhead:** First run compiles CK modules (~25s sorting, ~100s GEMM), eats into 12 min timeout
- **AITER version on eval:** Older than git HEAD, missing some Triton GEMM functions
