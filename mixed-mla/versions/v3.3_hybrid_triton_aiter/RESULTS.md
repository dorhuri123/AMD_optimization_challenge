# v3.3 — Hybrid Triton + AITER

**Date:** 2026-03-31
**Status:** NOT YET SUBMITTED (rate limited). Test passed on MI355X.

## Approach
- Bare Triton FP8 kernel for bs=4,kv=1024 (eliminates AITER+Q_quant overhead entirely)
- AITER cached a8w8 with metadata caching for all other configs
- Single-file submission (Triton kernel inlined)

## MI300X Benchmark Results (local)

| Config | Ref (ms) | Sub (ms) | Speedup |
|---|---|---|---|
| bs=4, kv=1024 | 0.408 | 0.063 | 6.48x |
| bs=4, kv=8192 | 0.406 | 0.282 | 1.44x |
| bs=32, kv=1024 | 0.410 | 0.272 | 1.51x |
| bs=32, kv=8192 | 0.413 | 0.274 | 1.51x |
| bs=64, kv=1024 | 0.387 | 0.262 | 1.47x |
| bs=64, kv=8192 | 0.468 | 0.273 | 1.71x |
| bs=256, kv=1024 | 0.490 | 0.270 | 1.81x |
| bs=256, kv=8192 | 0.817 | 0.544 | 1.50x |
| **geomean** | **0.461** | **0.247** | **1.86x** |

## MI355X Test Results
- Passed all 4 correctness tests (max_error: 0.0098 for Triton config, 0.0 for AITER configs)

## Estimated MI355X Leaderboard Score
- ~85 μs (est. from v2.0 of 92.534μs minus Triton savings on bs=4,kv=1024)

## Key Learnings
- Triton only wins on smallest config (bs=4,kv=1024) — AITER ASM is 10-45x faster on large configs
- Q FP8 cache by data_ptr is unsafe for leaderboard (eval changes seed each run)
- Fixed-scale Q quant is unsafe (correctness failures on some seeds)
