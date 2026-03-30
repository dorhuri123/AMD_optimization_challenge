# v1.1 — kv_splits Tuned

**Date:** 2026-03-30
**Approach:** Tuned NUM_KV_SPLITS per config from MI300X sweep. a16w8 tested and rejected (always slower).

## kv_splits Sweep Results (MI300X)

| Config | Best splits | Default (32) ms | Tuned ms | Delta |
|---|---|---|---|---|
| bs=4, kv=1024 | 16 | 0.326 | 0.324 | -0.6% |
| bs=4, kv=8192 | 32 | 0.322 | 0.322 | 0% |
| bs=32, kv=1024 | 16 | 0.328 | 0.325 | -0.9% |
| bs=32, kv=8192 | 48 | 0.359 | 0.350 | -2.5% |
| bs=64, kv=1024 | 16 | 0.338 | 0.336 | -0.6% |
| bs=64, kv=8192 | 24 | 0.423 | 0.419 | -1.0% |
| bs=256, kv=1024 | 16 | 0.443 | 0.439 | -0.9% |
| bs=256, kv=8192 | 24 | 0.756 | 0.751 | -0.7% |

## a16w8 Test Results

a16w8 (bf16 Q + fp8 KV) is **always slower** than a8w8 (fp8 Q + fp8 KV) across all 8 configs. Eliminated from optimization paths.

## MI300X Local Benchmark

| Metric | Reference | v1.1 |
|---|---|---|
| Geomean | 0.456 ms | 0.451 ms |
| Improvement | — | 1.2% |

## MI355X Leaderboard Results

| Metric | Value |
|---|---|
| **Geomean** | **189.160 μs** |
| Delta from v1.0 | +0.064 μs improvement |
| Rank | ~65 (near dorhuri123 on board) |

Improvement over v1.0 (~204 μs) = **~8% faster** from kv_splits tuning alone.

## MXFP4 Prototype Results (v2.2 two-pass, not submitted)

| Config | Reference (ms) | MXFP4 (ms) | Speedup |
|---|---|---|---|
| bs=4, kv=1024 | 0.452 | **0.247** | **1.83x** |
| bs=4, kv=8192 | 0.426 | 1.922 | 0.22x |
| bs=64, kv=8192 | 0.462 | 9.434 | 0.05x |
| bs=256, kv=8192 | 0.815 | 39.280 | 0.02x |

Two-pass approach works for small configs but scores buffer kills large configs.
Need single-pass online softmax rewrite.
