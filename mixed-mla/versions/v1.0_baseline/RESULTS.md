# v1.0 — Baseline (Reference Clone)

**Date:** 2026-03-29
**Submission:** First leaderboard entry
**Approach:** Exact copy of AITER a8w8 reference kernel (NUM_KV_SPLITS=32)

## MI355X Leaderboard Results (popcorn submit)

| Config | Latency (μs) |
|---|---|
| bs=4, kv=1024 | 152 |
| bs=4, kv=8192 | 155 |
| bs=32, kv=1024 | 163 |
| bs=32, kv=8192 | 200 |
| bs=64, kv=1024 | 178 |
| bs=64, kv=8192 | 256 |
| bs=256, kv=1024 | 214 |
| bs=256, kv=8192 | 403 |
| **Geomean** | **~204 μs** |

## MI300X Local Results

| Config | Latency (ms) |
|---|---|
| Geomean | 0.457 |

## Rank: ~62-65
