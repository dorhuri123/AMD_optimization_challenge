# v2.0 — Cached AITER Metadata

**Date:** 2026-03-31
**Approach:** Cache get_mla_metadata_v1 output across repeated calls. Same AITER a8w8 kernel, just skip metadata recomputation.

## MI355X Leaderboard Results

| Config | Latency (μs) |
|---|---|
| bs=4, kv=1024 | 49.7 |
| bs=4, kv=8192 | 58.1 |
| bs=32, kv=1024 | 55.7 |
| bs=32, kv=8192 | 105 |
| bs=64, kv=1024 | 64.3 |
| bs=64, kv=8192 | 154 |
| bs=256, kv=1024 | 107 |
| bs=256, kv=8192 | 302 |
| **Geomean** | **~97 μs** |

## Improvement over v1.1 (189 μs): ~2x faster
## Rank estimate: ~35-40
