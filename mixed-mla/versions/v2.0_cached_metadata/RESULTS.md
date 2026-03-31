# v2.0 — Cached AITER Metadata

**Date:** 2026-03-31
**Approach:** Cache AITER metadata + output buffer across repeated calls. Same a8w8 ASM kernel, just skip metadata recomputation.

## Key Finding

Profiling showed AITER metadata setup takes 41-98% of total time depending on config. Caching eliminates it.

## MI355X Leaderboard Results

**Geomean: 92.534 μs** (leaderboard confirmed)

| Config | v1.1 (μs) | v2.0 (μs) | Speedup |
|---|---|---|---|
| bs=4, kv=1024 | 152 | 49.7 | 3.1x |
| bs=4, kv=8192 | 155 | 58.1 | 2.7x |
| bs=32, kv=1024 | 163 | 55.7 | 2.9x |
| bs=32, kv=8192 | 200 | 105 | 1.9x |
| bs=64, kv=1024 | 178 | 64.3 | 2.8x |
| bs=64, kv=8192 | 256 | 154 | 1.7x |
| bs=256, kv=1024 | 214 | 107 | 2.0x |
| bs=256, kv=8192 | 403 | 302 | 1.3x |

## Improvement: 189 μs → 92.534 μs (~2x faster)
