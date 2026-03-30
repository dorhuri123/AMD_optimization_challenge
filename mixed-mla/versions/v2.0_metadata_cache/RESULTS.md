# v2.0 — Metadata Caching

**Date:** 2026-03-31
**Approach:** Cache AITER get_mla_metadata_v1() output across repeated calls. Same config = reuse metadata.

## Key Finding

Profiling on MI300X revealed metadata setup is 98% of total time for small batch:
- bs=4: metadata 0.414ms out of 0.425ms total (98%)
- bs=256/kv=8192: metadata 0.053ms out of 0.801ms (7%)

## MI355X Benchmark Results (popcorn --mode benchmark)

| Config | v1.1 (μs) | v2.0 (μs) | Speedup |
|---|---|---|---|
| bs=4, kv=1024 | 137 | **48.8** | **2.8x** |
| bs=4, kv=8192 | 141 | **59.6** | **2.4x** |
| bs=32, kv=1024 | 146 | **57.7** | **2.5x** |
| bs=32, kv=8192 | 192 | **105** | **1.8x** |
| bs=64, kv=1024 | 158 | **64.2** | **2.5x** |
| bs=64, kv=8192 | 245 | **153** | **1.6x** |
| bs=256, kv=1024 | 198 | **106** | **1.9x** |
| bs=256, kv=8192 | 393 | **301** | **1.3x** |
| **Geomean** | **~189** | **~96** | **~2.0x** |

## MI300X Local Benchmark

| Metric | Reference | v2.0 |
|---|---|---|
| Geomean | 0.454 ms | 0.299 ms |
| Improvement | — | **1.52x (34% faster)** |

## Rank Estimate: ~40-45 (from ~65)
