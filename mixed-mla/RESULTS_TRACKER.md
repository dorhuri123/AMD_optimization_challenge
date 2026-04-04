# Per-Config Results Tracker (MI355X Leaderboard)

All times in **μs**. Only versions with leaderboard data included.

## Best Per-Config (for hybrid routing)

| Config | Best μs | Best Version | Path |
|---|---|---|---|
| bs=4, kv=1024 | **20.9** | v12 | MXFP4 Triton (e2m1 Q + e2m1 K) |
| bs=4, kv=8192 | **33.2** | v12 | MXFP4 Triton |
| bs=32, kv=1024 | **29.5** | v12 | MXFP4 Triton |
| bs=32, kv=8192 | **55.7** | v23b | a16w8 pg2 (bf16 Q + fp8 KV, page_size=2) |
| bs=64, kv=1024 | **34.8** | v23b | a16w8 pg2 |
| bs=64, kv=8192 | **91.6** | v23b | a16w8 pg2 |
| bs=256, kv=1024 | **51.9** | v23b | a16w8 pg2 |
| bs=256, kv=8192 | **293** | v17 | a8w8 persistent (fp8 Q + fp8 KV, splits=24) |
| **Geomean (best-of)** | **~50** | hybrid | |

## Version Results

### v2.0 — AITER a8w8 cached (geomean: 92.534 μs)
| Config | μs | Path |
|---|---|---|
| bs=4, kv=1024 | 49.7 | AITER a8w8, splits=16 |
| bs=4, kv=8192 | 58.1 | AITER a8w8, splits=32 |
| bs=32, kv=1024 | 55.7 | AITER a8w8, splits=16 |
| bs=32, kv=8192 | 105 | AITER a8w8, splits=48 |
| bs=64, kv=1024 | 64.3 | AITER a8w8, splits=16 |
| bs=64, kv=8192 | 154 | AITER a8w8, splits=24 |
| bs=256, kv=1024 | 107 | AITER a8w8, splits=16 |
| bs=256, kv=8192 | 302 | AITER a8w8, splits=24 |

### v5.0 — Hybrid MXFP4 + AITER (geomean: 71.4 μs)
| Config | μs | Path |
|---|---|---|
| bs=4, kv=1024 | 20.9 | MXFP4 Triton |
| bs=4, kv=8192 | 33.8 | MXFP4 Triton |
| bs=32, kv=1024 | 31.1 | MXFP4 Triton |
| bs=32, kv=8192 | 105 | AITER a8w8 cached |
| bs=64, kv=1024 | 53.6 | MXFP4 Triton |
| bs=64, kv=8192 | 154 | AITER a8w8 cached |
| bs=256, kv=1024 | 107 | AITER a8w8 cached |
| bs=256, kv=8192 | 303 | AITER a8w8 cached |

### v12.0 — Hybrid MXFP4 + AITER bypass (geomean: 67.3 μs)
| Config | μs | Path |
|---|---|---|
| bs=4, kv=1024 | 20.9 | MXFP4 Triton |
| bs=4, kv=8192 | 33.2 | MXFP4 Triton |
| bs=32, kv=1024 | 29.5 | MXFP4 Triton |
| bs=32, kv=8192 | 79.9 | AITER bypass fresh-alloc, splits=24 |
| bs=64, kv=1024 | 51.4 | MXFP4 Triton |
| bs=64, kv=8192 | 131 | AITER bypass fresh-alloc, splits=16 |
| bs=256, kv=1024 | 107 | AITER wrapper, splits=8 |
| bs=256, kv=8192 | 302 | AITER wrapper, splits=16 |

### v17.0 — bf16-Q MXFP4 + AITER (geomean: 72.0 μs)
| Config | μs | Path |
|---|---|---|
| bs=4, kv=1024 | 22.4 | MXFP4 Triton (bf16 Q, dot_scaled) |
| bs=4, kv=8192 | 37.7 | MXFP4 Triton (bf16 Q) |
| bs=32, kv=1024 | 34.2 | MXFP4 Triton (bf16 Q) |
| bs=32, kv=8192 | 99.6 | AITER a8w8 |
| bs=64, kv=1024 | 57.8 | MXFP4 Triton (bf16 Q) |
| bs=64, kv=8192 | 147 | AITER a8w8 |
| bs=256, kv=1024 | 101 | AITER a8w8, splits=16 |
| bs=256, kv=8192 | 293 | AITER a8w8, splits=24 |

### v23b — a16w8 + PAGE_SIZE=2 all configs (geomean: 57.1 μs)
| Config | μs | Path |
|---|---|---|
| bs=4, kv=1024 | 32.0 | a16w8 pg2 (bf16 Q, fp8 KV, page_size=2) |
| bs=4, kv=8192 | 36.9 | a16w8 pg2 |
| bs=32, kv=1024 | 34.2 | a16w8 pg2 |
| bs=32, kv=8192 | 55.7 | a16w8 pg2 |
| bs=64, kv=1024 | 34.8 | a16w8 pg2 |
| bs=64, kv=8192 | 91.6 | a16w8 pg2 |
| bs=256, kv=1024 | 51.9 | a16w8 pg2 |
| bs=256, kv=8192 | 304 | a16w8 pg2 |

### v24 — 3-way hybrid (geomean: 46.0 μs) ⭐ NEW BEST
| Config | μs | Path |
|---|---|---|
| bs=4, kv=1024 | 20.9 | MXFP4 Triton |
| bs=4, kv=8192 | 33.3 | MXFP4 Triton |
| bs=32, kv=1024 | 29.7 | MXFP4 Triton |
| bs=32, kv=8192 | 55.7 | a16w8 pg2 |
| bs=64, kv=1024 | 34.3 | a16w8 pg2 |
| bs=64, kv=8192 | 91.5 | a16w8 pg2 |
| bs=256, kv=1024 | 51.9 | a16w8 pg2 |
| bs=256, kv=8192 | **107** | a8w8 pg1 splits=24 (2.8x faster than v12!) |

### v25 — v24 + PAGE_SIZE=8 for large kv (pending leaderboard)
Same as v24 but bs=256,kv=8192 uses a8w8 pg8 instead of pg1.

## Failed Versions (for reference)

| Version | Geomean | Issue |
|---|---|---|
| v4.0 MXFP4 only | 95.9 | Large configs 3-6x slower than AITER |
| v6.1 all-MXFP4 2D grid | 87.3 | Same large config regression |
| v7.0 MXFP4 V dequant | 97.98 | Manual V dequant compute > bandwidth savings |
| v15 low splits | failed | Correctness failures on kv=8192 recheck |
| v16 pure AITER | 94.0 | No MXFP4 for small configs = slow |
| v18 MXFP4+FP8V all | 213.6 | Triton MXFP4 terrible on large configs |
| v19 non-persistent | failed | Correctness failures on recheck |
| Phase 4a MFMA | 916+ | Single-wave (64 threads) = terrible occupancy |
