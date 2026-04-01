# Mixed-MLA Session Handoff — Continue From Here

## Current State (2026-03-31, night)

### Leaderboard Position
- **v5.0 geomean: 70.6 μs** on MI355X (NEW PERSONAL BEST)
- Previous: v2.0 was 92.534 μs → 23.8% improvement
- Top 1 is 12.7 μs — need 5.6x more. Top 3-8 cluster at ~32 μs — need 2.2x more.
- Leaderboard username: `dorhuri123`

### Per-Config Breakdown (v5.0)

| Config | v2.0 (μs) | v5.0 (μs) | Path | Speedup |
|---|---|---|---|---|
| bs=4, kv=1024 | 49.7 | **20.9** | MXFP4 | 2.4x |
| bs=4, kv=8192 | 58.1 | **33.8** | MXFP4 | 1.7x |
| bs=32, kv=1024 | 55.7 | **31.1** | MXFP4 | 1.8x |
| bs=32, kv=8192 | 105 | 105 | AITER | same |
| bs=64, kv=1024 | 64.3 | **53.6** | MXFP4 | 1.2x |
| bs=64, kv=8192 | 154 | 154 | AITER | same |
| bs=256, kv=1024 | 107 | 107 | AITER | same |
| bs=256, kv=8192 | 302 | 303 | AITER | same |

**Bottleneck: The 4 AITER configs are unchanged. All improvement came from MXFP4 on small configs.**

### What's Deployed
- `submission.py` = v5.0 hybrid (MXFP4 for small configs + AITER cached for large)
- `submission_v5_hybrid.py` = same (backup)

### Version History

| Version | Geomean (MI355X) | Key Change | Submitted? |
|---|---|---|---|
| v1.0 | ~204 μs | Baseline | Yes |
| v1.1 | ~189 μs | Tuned kv_splits | Yes |
| v2.0 | 92.534 μs | Cache AITER metadata | Yes |
| v3.3 | ~85 μs (est.) | Hybrid Triton FP8 + AITER | Test passed |
| v4.0 | 95.9 μs | Pure MXFP4 dot_scaled | Yes (worse) |
| **v5.0** | **70.6 μs** | **Hybrid MXFP4 + AITER** | **Yes (current)** |

---

## Key Research Findings (This Session)

### 1. Leaderboard Analysis
- #1 Borui Xu: 12.7 μs (61 submissions — custom kernel, likely MXFP4 ASM)
- #2 josusanmartin: 19.8 μs (1346 submissions — brute-force tuning)
- #3-8 cluster: 32-34 μs (~400 each)
- Leaderboard mirror: leaderboard.ooousay.com

### 2. Bandwidth Floor Analysis (MI355X @ 8 TB/s)
- FP8 geomean floor: ~7.5 μs
- MXFP4 geomean floor: ~4.0 μs
- Our 70.6 μs is ~9.4x above FP8 floor — huge overhead remaining

### 3. Where Our Time Goes (AITER configs)
- Q FP8 quantization: ~20-30 μs (3 kernel launches)
- AITER ASM kernel: ~30-60 μs (depending on config)
- Metadata (cached): ~5 μs
- Python dispatch: ~5 μs

### 4. Dead Ends (Don't Retry)
- **a16w8 (bf16 Q)**: Fails correctness (max_diff=0.44 > 0.1 tolerance)
- **Fixed-scale Q quant**: Fails correctness on some seeds (max_diff up to 2.83)
- **Q cache by data_ptr**: Unsafe — leaderboard recheck=True changes Q each run
- **ASM binary patching**: ~40-50% of instructions need changing, not feasible
- **CK-based kernel**: 14-26 hours work, hdim=576 not supported, MXFP4 not wired through FMHA

### 5. ASM Kernel Insight
The "a8w8" kernel actually uses Q=FP8 × K=BF6 and P=FP8 × V=BF8 (mixed precision, not pure FP8).

---

## Optimization Paths for Next Session

### PATH 1: Fix MXFP4 Kernel for Large Configs (HIGHEST PRIORITY)
**Current problem**: v4.0 MXFP4 is 2-3x SLOWER than AITER on large configs (bs≥32, kv=8192).
**Root cause**: Each v_chunk (4 chunks of 128 dims) redundantly recomputes ALL QK scores. For bs=256,kv=8192 that's 4x redundant work on 2M tokens.
**Fix**: Accumulate all 512 V dims in a single program (like v3.3's FP8 Triton kernel uses 8 separate accumulators). This eliminates the 3D grid v_chunk dimension.
**Expected impact**: If MXFP4 beats AITER on ALL configs → geomean could drop to ~30-40 μs.
**GPU needed**: MI355X (popcorn test) only — can't test MXFP4 on MI300X.

### PATH 2: Fused Q Quantization Triton Kernel (MEDIUM PRIORITY)
**Current**: 3 separate PyTorch ops (amax, div+clamp, cast) = 3 kernel launches = ~20-30 μs
**Fix**: Single Triton kernel that does amax reduction + scale + FP8 cast in one launch
**Expected impact**: Saves 5-8 μs per AITER call → geomean ~65-67 μs
**GPU needed**: MI300X works for testing.

### PATH 3: num_kv_splits=1 Fast Path (LOW EFFORT)
AITER skips reduce kernel when num_kv_splits=1. For bs=4,kv=1024, 1 split may suffice.
**Expected impact**: ~5 μs per small config
**GPU needed**: MI300X for testing, MI355X for validation.

### PATH 4: Optimize MXFP4 V Accumulation (MEDIUM PRIORITY)
Current: V loaded from kv_data["bf16"] (576 bytes/token).
Alternative: Dequant V from MXFP4 inline (288 + 18 bytes/token = 53% less bandwidth).
**Expected impact**: 1.5-2x on V-load-dominated configs
**GPU needed**: MI355X only.

### PATH 5: HIP Graphs (HIGH RISK)
Capture Q_quant + AITER kernel + reduce into single graph replay.
ROCm has known bugs. May not work.
**GPU needed**: MI300X for testing.

---

## Files Layout
```
mixed-mla/
  submission.py              ← v5.0 hybrid (current leaderboard)
  submission_v4_mxfp4.py     ← pure MXFP4 (tested, 95.9 μs)
  submission_v5_hybrid.py    ← v5.0 backup
  triton_fp8_decode.py       ← standalone Triton FP8 kernel
  triton_mxfp4_decode.py     ← standalone MXFP4 kernel (with sm_scale fix)
  triton_mxfp4_dotscaled.py  ← early WIP (superseded)
  asm_dumps/                 ← disassembled ASM, CSV configs, codegen.py, aiter_mla.py
  versions/                  ← archived submissions
```

## Submit Commands
```bash
export POPCORN_API_URL="https://site--bot--dxfjds728w5v.code.run"
# Test (4/hour):
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui submission.py
# Leaderboard (1/hour):
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py
```

## Rate Limits
- Test: 4 per hour
- Leaderboard: 1 per hour
- Last leaderboard submit: v5.0 at ~20:30 IST March 31
