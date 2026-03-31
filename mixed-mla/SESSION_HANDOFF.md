# Mixed-MLA Session Handoff — Continue From Here

## Current State (2026-03-31, afternoon)

### Leaderboard Position
- **v2.0 geomean: 92.534 μs** on MI355X (current best on leaderboard)
- **v3.3 ready to submit**: Hybrid Triton+AITER, estimated ~85 μs
- Top 1 is 12.7 μs — we need another ~7x to compete for top spots
- Leaderboard username: `dorhuri123`

### What's Ready to Submit
- `submission.py` = v3.3 (Hybrid: Triton for bs=4,kv=1024 + AITER cached for rest)
- **Passed all 4 MI355X tests** (validated via popcorn test mode)
- **NOT yet scored on leaderboard** (was rate limited — submit ASAP)

### Version History

| Version | Geomean (MI355X) | Key Change | Submitted? |
|---|---|---|---|
| v1.0 | ~204 μs | Baseline reference clone | Yes |
| v1.1 | ~189 μs | Tuned kv_splits per config | Yes |
| v2.0 | **92.534 μs** | Cache AITER metadata (~2x) | **Yes (current best)** |
| v3.3 | ~85 μs (est.) | Hybrid Triton+AITER | **Ready, not submitted** |

### Files Layout
```
mixed-mla/
  submission.py          ← v3.3 hybrid (ready to submit)
  triton_fp8_decode.py   ← standalone Triton FP8 kernel (also inlined in submission.py)
  triton_mla_mxfp4.py    ← MXFP4 Triton kernel v3.2 (correct but slow, manual dequant)
  triton_mxfp4_dotscaled.py ← WIP tl.dot_scaled MXFP4 kernel (placeholder)
  reference.py           ← official reference
  task.py, task.yml      ← official task spec
  benchmark.py           ← local ref vs submission benchmark
  versions/              ← archived submissions with results
```

---

## GPU Setup (Hot Aisle MI300X)

### Provision
1. Go to [hotaisle.xyz](https://hotaisle.xyz), login (email: Dorhuri123@gmail.com)
2. Provision a Virtual Machine (1x MI300X, ~$1.99/hr)
3. SSH: `ssh hotaisle@<IP> -i ~/.ssh/id_ed25519`

### Setup Script (8 min on fresh VM)
```bash
# ── REPO IS PRIVATE — must use PAT to clone ──
git clone https://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@github.com/dorhuri123/AMD_optimization_challenge.git
cd AMD_optimization_challenge && git checkout mixed-mla-dev

# ── PyTorch + deps ──
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2 --no-cache-dir
pip install ninja cmake triton einops numpy tqdm rich psutil pybind11 pandas

# ── Reference kernels ──
git clone https://github.com/gpu-mode/reference-kernels ~/reference-kernels
cp ~/reference-kernels/problems/amd_202602/eval.py .
cp ~/reference-kernels/problems/amd_202602/utils.py .
cp ~/reference-kernels/problems/amd_202602/mixed-mla/reference.py mixed-mla/
cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.py mixed-mla/
cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.yml mixed-mla/

# ── AITER ──
cd ~ && git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && export PATH=$HOME/.local/bin:$PATH && python3 setup.py develop

# ── Environment ──
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$HOME/aiter:$HOME/AMD_optimization_challenge:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# ── Verify ──
cd ~/AMD_optimization_challenge/mixed-mla
python3 -u benchmark.py   # first run: ~5 min for AITER JIT, then fast
```

### Submit (from ANY machine — no GPU needed)
```bash
export PATH=$HOME/.local/bin:$PATH
export POPCORN_API_URL="https://site--bot--dxfjds728w5v.code.run"
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui submission.py
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py
```

### Important Notes
- **Repo is PRIVATE** — use PAT in clone URL (see setup script above)
- **RunPod MI300X is BROKEN** — SR-IOV virtualization causes hipInit() hang. Don't use.
- **Hot Aisle host ROCm is 7.2** — must use `torch +rocm7.2`
- **First AITER run** takes ~5 min for JIT compilation. Cached in `~/aiter/aiter/jit/build/`
- **Rate limits**: 1 leaderboard submission/hour, 4 test submissions/hour

---

## Key Profiling Results

### MI300X Performance (v3.3 hybrid)
```
Config         | Reference | v3.3 Sub | Speedup
bs=4, kv=1024  | 0.408ms   | 0.063ms  | 6.48x  (Triton path)
bs=4, kv=8192  | 0.406ms   | 0.282ms  | 1.44x  (AITER cached)
bs=32, kv=1024 | 0.410ms   | 0.272ms  | 1.51x
bs=32, kv=8192 | 0.413ms   | 0.274ms  | 1.51x
bs=64, kv=1024 | 0.387ms   | 0.262ms  | 1.47x
bs=64, kv=8192 | 0.468ms   | 0.273ms  | 1.71x
bs=256, kv=1024| 0.490ms   | 0.270ms  | 1.81x
bs=256, kv=8192| 0.817ms   | 0.544ms  | 1.50x
geomean        | 0.461ms   | 0.247ms  | 1.86x
```

### MI355X v2.0 per-config (leaderboard confirmed)
```
bs=4, kv=1024:   49.7 μs
bs=4, kv=8192:   58.1 μs
bs=32, kv=1024:  55.7 μs
bs=32, kv=8192:  105 μs
bs=64, kv=1024:  64.3 μs
bs=64, kv=8192:  154 μs
bs=256, kv=1024: 107 μs
bs=256, kv=8192: 302 μs
geomean: 92.534 μs
```

---

## Critical Findings This Session

### 1. Q FP8 Caching by data_ptr is UNSAFE
The leaderboard eval uses `recheck=True` which changes the seed and regenerates data each timing run. CUDA memory allocator can reuse addresses for different data, causing stale cache hits → wrong results. **Do NOT cache Q by data_ptr.**

### 2. Fixed-scale Q Quantization is UNSAFE
Using a fixed amax (e.g., 7.0) instead of dynamic per-tensor amax causes max_diff up to 2.83 on some seeds. The AITER ASM kernel is sensitive to Q quantization precision. **Always use dynamic per-tensor quantization.**

### 3. Triton vs AITER per-config (MI300X)
Triton only wins on bs=4,kv=1024 (4.4x). ALL other configs: AITER ASM wins by 1.5-44x. The AITER kernel is unbeatable for large configs.

### 4. tl.dot_scaled API (from AITER fav3_sage_attention_mxfp4.py)
```python
# QK scores via hardware MXFP4:
scores = tl.dot_scaled(q_fp4, q_scale, "e2m1", k_fp4, k_scale, "e2m1", fast_math=True)
# K loaded TRANSPOSED: [D/2, BLOCK_N]
# Q/K scales: [rows, D//32] E8M0
# V uses regular tl.dot (not dot_scaled), V_Descale applied at end
```

### 5. Leaderboard Eval Methodology
- `recheck=True` on leaderboard: seed changes each timing run, Q/KV data regenerated
- L2 cache cleared between runs via `clear_l2_cache()`
- Stats: mean, std, min, max, err across runs
- This means NO caching optimizations work on the leaderboard (Q changes every run)

---

## Optimization Paths — Updated Priority for Top 10

### Path A: Submit v3.3 NOW (IMMEDIATE)
Just submit v3.3 as-is. Expected ~85 μs (est. from Triton savings on bs=4,kv=1024).

### Path B: tl.dot_scaled MXFP4 Kernel (HIGH priority, HIGH potential)
MI355X has hardware MXFP4 dot product — 2x throughput of FP8.
- Use `kv_data["mxfp4"]` directly (no conversion needed)
- Q needs MXFP4 quantization (~46μs via dynamic_mxfp4_quant)
- BLOCK_M=16 (query heads → perfect 16×16 MFMA tile)
- Must pad QK_DIM from 576 to 1024 (next power of 2)
- Can only test via MI355X submit (no gfx950 locally)
- **Expected: 40-60 μs if kernel is well-written**

### Path C: CK-Based Decode (MEDIUM priority)
Composable Kernel C++ templates. CK already has MLA decode templates.
- Clone: `git clone https://github.com/ROCm/composable_kernel`
- Study `include/ck_tile/ops/mla/` for decode examples
- Compile with hipcc, wrap via pybind11
- **Expected: 30-50 μs**

### Path D: ASM-Level (LOW priority, HIGHEST ceiling)
AITER's `.co` files are hand-tuned assembly. Modifying them could yield best results.
- Files: `~/aiter/hsa/gfx950/mla/mla_a8w8_*.co`
- Codegen: `~/aiter/hsa/codegen.py`
- **Expected: 15-30 μs (top 3)**

### Path E: Per-Config Routing (ONGOING)
Route each config to the fastest kernel. Current routing:
- bs=4, kv=1024 → Triton FP8 (zero overhead)
- Everything else → AITER a8w8 cached

---

## Prompt to Start Next Session

```
I'm continuing work on the AMD x GPU MODE mixed-MLA decode optimization challenge.

Current state: v3.3 at estimated ~85 μs (v2.0 was 92.534 μs). Top 1 is 12.7 μs.
Repo: github.com/dorhuri123/AMD_optimization_challenge (PRIVATE, branch: mixed-mla-dev)
Deadline: April 6, 2026.

FIRST: Submit v3.3 for leaderboard scoring:
  cd mixed-mla && popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py

Read mixed-mla/SESSION_HANDOFF.md for full context, then work on Path B (tl.dot_scaled MXFP4 kernel).

Hot Aisle MI300X SSH: ssh hotaisle@<IP> -i ~/.ssh/id_ed25519
```
