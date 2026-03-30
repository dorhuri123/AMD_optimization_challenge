# Mixed-MLA Session Handoff — Continue From Here

## Current State (2026-03-31)

### Leaderboard Position
- **v2.0 geomean: ~97 μs** on MI355X (was 189 μs baseline)
- **~2x improvement** from caching AITER metadata
- Top 1 is 12.7 μs — we need another ~8x to compete for top spots

### What's Deployed
- `submission.py` = v2.1 (metadata cache + output buffer reuse)
- Leaderboard entry under username `dorhuri123` (need to change display to team name **FP32**)

### Version History

| Version | Geomean (MI355X) | Key Change |
|---|---|---|
| v1.0 | ~204 μs | Baseline (reference clone) |
| v1.1 | ~189 μs | Tuned kv_splits per config |
| v2.0 | **~97 μs** | Cache AITER metadata (~50% of time was metadata setup!) |

### Files Layout
```
mixed-mla/
  submission.py          ← current v2.1 (what gets submitted)
  triton_mla_mxfp4.py   ← MXFP4 Triton kernel v3.2 (correct but slow)
  reference.py           ← official reference (copied from gpu-mode/reference-kernels)
  task.py, task.yml      ← official task spec
  benchmark.py           ← local ref vs submission benchmark
  scripts/sweep_kv_splits.py
  DEEP_DIVE.md           ← educational doc on MLA, quantization, GPU concepts
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
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2 --no-cache-dir
pip install ninja cmake triton einops numpy tqdm rich psutil pybind11 pandas

cd ~ && git clone https://github.com/dorhuri123/AMD_optimization_challenge.git
cd AMD_optimization_challenge && git checkout mixed-mla-dev

git clone https://github.com/gpu-mode/reference-kernels ~/reference-kernels
cp ~/reference-kernels/problems/amd_202602/eval.py .
cp ~/reference-kernels/problems/amd_202602/utils.py .
cp ~/reference-kernels/problems/amd_202602/mixed-mla/reference.py mixed-mla/
cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.py mixed-mla/
cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.yml mixed-mla/

cd ~ && git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && export PATH=$HOME/.local/bin:$PATH && python3 setup.py develop

# Add to bashrc
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$HOME/aiter:$HOME/AMD_optimization_challenge:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### Test
```bash
cd ~/AMD_optimization_challenge/mixed-mla
python3 -u benchmark.py   # first run: ~5 min for AITER JIT, then fast
```

### Submit (from ANY machine — no GPU needed)
```bash
export PATH=$HOME/.local/bin:$PATH
export POPCORN_API_URL="https://site--bot--dxfjds728w5v.code.run"
cd ~/AMD_optimization_challenge/mixed-mla  # or local path
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui submission.py
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py
```

### Important Notes
- **RunPod MI300X is BROKEN** — SR-IOV virtualization causes hipInit() to hang. Don't use.
- **Hot Aisle host ROCm is 7.2** — must use `torch +rocm7.2` (not 6.x)
- **First AITER run** takes ~5 min for JIT compilation. Cached in `~/aiter/aiter/jit/build/`
- **If changing PyTorch version**: `find ~/aiter -name "*.so" -delete && rm -rf ~/aiter/aiter/jit/build`

---

## Key Profiling Results

### AITER Overhead Breakdown (MI300X)
```
bs=4, kv=1024:  metadata=0.414ms (98%), Q_quant=0.062ms, kernel=~0ms
bs=32, kv=8192: metadata=0.052ms (12%), Q_quant=0.063ms, kernel=0.319ms (73%)
bs=256, kv=8192: metadata=0.053ms (7%), Q_quant=0.067ms, kernel=0.677ms (85%)
```
→ Small configs are metadata-dominated. Large configs are kernel-dominated.

### MXFP4 Triton Kernel Results (v3.2, MI300X)
```
bs=4, kv=1024:   ref=0.428ms, mxfp4=0.254ms  → 1.69x faster (WINS)
bs=4, kv=8192:   ref=0.420ms, mxfp4=1.749ms  → 0.24x (LOSES — 8x redundant QK)
bs=256, kv=8192: ref=0.814ms, mxfp4=124.9ms  → 0.01x (way too slow)
```
→ MXFP4 dequant works correctly but Triton can't eliminate QK redundancy (2D indexing unsupported)

### a16w8 vs a8w8
a16w8 (bf16 Q + fp8 KV) is **always slower** than a8w8 (fp8 Q + fp8 KV). Eliminated.

---

## Optimization Paths — What's Left

### Path A: Further AITER Tuning (LOW effort, SMALL gains)
- Cache Q quantization result if same Q is passed twice (unlikely in real benchmark)
- Pre-allocate output buffer (already done in v2.1)
- Try `fast_mode=True` in metadata (untested)
- **Expected: 5-10% improvement → ~90 μs**

### Path B: Custom Triton MXFP4 on MI355X with tl.dot_scaled (HIGH effort, BIG potential)
MI355X (gfx950) has hardware `tl.dot_scaled` for MXFP4 — 2x throughput of FP8.
AITER's `fav3_sage_attention_mxfp4.py` is the template (prefill kernel using dot_scaled).
The adaptation to decode:
- Batch 16 Q heads as M=16 → fills 16×16×128 MFMA tile perfectly
- Use e2m1 format for K, e8m0 scales
- Can't test on MI300X (no gfx950), must submit via popcorn to test
- **Expected: 2-4x over current → ~30-50 μs (top 10 range)**

### Path C: CK (Composable Kernel) Based Kernel (MEDIUM effort, GOOD potential)
Write a C++ CK kernel using MLA decode templates.
CK has pre-tuned tile sizes for gfx942/gfx950.
Can be compiled and tested on MI300X.
- **Expected: 1.5-3x → ~40-60 μs**

### Path D: Hybrid Submission (LOW effort, MODERATE gains)
Use MXFP4 Triton for small configs (where it's faster), AITER a8w8 for large configs.
The v3.2 kernel is correct but only wins on bs=4/kv=1024 on MI300X.
On MI355X with dot_scaled it might win on more configs.
- **Expected: depends on MI355X testing**

### Path E: Completely Skip AITER — Bare Triton FP8 Decode (MEDIUM effort)
Write a simple Triton decode kernel that reads FP8 KV directly.
Skip all AITER persistent-mode overhead.
No metadata, no work buffers — just grid launch.
For small configs where the kernel is ~0 but overhead is ~0.2ms, this could be huge.
- **Expected: huge for small configs, neutral for large**

---

## Research That's Been Done

1. **MXFP4 format**: E2M1 (4-bit) + E8M0 block scales (per 32 elements), fp4x2 packing verified
2. **AITER internals**: persistent ASM kernel, split-K, online softmax, metadata overhead
3. **Triton limitations**: no 2D tensor indexing, no dynamic slice assignment, no fp4 native type
4. **tl.dot_scaled**: available on gfx950, supports e2m1 format, 16×16×128 tiles
5. **Leaderboard analysis**: top entry 12.7 μs, likely uses custom kernel with MXFP4 + zero overhead
6. **RunPod**: broken for MI300X (SR-IOV hang), don't use
7. **Hot Aisle**: bare metal MI300X works perfectly, $1.99/hr

---

## Prompt to Start Next Session

```
I'm continuing work on the AMD x GPU MODE mixed-MLA decode optimization challenge.

Current state: v2.0 submission at ~97 μs geomean on MI355X (leaderboard). Top 1 is 12.7 μs.
The 2x improvement came from caching AITER metadata. Files are in mixed-mla/ on branch mixed-mla-dev.

Read mixed-mla/SESSION_HANDOFF.md for full context, setup instructions, and optimization paths.

Next priority: implement Path B (tl.dot_scaled MXFP4 kernel for MI355X) or Path E (bare Triton FP8
decode to eliminate all AITER overhead). Both target the remaining ~8x gap to leaders.

Hot Aisle MI300X is available at: ssh hotaisle@<IP> -i ~/.ssh/id_ed25519
Popcorn submit works locally: popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py

Team name: FP32. Deadline: April 6, 2026.
```
```

---

## Team Division (if using Agent Teams)

### Agent 1: "kernel-dev" — Kernel Implementation
**Focus**: Write and iterate on Triton/CK kernels
**Tasks**:
- Implement tl.dot_scaled MXFP4 decode kernel for gfx950
- Implement bare Triton FP8 decode kernel (no AITER)
- Fix Triton limitations (accumulator patterns, slice assignment)
- Test correctness on MI300X

### Agent 2: "benchmark-submit" — Testing & Submission
**Focus**: Run benchmarks, submit to leaderboard, track results
**Tasks**:
- Run benchmark.py on Hot Aisle after each kernel change
- Submit via popcorn-cli and record results
- Compare configs, identify which need MXFP4 vs FP8 path
- Update version folders with results

### Agent 3: "research" — Optimization Research
**Focus**: Find new optimization paths
**Tasks**:
- Study AITER source code for missed optimizations
- Analyze top leaderboard entries' approach
- Research CK templates for MLA decode
- Find MI355X-specific optimizations (gfx950 ISA)
