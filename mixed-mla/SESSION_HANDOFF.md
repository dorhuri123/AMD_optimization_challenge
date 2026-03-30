# Mixed-MLA Session Handoff — Continue From Here

## Current State (2026-03-31)

### Leaderboard Position
- **v2.0 geomean: 92.534 μs** on MI355X (was 189 μs baseline)
- **~2x improvement** from caching AITER metadata
- Top 1 is 12.7 μs — we need another ~7x to compete for top spots
- Leaderboard username: `dorhuri123`

### What's Deployed
- `submission.py` = v2.1 (metadata cache + output buffer reuse)
- Repo is **PRIVATE** — need PAT to clone on GPU machines

### Version History

| Version | Geomean (MI355X) | Key Change |
|---|---|---|
| v1.0 | ~204 μs | Baseline (reference clone) |
| v1.1 | ~189 μs | Tuned kv_splits per config |
| v2.0 | **92.534 μs** | Cache AITER metadata (~50% of time was metadata setup!) |

### Files Layout
```
mixed-mla/
  submission.py          ← current v2.1 (what gets submitted)
  triton_mla_mxfp4.py   ← MXFP4 Triton kernel v3.2 (correct but slow on MI300X)
  reference.py           ← official reference (copied from gpu-mode/reference-kernels)
  task.py, task.yml      ← official task spec
  benchmark.py           ← local ref vs submission benchmark
  scripts/sweep_kv_splits.py
  DEEP_DIVE.md           ← educational doc on MLA, quantization, GPU concepts
  SESSION_HANDOFF.md     ← this file
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
- **If changing PyTorch version**: `find ~/aiter -name "*.so" -delete && rm -rf ~/aiter/aiter/jit/build`

---

## Key Profiling Results

### AITER Overhead Breakdown (MI300X)
```
bs=4, kv=1024:   metadata=0.414ms (98%), Q_quant=0.062ms, kernel=~0ms
bs=32, kv=8192:  metadata=0.052ms (12%), Q_quant=0.063ms, kernel=0.319ms (73%)
bs=256, kv=8192: metadata=0.053ms (7%),  Q_quant=0.067ms, kernel=0.677ms (85%)
```
→ Small configs: metadata-dominated (fixed by v2.0 caching)
→ Large configs: kernel-dominated (need faster kernel)

### With Metadata Cached (MI300X)
```
bs=4, kv=1024:   0.205ms (was 0.418ms → 51% savings)
bs=256, kv=8192: 0.470ms (was 0.801ms → 41% savings)
```

### MXFP4 Triton Kernel (v3.2, MI300X — correct but slow)
```
bs=4, kv=1024:   0.254ms → 1.69x faster than ref (WINS on small config)
bs=256, kv=8192: 124.9ms → way too slow (8x redundant QK computation)
```
→ Triton can't do 2D accumulator indexing, forcing redundant QK per V-chunk
→ On MI355X with tl.dot_scaled hardware, this may be different

### a16w8 vs a8w8
a16w8 (bf16 Q + fp8 KV) is **always slower**. Eliminated.

---

## Optimization Paths — Priority Order for Top 10

### Path A: Bare Triton FP8 Decode — Skip AITER Entirely (MEDIUM effort, BIG potential)
Write a simple Triton decode kernel: load FP8 KV, dequant in registers, dot product, online softmax.
No AITER metadata, no persistent mode overhead, no work buffers.
For small configs the kernel is ~0ms but AITER overhead is ~0.05ms even cached.
- **Expected: 30-50 μs on small configs → big geomean improvement**
- **Why**: eliminates all framework overhead, direct grid launch

### Path B: tl.dot_scaled MXFP4 Kernel for MI355X (HIGH effort, HIGHEST potential)
MI355X (gfx950) has hardware `tl.dot_scaled` for MXFP4 — 2x throughput of FP8.
AITER's `fav3_sage_attention_mxfp4.py` is the template (prefill kernel using dot_scaled).
Batch 16 Q heads as M=16 → fills 16×16×128 MFMA tile perfectly.
Can't test on MI300X (no gfx950), must submit via popcorn to test.
- **Expected: 20-40 μs (top 5-10 range)**
- **Risk**: can't debug locally, submit-only iteration

### Path C: CK (Composable Kernel) Based Decode (MEDIUM effort, GOOD potential)
CK is AMD's C++ template library for GPU kernels. Not raw assembly — you compose
pre-built tile operators. CK already has MLA decode templates in:
`composable_kernel/include/ck_tile/ops/mla/`
Steps:
1. Clone CK: `git clone https://github.com/ROCm/composable_kernel`
2. Study `example/ck_tile/` for MLA examples
3. Write a CK-based decode kernel with MXFP4 KV support
4. Compile with hipcc, wrap in Python via ctypes or pybind11
- **Expected: 30-50 μs**
- **Pro**: CK generates near-ASM quality code, works on MI300X for testing
- **Con**: C++ development, longer iteration cycle

### Path D: ASM-Level Optimization (HIGH effort, HIGHEST ceiling)
AITER's core MLA kernel is hand-tuned assembly (`mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co`).
The `.co` files are in `~/aiter/hsa/gfx942/mla/` (MI300X) and `~/aiter/hsa/gfx950/mla/` (MI355X).
AITER's `hsa/codegen.py` generates ASM from CSV config tables.
Options:
1. Modify the ASM codegen configs to add MXFP4 KV support
2. Disassemble the existing `.co`, modify tile sizes / add dequant instructions
3. Write new ASM using AMD's ISA docs
- **Expected: 15-30 μs (top 3 range)**
- **Pro**: maximum performance, exactly what top entries use
- **Con**: extremely difficult, ISA-level debugging

### Path E: Hybrid Submission (LOW effort, uses all above)
Different kernel per config:
- bs≤32: bare Triton (skip AITER overhead entirely)
- bs=64+, kv≤1024: AITER cached a8w8
- bs=64+, kv=8192: AITER cached a8w8 or CK/MXFP4
- **Expected: best of all approaches per config**

---

## Research Completed

1. **MXFP4 format**: E2M1 + E8M0, fp4x2 packing, dequant verified correct
2. **AITER internals**: persistent ASM kernel, split-K, online softmax, metadata overhead
3. **Triton limitations**: no 2D indexing, no slice assignment, no fp4 type
4. **tl.dot_scaled**: gfx950 only, e2m1 format, 16×16×128 tiles, 2x FP8 throughput
5. **Leaderboard top entries**: likely custom kernel + MXFP4 + zero overhead
6. **RunPod**: broken (SR-IOV), use Hot Aisle instead
7. **AITER has NO MXFP4 decode path** — only bf16/fp8 in mla_decode_fwd

---

## Prompt to Start Next Session

```
I'm continuing work on the AMD x GPU MODE mixed-MLA decode optimization challenge.

Current state: v2.0 at 92.534 μs geomean on MI355X. Top 1 is 12.7 μs.
Repo: github.com/dorhuri123/AMD_optimization_challenge (PRIVATE, branch: mixed-mla-dev)

Read mixed-mla/SESSION_HANDOFF.md for full context — it has setup instructions,
profiling data, 5 optimization paths ranked by priority, and all research done so far.

Next priority: Path A (bare Triton FP8 decode to skip AITER overhead) and/or
Path B (tl.dot_scaled MXFP4 for MI355X hardware). Also consider Path C (CK kernel)
and Path D (ASM) for maximum performance.

Hot Aisle MI300X SSH: ssh hotaisle@<IP> -i ~/.ssh/id_ed25519
Popcorn submit locally: popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py
Deadline: April 6, 2026.
```

---

## Agent Teams (Experimental)

Agent Teams requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.json (already set).
Teams are spawned dynamically during the conversation — no pre-defined files needed.
Claude can launch sub-agents for parallel work using the Agent tool.

### Suggested Division of Work

**When starting a session**, tell Claude to spawn these in parallel:

1. **"kernel-dev"** — Implement and iterate on kernel code (Triton/CK/ASM)
2. **"benchmark-submit"** — Run benchmarks on GPU, submit via popcorn, track results
3. **"research"** — Deep-dive into AITER source, CK templates, ISA docs for new optimizations

These run as sub-agents within the same conversation — they share context and can
hand off results to each other. No separate config files needed.
