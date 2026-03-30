# Mixed-MLA Handoff — Continue From Here

**Date:** 2026-03-31
**Branch:** `mixed-mla-dev`
**Hot Aisle VM:** `ssh hotaisle@23.183.40.74 -i ~/.ssh/id_ed25519` (if still running)

## Current Best: v2.0 (~97 μs geomean on MI355X)

The submission at `mixed-mla/submission.py` (v2.1) caches AITER metadata + output allocations. This alone gave **2x improvement** over the reference (189 → ~97 μs).

### MI355X Leaderboard Results (v2.0)

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

v2.1 (with output alloc caching) was submitted — results pending.

### Leaders for reference: #1 is 12.7 μs, #2 is 23 μs

## What's Been Tried

| Approach | Result | Status |
|---|---|---|
| kv_splits tuning | 189 → 189 μs (minimal on MI355X) | Done, in submission |
| a16w8 (bf16 Q) | Always slower than a8w8 | Rejected |
| AITER metadata caching | **189 → 97 μs (2x!)** | In submission |
| Triton MXFP4 kernel (v3.2) | Correct but slow (8x redundant QK) | Working, not competitive |
| Triton MXFP4 v4.0 (no redundancy) | Triton can't do 2D tensor indexing | Blocked by Triton limitation |

## Key Profiling Data

On MI300X:
- **AITER metadata setup:** 0.41ms for bs=4 (98% of total!) → **caching this was the biggest win**
- **Q quantization:** 0.06ms (consistent across configs)
- **Actual AITER ASM kernel:** 0.0ms (bs=4) to 0.47ms (bs=256/kv=8192)
- For large configs, the kernel itself dominates. For small configs, overhead dominates.

## Next Optimization Paths (priority order)

### 1. Skip Q quantization entirely — try a16w8 on MI355X
On MI300X, a16w8 was slower. But MI355X might be different. Test:
```python
# In custom_kernel, try: q_input = q (bf16), q_scale = None
# The AITER kernel supports a16w8 natively
```

### 2. Cache more aggressively
- `kv_4d` view creation
- Q fp8 tensor allocation (pre-allocate, write into it)

### 3. tl.dot_scaled MXFP4 kernel for MI355X (gfx950 only)
The real game-changer. MI355X has hardware MXFP4 MFMA instructions.
- Use `tl.dot_scaled(q_block, q_scales, "e2m1", k_block, k_scales, "e2m1")`
- Batch all 16 Q heads as M=16 → fills 16×16×128 MFMA tile perfectly
- Template: `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`
- **Can't test on MI300X** — must submit via popcorn to test on MI355X

### 4. Custom Triton decode kernel (skip AITER entirely)
For small configs (bs≤32, kv≤1024), a simple Triton kernel with no persistent-mode overhead might be faster than AITER even with fp8:
- No metadata computation at all
- Direct grid launch: (batch, heads) or (batch*splits, heads)
- Simple online softmax loop

### 5. Hybrid: MXFP4 for small configs, AITER for large
If MXFP4 Triton beats AITER on small configs, use a dispatcher:
```python
if bs * kvlen < threshold:
    return mxfp4_triton_kernel(data)
else:
    return aiter_a8w8_cached(data)
```

## How to Set Up a Fresh Hot Aisle VM

See `SETUP_HOTAISLE.md` in the repo root.

Quick version:
```bash
pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/rocm7.2 --no-cache-dir && pip install ninja cmake triton einops numpy tqdm rich psutil pybind11 pandas
cd ~ && git clone https://github.com/dorhuri123/AMD_optimization_challenge.git && cd AMD_optimization_challenge && git checkout mixed-mla-dev
git clone https://github.com/gpu-mode/reference-kernels ~/reference-kernels
cp ~/reference-kernels/problems/amd_202602/eval.py . && cp ~/reference-kernels/problems/amd_202602/utils.py . && cp ~/reference-kernels/problems/amd_202602/mixed-mla/reference.py mixed-mla/ && cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.py mixed-mla/ && cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.yml mixed-mla/
cd ~ && git clone --recursive https://github.com/ROCm/aiter.git && cd aiter && export PATH=$HOME/.local/bin:$PATH && python3 setup.py develop
```

## How to Submit Locally (no GPU needed)

```bash
export PATH=$HOME/.local/bin:$PATH
export POPCORN_API_URL="https://site--bot--dxfjds728w5v.code.run"
cd ~/dor/AMD_optimization_challenge/mixed-mla
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui submission.py
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py
```

## Files

| File | Purpose |
|---|---|
| `submission.py` | **The submission file** — edit this |
| `triton_mla_mxfp4.py` | MXFP4 Triton kernel (v3.2, correct but slow) |
| `benchmark.py` | Local ref vs submission benchmark |
| `reference.py` | Official AITER baseline (don't edit) |
| `versions/` | Historical submissions with results |
| `DEEP_DIVE.md` | Educational doc on MLA/GPU concepts |
