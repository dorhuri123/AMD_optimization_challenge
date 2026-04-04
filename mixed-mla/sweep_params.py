"""
Parameter sweep script for MLA decode optimization.
Tests all combinations of: page_size, kv_granularity, fast_mode, num_kv_splits, intra_batch_mode.

Run on MI300X: python3 -u sweep_params.py
"""

import torch
import triton
import math
import itertools
import traceback
from reference import generate_input, ref_kernel
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
finfo = torch.finfo(FP8_DTYPE)
SM_SCALE = 1.0 / (576 ** 0.5)
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512

def quantize_fp8(t):
    amax = t.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    return (t / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE), scale.reshape(1)


def test_config(bs, kv, page_size, kv_gran, fast_mode, num_splits, ibm):
    """Test one parameter combination. Returns (time_us, max_diff) or (None, error_msg)."""
    try:
        data = generate_input(bs, 1, kv, 42)
        q, kv_data, qo_indptr, kv_indptr, config = data
        q_fp8, q_scale = quantize_fp8(q)
        kv_fp8, kv_scale = kv_data["fp8"]

        total_kv = int(kv_indptr[-1].item())

        # Reshape KV for page_size
        if total_kv % page_size != 0:
            return None, f"total_kv={total_kv} not divisible by page_size={page_size}"

        num_pages = total_kv // page_size
        kv_4d = kv_fp8.view(num_pages, page_size, NUM_KV_HEADS, QK_HEAD_DIM)

        # Adjust kv_indptr for page_size > 1
        if page_size > 1:
            # kv_indptr counts tokens, need to convert to pages
            kv_indptr_pages = kv_indptr // page_size
            kv_last_page_len = torch.full((bs,), page_size, dtype=torch.int32, device="cuda")
        else:
            kv_indptr_pages = kv_indptr
            kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        num_pages_total = int(kv_indptr_pages[-1].item())
        kv_indices = torch.arange(num_pages_total, dtype=torch.int32, device="cuda")

        o = torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda")

        # Build metadata (persistent mode)
        info = get_mla_metadata_info_v1(
            bs, 1, NUM_HEADS, q_fp8.dtype, kv_fp8.dtype,
            is_sparse=False, fast_mode=fast_mode,
            num_kv_splits=num_splits, intra_batch_mode=ibm,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        wm, wi, wis, ri, rfm, rpm = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=page_size, kv_granularity=kv_gran,
            max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=fast_mode, max_split_per_batch=num_splits,
            intra_batch_mode=ibm,
            dtype_q=q_fp8.dtype, dtype_kv=kv_fp8.dtype,
        )

        def run():
            mla_decode_fwd(
                q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
                kv_4d, o,
                qo_indptr, kv_indptr,
                kv_indices, kv_last_page_len,
                1,
                page_size=page_size, nhead_kv=NUM_KV_HEADS,
                sm_scale=SM_SCALE, logit_cap=0.0,
                num_kv_splits=num_splits,
                q_scale=q_scale, kv_scale=kv_scale,
                intra_batch_mode=ibm,
                work_meta_data=wm, work_indptr=wi, work_info_set=wis,
                reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
            )

        # Correctness check
        run()
        torch.cuda.synchronize()
        ref = ref_kernel(data)
        max_diff = (ref - o).abs().max().item()

        if max_diff > 0.1:
            return None, f"FAIL max_diff={max_diff:.4f}"

        # Benchmark
        t = triton.testing.do_bench(run, warmup=10, rep=40) * 1000  # us
        return t, max_diff

    except Exception as e:
        return None, str(e)[:80]


# ═══════════════════════════════════════════════════════════
# PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════

configs = [
    (4, 1024), (4, 8192), (32, 1024), (32, 8192),
    (64, 1024), (64, 8192), (256, 1024), (256, 8192),
]

# Parameters to sweep
page_sizes = [1, 2]
kv_granularities = [16, 32, 64, 128]
fast_modes = [False, True]
split_options = {
    (4, 1024): [8, 16, 32],
    (4, 8192): [8, 16, 32],
    (32, 1024): [8, 16, 32],
    (32, 8192): [8, 16, 24, 32, 48],
    (64, 1024): [4, 8, 16],
    (64, 8192): [8, 16, 24, 32],
    (256, 1024): [4, 8, 16],
    (256, 8192): [8, 16, 24, 32],
}
ibm_options = [True, False]

print("=" * 100)
print("PARAMETER SWEEP — MLA DECODE OPTIMIZATION")
print("=" * 100)

# First: sweep page_size and kv_granularity (most impactful, fewest combos)
print("\n### PHASE 1: page_size × kv_granularity (fixed: fast_mode=False, splits=16, ibm=True)")
print(f"{'Config':<16} | {'PS':>3} | {'Gran':>4} | {'Time(us)':>10} | {'MaxDiff':>10} | Status")
print("-" * 70)

best_per_config = {}

for bs, kv in configs:
    best_t = float('inf')
    best_params = None
    for ps in page_sizes:
        for gran in kv_granularities:
            if gran < ps:
                continue  # kv_granularity must be >= page_size
            t, info = test_config(bs, kv, ps, gran, False, 16, True)
            status = f"{t:.0f}us" if t else info
            if t and t < best_t:
                best_t = t
                best_params = (ps, gran)
            if t:
                print(f"bs={bs:>3} kv={kv:>5}   | {ps:>3} | {gran:>4} | {t:>10.0f} | {info:>10.4f} | OK")
            else:
                print(f"bs={bs:>3} kv={kv:>5}   | {ps:>3} | {gran:>4} | {'---':>10} | {'---':>10} | {info}")
    if best_params:
        best_per_config[(bs, kv)] = {"ps": best_params[0], "gran": best_params[1], "time": best_t}
        print(f"  → BEST: ps={best_params[0]} gran={best_params[1]} = {best_t:.0f}us")

# Phase 2: sweep fast_mode and ibm with best ps/gran
print("\n### PHASE 2: fast_mode × intra_batch_mode (using best ps/gran from Phase 1)")
print(f"{'Config':<16} | {'FM':>5} | {'IBM':>5} | {'Time(us)':>10} | Status")
print("-" * 60)

for bs, kv in configs:
    if (bs, kv) not in best_per_config:
        continue
    bp = best_per_config[(bs, kv)]
    best_t = bp["time"]
    for fm in fast_modes:
        for ibm in ibm_options:
            t, info = test_config(bs, kv, bp["ps"], bp["gran"], fm, 16, ibm)
            marker = " ***" if t and t < best_t else ""
            if t:
                if t < best_t:
                    best_t = t
                    bp["fm"] = fm
                    bp["ibm"] = ibm
                    bp["time"] = t
                print(f"bs={bs:>3} kv={kv:>5}   | {fm:>5} | {ibm:>5} | {t:>10.0f} | OK{marker}")
            else:
                print(f"bs={bs:>3} kv={kv:>5}   | {fm:>5} | {ibm:>5} | {'---':>10} | {info}")

# Phase 3: sweep num_kv_splits with best params
print("\n### PHASE 3: num_kv_splits sweep (using best params from Phase 1+2)")
print(f"{'Config':<16} | {'Splits':>6} | {'Time(us)':>10} | Status")
print("-" * 50)

for bs, kv in configs:
    if (bs, kv) not in best_per_config:
        continue
    bp = best_per_config[(bs, kv)]
    ps = bp["ps"]
    gran = bp["gran"]
    fm = bp.get("fm", False)
    ibm = bp.get("ibm", True)
    best_t = bp["time"]

    for ns in split_options.get((bs, kv), [16]):
        t, info = test_config(bs, kv, ps, gran, fm, ns, ibm)
        marker = " ***" if t and t < best_t else ""
        if t:
            if t < best_t:
                best_t = t
                bp["splits"] = ns
                bp["time"] = t
            print(f"bs={bs:>3} kv={kv:>5}   | {ns:>6} | {t:>10.0f} | OK{marker}")
        else:
            print(f"bs={bs:>3} kv={kv:>5}   | {ns:>6} | {'---':>10} | {info}")

# Final summary
print("\n" + "=" * 100)
print("OPTIMAL PARAMETERS PER CONFIG")
print("=" * 100)

all_times = []
for bs, kv in configs:
    if (bs, kv) in best_per_config:
        bp = best_per_config[(bs, kv)]
        all_times.append(bp["time"])
        print(f"bs={bs:>3} kv={kv:>5}: {bp['time']:>8.0f}us  ps={bp['ps']} gran={bp['gran']} "
              f"fm={bp.get('fm', False)} ibm={bp.get('ibm', True)} splits={bp.get('splits', 16)}")
    else:
        print(f"bs={bs:>3} kv={kv:>5}: FAILED all params")

if all_times:
    geo = math.exp(sum(math.log(t) for t in all_times) / len(all_times))
    print(f"\nGeomean: {geo:.0f}us")
    print(f"\nPython dict for submission:")
    print("OPTIMAL_PARAMS = {")
    for bs, kv in configs:
        if (bs, kv) in best_per_config:
            bp = best_per_config[(bs, kv)]
            print(f"    ({bs}, {kv}): {{'ps': {bp['ps']}, 'gran': {bp['gran']}, "
                  f"'fm': {bp.get('fm', False)}, 'ibm': {bp.get('ibm', True)}, "
                  f"'splits': {bp.get('splits', 16)}}},")
    print("}")
