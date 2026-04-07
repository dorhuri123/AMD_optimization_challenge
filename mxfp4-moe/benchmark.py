"""
moe-mxfp4 — benchmark: reference vs submission
Run: python benchmark.py
Requires: reference.py and task.py (run setup.sh to fetch from gpu-mode/reference-kernels)
"""

import math
import torch
import triton
from reference import generate_input, ref_kernel
from submission import custom_kernel

# Benchmark configs from task.yml (DeepSeek-R1 shapes)
#
# EP-off: all 257 experts on 1 GPU, d_expert=256 (TP-split down)
# EP-on:  EP=8, 33 experts per GPU, d_expert=2048 (full expert width)
CONFIGS = [
    # EP-off
    {"dhidden": 7168, "dexpert":  256, "nroutedexperts": 256, "nexpertspertoken": 8, "nsharedexperts": 1, "bs":    4, "seed": 0},
    {"dhidden": 7168, "dexpert":  256, "nroutedexperts": 256, "nexpertspertoken": 8, "nsharedexperts": 1, "bs":   64, "seed": 0},
    {"dhidden": 7168, "dexpert":  256, "nroutedexperts": 256, "nexpertspertoken": 8, "nsharedexperts": 1, "bs":  256, "seed": 0},
    # EP-on
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts":  32, "nexpertspertoken": 8, "nsharedexperts": 1, "bs":   64, "seed": 0},
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts":  32, "nexpertspertoken": 8, "nsharedexperts": 1, "bs":  256, "seed": 0},
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts":  32, "nexpertspertoken": 8, "nsharedexperts": 1, "bs": 1024, "seed": 0},
]

# Reference times from README (for sanity check)
REFERENCE_TIMES_US = {
    (4,   257): 46.9,
    (64,  257): 187.7,
    (256, 257): 245.7,
    (64,  33):  220.6,
    (256, 33):  276.4,
    (1024, 33): 572.2,
}


def bench(fn, data, warmup=10, rep=50):
    return triton.testing.do_bench(lambda: fn(data), warmup=warmup, rep=rep)


print(f"\n{'bs':>6} {'E':>5} {'d_exp':>6} {'Ref (ms)':>10} {'Sub (ms)':>10} {'Speedup':>9}")
print("-" * 52)

times_ref, times_sub = [], []
for cfg in CONFIGS:
    data = generate_input(**cfg)
    E = cfg["nroutedexperts"] + cfg["nsharedexperts"]
    t_ref = bench(ref_kernel, data)
    t_sub = bench(custom_kernel, data)
    times_ref.append(t_ref)
    times_sub.append(t_sub)
    speedup = t_ref / t_sub
    flag = " ✓" if speedup >= 1.0 else " ✗"
    print(f"{cfg['bs']:>6} {E:>5} {cfg['dexpert']:>6} {t_ref:>10.3f} {t_sub:>10.3f} {speedup:>8.2f}x{flag}")

geo_ref = math.exp(sum(math.log(t) for t in times_ref) / len(times_ref))
geo_sub = math.exp(sum(math.log(t) for t in times_sub) / len(times_sub))
print("-" * 52)
print(f"{'geomean':>18} {geo_ref:>10.3f} {geo_sub:>10.3f} {geo_ref/geo_sub:>8.2f}x")
print("\n(Leaderboard ranks by geometric mean latency — lower is better)")
