"""
mxfp4-mm — benchmark: reference vs submission
Run: python benchmark.py
Requires: reference.py and task.py (run setup.sh to fetch from gpu-mode/reference-kernels)
"""

import math
import torch
import triton
from reference import generate_input, ref_kernel
from submission import custom_kernel

# 6 benchmark configs from task.yml — leaderboard ranks by geometric mean
# Shapes reflect real LLM weight matrix dimensions (M=tokens, N=out_features, K=in_features)
# Constraint: M,N divisible by 64; K divisible by 64
CONFIGS = [
    {"m":   4, "n": 4096, "k": 7168, "seed": 0},
    {"m":  16, "n": 4096, "k": 7168, "seed": 0},
    {"m":  64, "n": 4096, "k": 7168, "seed": 0},
    {"m": 128, "n": 4096, "k": 7168, "seed": 0},
    {"m": 256, "n": 4096, "k": 7168, "seed": 0},
    {"m": 256, "n": 7168, "k": 4096, "seed": 0},
]


def tflops(m, n, k, ms):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)


def bench(fn, data, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(data), warmup=warmup, rep=rep)


print(f"\n{'(m,n,k)':<22} {'Ref (ms)':>10} {'Sub (ms)':>10} {'TFLOPS':>8} {'Speedup':>9}")
print("-" * 64)

times_ref, times_sub = [], []
for cfg in CONFIGS:
    data = generate_input(**cfg)
    t_ref = bench(ref_kernel, data)
    t_sub = bench(custom_kernel, data)
    times_ref.append(t_ref)
    times_sub.append(t_sub)
    speedup = t_ref / t_sub
    tf = tflops(cfg["m"], cfg["n"], cfg["k"], t_sub)
    label = f"({cfg['m']},{cfg['n']},{cfg['k']})"
    flag = " ✓" if speedup >= 1.0 else " ✗"
    print(f"{label:<22} {t_ref:>10.3f} {t_sub:>10.3f} {tf:>7.0f}T {speedup:>8.2f}x{flag}")

geo_ref = math.exp(sum(math.log(t) for t in times_ref) / len(times_ref))
geo_sub = math.exp(sum(math.log(t) for t in times_sub) / len(times_sub))
print("-" * 64)
print(f"{'geomean':>22} {geo_ref:>10.3f} {geo_sub:>10.3f} {'':>8} {geo_ref/geo_sub:>8.2f}x")
print("\n(Leaderboard ranks by geometric mean latency — lower is better)")
