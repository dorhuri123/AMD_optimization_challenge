"""
mixed-mla — benchmark: reference vs submission
Run: python benchmark.py
Requires: reference.py and task.py (run setup.sh to fetch from gpu-mode/reference-kernels)
"""

import math
import torch
import triton
from reference import generate_input, ref_kernel
from submission import custom_kernel

# 8 configs from task.yml — leaderboard ranks by geometric mean across these
CONFIGS = [
    {"batchsize":   4, "qseqlen": 1, "kvseqlen":  512, "seed": 0},
    {"batchsize":   4, "qseqlen": 1, "kvseqlen": 2048, "seed": 0},
    {"batchsize":  16, "qseqlen": 1, "kvseqlen":  512, "seed": 0},
    {"batchsize":  16, "qseqlen": 1, "kvseqlen": 2048, "seed": 0},
    {"batchsize":  64, "qseqlen": 1, "kvseqlen":  512, "seed": 0},
    {"batchsize":  64, "qseqlen": 1, "kvseqlen": 2048, "seed": 0},
    {"batchsize": 128, "qseqlen": 1, "kvseqlen": 4096, "seed": 0},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen": 8192, "seed": 0},
]


def bench(fn, data, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(data), warmup=warmup, rep=rep)


print(f"\n{'bs':>5} {'kv_len':>7} {'Ref (ms)':>10} {'Sub (ms)':>10} {'Speedup':>9}")
print("-" * 46)

times_ref, times_sub = [], []
for cfg in CONFIGS:
    data = generate_input(**cfg)
    t_ref = bench(ref_kernel, data)
    t_sub = bench(custom_kernel, data)
    times_ref.append(t_ref)
    times_sub.append(t_sub)
    speedup = t_ref / t_sub
    flag = " ✓" if speedup >= 1.0 else " ✗"
    print(f"{cfg['batchsize']:>5} {cfg['kvseqlen']:>7} {t_ref:>10.3f} {t_sub:>10.3f} {speedup:>8.2f}x{flag}")

geo_ref = math.exp(sum(math.log(t) for t in times_ref) / len(times_ref))
geo_sub = math.exp(sum(math.log(t) for t in times_sub) / len(times_sub))
print("-" * 46)
print(f"{'geomean':>13} {geo_ref:>10.3f} {geo_sub:>10.3f} {geo_ref/geo_sub:>8.2f}x")
print("\n(Leaderboard ranks by geometric mean latency — lower is better)")
