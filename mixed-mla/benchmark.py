"""
mixed-mla — benchmark: reference vs submission
Run on MI355X:  python benchmark.py
Requires: reference.py, task.py in this directory (copied by setup.sh)
"""

import math
import torch
import triton
from reference import generate_input, ref_kernel
from submission import custom_kernel

# 8 benchmark configs from task.yml (leaderboard ranks by geometric mean)
BENCHMARKS = [
    {"batchsize":   4, "qseqlen": 1, "kvseqlen":  1024, "seed": 4217},
    {"batchsize":   4, "qseqlen": 1, "kvseqlen":  8192, "seed": 4220},
    {"batchsize":  32, "qseqlen": 1, "kvseqlen":  1024, "seed": 5412},
    {"batchsize":  32, "qseqlen": 1, "kvseqlen":  8192, "seed": 5415},
    {"batchsize":  64, "qseqlen": 1, "kvseqlen":  1024, "seed": 1357},
    {"batchsize":  64, "qseqlen": 1, "kvseqlen":  8192, "seed": 1360},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen":  1024, "seed": 9823},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen":  8192, "seed": 9826},
]

# 4 correctness test configs from task.yml
TESTS = [
    {"batchsize":   4, "qseqlen": 1, "kvseqlen":  1024, "seed": 4220},
    {"batchsize":  32, "qseqlen": 1, "kvseqlen":  1024, "seed": 5412},
    {"batchsize":  64, "qseqlen": 1, "kvseqlen":  8192, "seed": 1360},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen":  8192, "seed": 9826},
]


def bench_fn(fn, data, warmup=25, rep=100):
    """Benchmark a kernel using triton's do_bench (median of rep runs)."""
    return triton.testing.do_bench(lambda: fn(data), warmup=warmup, rep=rep)


def check_correctness():
    """Validate submission against reference on all test configs."""
    print("=" * 60)
    print("CORRECTNESS CHECK (rtol=2e-02, atol=8e-03)")
    print("=" * 60)
    all_pass = True
    for cfg in TESTS:
        data = generate_input(**cfg)
        ref_out = ref_kernel(data)
        sub_out = custom_kernel(data)
        max_diff = (ref_out - sub_out).abs().max().item()
        # Use the official tolerance from the challenge README
        ok = torch.allclose(ref_out, sub_out, rtol=2e-02, atol=8e-03)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  bs={cfg['batchsize']:>3}, kv={cfg['kvseqlen']:>5}: "
              f"max_diff={max_diff:.2e}  [{status}]")
    print()
    return all_pass


def run_benchmarks():
    """Benchmark reference vs submission on all 8 configs."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"{'bs':>5} {'kv_len':>7} {'Ref (ms)':>10} {'Sub (ms)':>10} {'Speedup':>9}")
    print("-" * 46)

    times_ref, times_sub = [], []
    for cfg in BENCHMARKS:
        data = generate_input(**cfg)

        t_ref = bench_fn(ref_kernel, data)
        t_sub = bench_fn(custom_kernel, data)

        times_ref.append(t_ref)
        times_sub.append(t_sub)

        speedup = t_ref / t_sub
        flag = " +" if speedup > 1.01 else (" -" if speedup < 0.99 else " =")
        print(f"{cfg['batchsize']:>5} {cfg['kvseqlen']:>7} "
              f"{t_ref:>10.3f} {t_sub:>10.3f} {speedup:>8.2f}x{flag}")

    geo_ref = math.exp(sum(math.log(t) for t in times_ref) / len(times_ref))
    geo_sub = math.exp(sum(math.log(t) for t in times_sub) / len(times_sub))
    print("-" * 46)
    print(f"{'geomean':>13} {geo_ref:>10.3f} {geo_sub:>10.3f} "
          f"{geo_ref / geo_sub:>8.2f}x")
    print("\nLeaderboard metric: geometric mean latency (lower is better)")
    print(f"  Reference: {geo_ref:.3f} ms")
    print(f"  Ours:      {geo_sub:.3f} ms")
    if geo_sub < geo_ref:
        pct = (1 - geo_sub / geo_ref) * 100
        print(f"  => {pct:.1f}% faster than reference")
    else:
        pct = (geo_sub / geo_ref - 1) * 100
        print(f"  => {pct:.1f}% slower than reference (need optimization)")


if __name__ == "__main__":
    passed = check_correctness()
    if not passed:
        print("WARNING: Correctness check failed! Fix before submitting.\n")
    run_benchmarks()
