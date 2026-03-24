"""
Sweep num_kv_splits for each benchmark config to find optimal values.

Run on MI355X:
    cd mixed-mla && python scripts/sweep_kv_splits.py

Output: a KV_SPLITS_MAP dict you can paste directly into submission.py
"""

import sys
import math
import torch
import triton

sys.path.insert(0, ".")
from reference import generate_input, quantize_fp8
from submission import _mla_decode

# Benchmark configs from task.yml
CONFIGS = [
    (4, 1024),
    (4, 8192),
    (32, 1024),
    (32, 8192),
    (64, 1024),
    (64, 8192),
    (256, 1024),
    (256, 8192),
]

# Values to sweep
SPLITS_TO_TRY = [4, 8, 16, 24, 32, 48, 64]

WARMUP = 25
REP = 100


def main():
    print("Sweeping num_kv_splits for each benchmark config...")
    print(f"Splits to try: {SPLITS_TO_TRY}")
    print()

    results = {}

    for bs, kvlen in CONFIGS:
        print(f"--- bs={bs}, kv_len={kvlen} ---")
        data = generate_input(batchsize=bs, qseqlen=1, kvseqlen=kvlen, seed=42)
        q, kv_data, qo_indptr, kv_indptr, config = data

        # Prepare fp8 Q (same as reference)
        q_fp8, q_scale = quantize_fp8(q)
        kv_buffer_fp8, kv_scale = kv_data["fp8"]

        best_splits = 32
        best_time = float("inf")

        for splits in SPLITS_TO_TRY:
            try:
                # Warmup + benchmark
                t = triton.testing.do_bench(
                    lambda s=splits: _mla_decode(
                        q_fp8, kv_buffer_fp8, qo_indptr, kv_indptr, config,
                        q_scale=q_scale, kv_scale=kv_scale,
                        num_kv_splits=s,
                    ),
                    warmup=WARMUP, rep=REP,
                )
                marker = " <-- best" if t < best_time else ""
                print(f"  splits={splits:>3}: {t:.3f} ms{marker}")

                if t < best_time:
                    best_time = t
                    best_splits = splits
            except Exception as e:
                print(f"  splits={splits:>3}: FAILED ({e})")

        results[(bs, kvlen)] = best_splits
        print(f"  => Best: splits={best_splits} ({best_time:.3f} ms)")
        print()

    # Also sweep Q dtype (a16w8 vs a8w8) for each config
    print("\n--- Q dtype comparison (a8w8 vs a16w8) ---")
    for bs, kvlen in CONFIGS:
        data = generate_input(batchsize=bs, qseqlen=1, kvseqlen=kvlen, seed=42)
        q, kv_data, qo_indptr, kv_indptr, config = data
        q_fp8, q_scale = quantize_fp8(q)
        kv_buffer_fp8, kv_scale = kv_data["fp8"]

        splits = results[(bs, kvlen)]

        # a8w8: fp8 Q + fp8 KV
        t_a8w8 = triton.testing.do_bench(
            lambda: _mla_decode(
                q_fp8, kv_buffer_fp8, qo_indptr, kv_indptr, config,
                q_scale=q_scale, kv_scale=kv_scale, num_kv_splits=splits,
            ),
            warmup=WARMUP, rep=REP,
        )

        # a16w8: bf16 Q + fp8 KV
        try:
            t_a16w8 = triton.testing.do_bench(
                lambda: _mla_decode(
                    q, kv_buffer_fp8, qo_indptr, kv_indptr, config,
                    q_scale=None, kv_scale=kv_scale, num_kv_splits=splits,
                ),
                warmup=WARMUP, rep=REP,
            )
            winner = "a16w8" if t_a16w8 < t_a8w8 else "a8w8"
            print(f"  bs={bs:>3}, kv={kvlen:>5}: "
                  f"a8w8={t_a8w8:.3f}ms, a16w8={t_a16w8:.3f}ms => {winner}")
        except Exception as e:
            print(f"  bs={bs:>3}, kv={kvlen:>5}: "
                  f"a8w8={t_a8w8:.3f}ms, a16w8=FAILED ({e})")

    # Print pasteable dict
    print("\n" + "=" * 60)
    print("Paste this into submission.py KV_SPLITS_MAP:")
    print("=" * 60)
    print("KV_SPLITS_MAP = {")
    for (bs, kvlen), splits in sorted(results.items()):
        print(f"    ({bs}, {kvlen}): {splits},")
    print("}")


if __name__ == "__main__":
    main()
