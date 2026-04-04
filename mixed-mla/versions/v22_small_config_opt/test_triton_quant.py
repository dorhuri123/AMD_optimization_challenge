"""
Test script for v22: validates Triton MXFP4 quant against AITER's dynamic_mxfp4_quant.

Run on MI355X:
  cd mixed-mla
  python3 versions/v22_small_config_opt/test_triton_quant.py

This checks:
1. Triton MXFP4 quant produces same packed/scale outputs as AITER
2. Full kernel correctness with Triton quant vs reference
3. Performance comparison: Triton quant vs AITER quant
4. Fused single-split vs multi-split performance
"""

import torch
import triton
import sys
import os

# Add parent to path so we can import reference
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reference import generate_input, ref_kernel
from aiter.utility.fp4_utils import dynamic_mxfp4_quant


def test_quant_correctness():
    """Compare Triton MXFP4 quant against AITER's dynamic_mxfp4_quant."""
    print("=" * 60)
    print("TEST 1: Triton MXFP4 quant vs AITER dynamic_mxfp4_quant")
    print("=" * 60)

    # Import our Triton quant
    from versions.v22_small_config_opt.submission import triton_mxfp4_quant

    for bs in [4, 32, 64]:
        num_rows = bs * 16  # batch * num_heads
        q_2d = torch.randn(num_rows, 576, dtype=torch.bfloat16, device="cuda")

        # AITER reference
        aiter_packed_raw, aiter_scale_raw = dynamic_mxfp4_quant(q_2d)
        aiter_packed = aiter_packed_raw.view(torch.uint8)
        aiter_scale = aiter_scale_raw.view(torch.uint8)

        # Triton
        triton_packed, triton_scale = triton_mxfp4_quant(q_2d)

        # Compare scales
        scale_match = torch.all(triton_scale == aiter_scale).item()
        scale_diff = (triton_scale.float() - aiter_scale.float()).abs().max().item()

        # Compare packed data
        packed_match = torch.all(triton_packed == aiter_packed).item()
        packed_diff_count = (triton_packed != aiter_packed).sum().item()
        total_bytes = triton_packed.numel()

        status = "PASS" if scale_match and packed_match else "FAIL"
        print(f"  bs={bs:>3} (rows={num_rows}): [{status}]")
        print(f"    Scale: exact_match={scale_match}, max_diff={scale_diff}")
        print(f"    Packed: exact_match={packed_match}, "
              f"diff_bytes={packed_diff_count}/{total_bytes} "
              f"({packed_diff_count/total_bytes*100:.1f}%)")

        if not packed_match:
            # Show first few mismatches
            mismatch_idx = (triton_packed != aiter_packed).nonzero()[:5]
            for idx in mismatch_idx:
                r, c = idx[0].item(), idx[1].item()
                print(f"    Mismatch at [{r},{c}]: "
                      f"triton={triton_packed[r,c].item():3d} "
                      f"aiter={aiter_packed[r,c].item():3d}")

    print()


def test_full_correctness():
    """Test full kernel with both quant modes."""
    print("=" * 60)
    print("TEST 2: Full kernel correctness")
    print("=" * 60)

    # Test with Triton quant
    import importlib
    import versions.v22_small_config_opt.submission as sub_mod

    for use_triton in [True, False]:
        sub_mod.USE_TRITON_QUANT = use_triton
        # Clear caches
        sub_mod._mxfp4_buf_cache.clear()
        sub_mod._mxfp4_q_cache.clear()

        quant_name = "Triton" if use_triton else "AITER"
        print(f"\n  Using {quant_name} Q quant:")

        configs = [
            {"batchsize": 4, "qseqlen": 1, "kvseqlen": 1024, "seed": 4220},
            {"batchsize": 32, "qseqlen": 1, "kvseqlen": 1024, "seed": 5412},
            {"batchsize": 64, "qseqlen": 1, "kvseqlen": 8192, "seed": 1360},
        ]

        for cfg in configs:
            data = generate_input(**cfg)
            ref_out = ref_kernel(data)
            sub_out = sub_mod.custom_kernel(data)
            max_diff = (ref_out - sub_out).abs().max().item()
            ok = torch.allclose(ref_out, sub_out, rtol=2e-02, atol=8e-03)
            status = "PASS" if ok else "FAIL"
            print(f"    bs={cfg['batchsize']:>3} kv={cfg['kvseqlen']:>5}: "
                  f"max_diff={max_diff:.2e} [{status}]")

    print()


def test_quant_performance():
    """Benchmark Triton quant vs AITER quant."""
    print("=" * 60)
    print("TEST 3: Q quantization performance")
    print("=" * 60)

    from versions.v22_small_config_opt.submission import triton_mxfp4_quant

    for bs in [4, 32, 64]:
        num_rows = bs * 16
        q_2d = torch.randn(num_rows, 576, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(3):
            triton_mxfp4_quant(q_2d)
            dynamic_mxfp4_quant(q_2d)

        t_triton = triton.testing.do_bench(lambda: triton_mxfp4_quant(q_2d), warmup=10, rep=50) * 1000
        t_aiter = triton.testing.do_bench(lambda: dynamic_mxfp4_quant(q_2d), warmup=10, rep=50) * 1000

        print(f"  bs={bs:>3}: Triton={t_triton:.1f}μs, AITER={t_aiter:.1f}μs, "
              f"speedup={t_aiter/t_triton:.2f}x")

    print()


def test_full_performance():
    """Benchmark full kernel with different modes."""
    print("=" * 60)
    print("TEST 4: Full kernel performance")
    print("=" * 60)

    import versions.v22_small_config_opt.submission as sub_mod

    BENCHMARKS = [
        {"batchsize": 4, "qseqlen": 1, "kvseqlen": 1024, "seed": 4217},
        {"batchsize": 4, "qseqlen": 1, "kvseqlen": 8192, "seed": 4220},
        {"batchsize": 32, "qseqlen": 1, "kvseqlen": 1024, "seed": 5412},
        {"batchsize": 32, "qseqlen": 1, "kvseqlen": 8192, "seed": 5415},
        {"batchsize": 64, "qseqlen": 1, "kvseqlen": 1024, "seed": 1357},
        {"batchsize": 64, "qseqlen": 1, "kvseqlen": 8192, "seed": 1360},
        {"batchsize": 256, "qseqlen": 1, "kvseqlen": 1024, "seed": 9823},
        {"batchsize": 256, "qseqlen": 1, "kvseqlen": 8192, "seed": 9826},
    ]

    for use_triton in [True, False]:
        sub_mod.USE_TRITON_QUANT = use_triton
        sub_mod._mxfp4_buf_cache.clear()
        sub_mod._mxfp4_q_cache.clear()

        quant_name = "Triton" if use_triton else "AITER"
        print(f"\n  {quant_name} Q quant:")
        print(f"  {'bs':>5} {'kv':>7} {'Time(μs)':>10} {'Path':>12}")

        import math
        times = []
        for cfg in BENCHMARKS:
            data = generate_input(**cfg)
            t = triton.testing.do_bench(
                lambda: sub_mod.custom_kernel(data), warmup=15, rep=60) * 1000
            times.append(t)
            bs, kv = cfg["batchsize"], cfg["kvseqlen"]
            path = "fused" if (bs, kv) in sub_mod.MXFP4_SINGLE_SPLIT_CONFIGS else (
                "mxfp4" if (bs, kv) in sub_mod.MXFP4_CONFIGS else "aiter")
            print(f"  {bs:>5} {kv:>7} {t:>10.1f} {path:>12}")

        geo = math.exp(sum(math.log(t) for t in times) / len(times))
        print(f"  Geomean: {geo:.1f}μs")

    print()


if __name__ == "__main__":
    test_quant_correctness()
    test_full_correctness()
    test_quant_performance()
    test_full_performance()
