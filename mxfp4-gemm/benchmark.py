"""
MXFP4 GEMM — Benchmark: reference vs solution
Run: python benchmark.py
"""

import torch
import triton
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = "cuda"
MXFP4_BLOCK_SIZE = 32

# Shapes to benchmark (M, N, K)
# Covers: large square, decode (small M), prefill (large M), LLM FFN shapes
SHAPES = [
    (1,    4096, 4096),   # decode, single token
    (8,    4096, 4096),   # decode, small batch
    (32,   4096, 4096),
    (128,  4096, 4096),
    (512,  4096, 4096),   # prefill-ish
    (2048, 4096, 4096),   # large prefill
    (4096, 4096, 4096),   # large square
    (128,  14336, 4096),  # Llama FFN up-proj shape
    (128,  4096, 14336),  # Llama FFN down-proj shape
]


def make_inputs(M, N, K):
    a       = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b_fp4   = torch.randint(0, 256, (K // 2, N), device=DEVICE, dtype=torch.uint8)
    b_scales = torch.ones(K // MXFP4_BLOCK_SIZE, N, device=DEVICE, dtype=torch.float32)
    b_full  = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)  # for reference
    return a, b_fp4, b_scales, b_full


def reference_forward(a, b_full):
    """Baseline: plain bf16 matmul (stand-in until reference.py is available)."""
    return torch.mm(a, b_full)


def solution_forward(a, b_fp4, b_scales):
    from solution import mxfp4_gemm
    return mxfp4_gemm(a, b_fp4, b_scales)


def bench(fn, *args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


def tflops(M, N, K, ms):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


print(f"\n{'Shape (M,N,K)':<26} {'Ref (ms)':>10} {'Sol (ms)':>10} {'Sol TFLOPS':>12} {'Speedup':>9}")
print("-" * 72)

for (M, N, K) in SHAPES:
    a, b_fp4, b_scales, b_full = make_inputs(M, N, K)
    try:
        t_ref = bench(reference_forward, a, b_full)
        t_sol = bench(solution_forward, a, b_fp4, b_scales)
        speedup = t_ref / t_sol
        tf = tflops(M, N, K, t_sol)
        label = f"({M},{N},{K})"
        flag = " ✓" if speedup >= 1.0 else " ✗"
        print(f"{label:<26} {t_ref:>10.3f} {t_sol:>10.3f} {tf:>11.1f}T {speedup:>8.2f}x{flag}")
    except Exception as e:
        print(f"({M},{N},{K})  ERROR: {e}")
