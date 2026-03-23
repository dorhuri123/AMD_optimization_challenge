"""
MXFP4 GEMM — Optimized Solution
==================================
Goal: implement a high-performance GEMM with MXFP4-quantized weights on MI355X.

MXFP4 format:
  - 4-bit values (e2m1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6})
  - 32 values share one FP8 (e8m0) scale
  - MI355X has native hardware matrix core support

Steps:
  1. Study the reference.py baseline and task.yml for exact interface
  2. Implement a Triton kernel that uses MI355X matrix cores natively
  3. Add double-buffering (tl.async_copy / pipeline stages) for HBM latency hiding
  4. Tune BLOCK_M, BLOCK_N, BLOCK_K, num_stages for MI355X
  5. Compare vs CK baseline in benchmark.py
"""

import torch
import triton
import triton.language as tl

# TODO: import official task interface once reference-kernels is cloned
# from task import input_t, output_t


# ── MXFP4 constants ──────────────────────────────────────────────────────────
MXFP4_BLOCK_SIZE = 32   # number of values per scale factor
# FP4 e2m1 values (unsigned, times sign bit): 0, 0.5, 1, 1.5, 2, 3, 4, 6
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


# ── Kernel ───────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # TODO: retune all configs for MI355X
        # Key insight: BLOCK_K must be a multiple of MXFP4_BLOCK_SIZE (32)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_stages=2, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _mxfp4_gemm_kernel(
    # Weight matrix B in MXFP4 (packed: 2 FP4 values per byte)
    B_fp4_ptr,    # [K/2, N] uint8 (2 FP4 values packed per byte)
    B_scale_ptr,  # [K // MXFP4_BLOCK_SIZE, N] float8 scales
    # Activation matrix A (BF16 or already dequantized)
    A_ptr,        # [M, K] bf16
    # Output
    C_ptr,        # [M, N] bf16 or fp32
    # Dims
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """MXFP4 GEMM: C = A (bf16) @ dequant(B_fp4, B_scale)

    TODO: replace the software dequantization below with native MI355X
    matrix core instructions once the Triton MI355X MXFP4 backend is verified.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k * BLOCK_K

        # Load A tile
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k_base + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k_base + offs_k)[None, :] < K),
            other=0.0,
        )

        # Load B tile (packed FP4 — 2 values per byte)
        # Each byte covers 2 consecutive K values for the same N column
        k_packed = k_base // 2 + tl.arange(0, BLOCK_K // 2)
        b_packed = tl.load(
            B_fp4_ptr + k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_packed[:, None] < K // 2) & (offs_n[None, :] < N),
            other=0,
        )
        # Unpack: low nibble = even K, high nibble = odd K
        b_even = (b_packed & 0x0F).to(tl.float32)
        b_odd  = ((b_packed >> 4) & 0x0F).to(tl.float32)
        # Interleave back to [BLOCK_K, BLOCK_N]
        # (simplified — actual layout depends on task.py's packing convention)
        b = tl.interleave(b_even, b_odd)  # placeholder; adjust to actual layout

        # Load scales for this K block
        scale_k = k_base // MXFP4_BLOCK_SIZE
        b_scales = tl.load(
            B_scale_ptr + scale_k * N + offs_n,
            mask=offs_n < N,
            other=1.0,
        )
        # Dequantize: multiply by scale (broadcast over K dim)
        b = b * b_scales[None, :].to(tl.float32)

        acc += tl.dot(a.to(tl.float32), b)

    c = acc.to(tl.bfloat16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def mxfp4_gemm(
    a: torch.Tensor,        # [M, K] bf16 activations
    b_fp4: torch.Tensor,    # [K//2, N] uint8 packed FP4 weights
    b_scales: torch.Tensor, # [K//MXFP4_BLOCK_SIZE, N] float8 or float32 scales
) -> torch.Tensor:
    M, K = a.shape
    _, N = b_scales.shape  # b_scales shape: [num_scale_groups, N]

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _mxfp4_gemm_kernel[grid](
        b_fp4, b_scales.float(), a, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_fp4.stride(0), b_fp4.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


if __name__ == "__main__":
    # Smoke test
    M, N, K = 128, 256, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    # Fake packed FP4 weights and scales
    b_fp4   = torch.randint(0, 256, (K // 2, N), device="cuda", dtype=torch.uint8)
    b_scales = torch.ones(K // MXFP4_BLOCK_SIZE, N, device="cuda", dtype=torch.float32)

    c = mxfp4_gemm(a, b_fp4, b_scales)
    print(f"a={a.shape}, b_fp4={b_fp4.shape}, b_scales={b_scales.shape}")
    print(f"c={c.shape} {c.dtype}")
    print("Smoke test passed.")
