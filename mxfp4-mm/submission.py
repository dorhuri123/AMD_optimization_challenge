"""
mxfp4-mm — custom_kernel submission
======================================
FP4 quant + FP4 GEMM: bf16 A, MXFP4 B -> bf16 C, on AMD MI355X.

Interface (from task.py):
  data = (A, B, B_q, B_shuffle, B_scale_sh)
    A          : [m, k]              bfloat16  — activation matrix (bf16, to be quantized)
    B          : [n, k]              bfloat16  — original weight (bf16, for reference only)
    B_q        : [n, k//2]           fp4x2     — B quantized to MXFP4 (raw, not shuffled)
    B_shuffle  : [n, k//2]           fp4x2     — B quantized + shuffle_weight((16,16))
    B_scale_sh : [...]               e8m0      — block scales, shuffled (from quant(B, shuffle=True))
  Output: C [m, n] bfloat16

Constraints:
  - M, N divisible by 64
  - K divisible by 64  (scale group 32 × fp4 pack 2)

Task:
  1. Quantize A to MXFP4 (per_1x32 dynamic quantization)
  2. Run gemm_a4w4 using pre-shuffled B and scales

Reference performance: aiter.gemm_a4w4 with bpreshuffle=True

Optimization opportunities:
  - Fuse A quantization into the GEMM prologue (save one global mem round-trip)
  - Custom Triton kernel targeting MI355X matrix cores natively
  - Tune tile sizes / pipeline depth for MI355X vs MI300X
"""

import torch
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils
from task import input_t, output_t


def _quant_mxfp4(x: torch.Tensor, shuffle: bool = True):
    """Dynamic MXFP4 quantization (per_1x32 block scaling).

    Returns (x_fp4x2, scale_e8m0) — same as aiter.get_torch_quant(per_1x32).
    """
    torch_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    x_q, x_scale = torch_quant(x, quant_dtype=dtypes.fp4x2)
    if shuffle:
        x_q   = shuffle_weight(x_q, layout=(16, 16))
        x_scale = fp4_utils.e8m0_shuffle(x_scale)
    return x_q, x_scale


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 GEMM: quantize A to MXFP4, run gemm_a4w4 with pre-shuffled B.

    TODO optimizations to beat the reference:
      1. Fuse A quantization into the GEMM kernel prologue — eliminates one
         global memory write + read for the quantized A tensor
      2. Write a custom Triton kernel using MI355X native FP4 matrix core
         instructions (tl.dot with fp4 dtypes once supported in ROCm Triton)
      3. Tune tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) and pipeline depth for
         MI355X — different from MI300X due to larger L2 cache and HBM3e BW
      4. Stream-K decomposition for better load balance on non-square shapes
    """
    A, B, B_q, B_shuffle, B_scale_sh = data

    A = A.contiguous()

    # Quantize A to MXFP4 (per_1x32 block, shuffled for CK kernel)
    A_q, A_scale = _quant_mxfp4(A, shuffle=True)

    # A4W4 GEMM: both activations and weights in MXFP4
    # bpreshuffle=True → B is already in shuffled tile-coalesced layout
    C = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        xscale=A_scale,
        wscale=B_scale_sh,
        bpreshuffle=True,
        out_dtype=torch.bfloat16,
    )

    return C
