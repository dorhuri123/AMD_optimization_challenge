"""
mxfp4-mm — custom_kernel submission
======================================
FP4 quant + FP4 GEMM: bf16 A, MXFP4 B -> bf16 C, on AMD MI355X.
"""

import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from task import input_t, output_t


def _quant_mxfp4(x: torch.Tensor, shuffle: bool = True):
    """Dynamic MXFP4 quantization (per_1x32 block scaling)."""
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
    if shuffle:
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
    return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    A = A.contiguous()

    # Quantize A to MXFP4
    A_q, A_scale = _quant_mxfp4(A, shuffle=True)

    # A4W4 GEMM — use positional arg names matching the C++ binding
    C = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale,
        B_scale_sh,
        bpreshuffle=True,
        dtype=torch.bfloat16,
    )

    return C
