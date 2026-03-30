"""
mxfp4-mm — optimized custom_kernel submission
Uses fused_dynamic_mxfp4_quant which fuses shuffle into the quant kernel.
"""

import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Quantize A to MXFP4 + shuffle scales
    A_fp4, A_scale_raw = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale_raw)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale = A_scale_sh.view(dtypes.fp8_e8m0)

    # A4W4 GEMM
    C = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale,
        B_scale_sh,
        bpreshuffle=True,
        dtype=torch.bfloat16,
    )

    return C
