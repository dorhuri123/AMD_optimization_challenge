"""Lean mxfp4-mm — minimize Python overhead."""

import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from task import input_t, output_t

_FP4X2 = dtypes.fp4x2
_E8M0 = dtypes.fp8_e8m0
_BF16 = torch.bfloat16


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_scale_raw = dynamic_mxfp4_quant(A)
    return aiter.gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        e8m0_shuffle(A_scale_raw).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_BF16,
    )
