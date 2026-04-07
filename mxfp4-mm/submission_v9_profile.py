"""v9: Profile/debug — print config info for each shape."""
import torch
import sys
from task import input_t, output_t

_BF16 = torch.bfloat16

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import _get_config

import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4


def custom_kernel(data):
    A, _, _, B_shuffle, B_scale_sh = data
    m, k = A.shape
    B_u8 = B_shuffle.view(torch.uint8)
    n, kp = B_u8.shape

    # Print AITER's auto-selected config
    try:
        cfg, src = _get_config(m, n, kp, shuffle=True)
        print(f"[CONFIG] m={m} n={n} k={k} kp={kp}: {cfg} (from {src})", file=sys.stderr)
    except Exception as e:
        print(f"[CONFIG] m={m} n={n} k={k}: config error: {e}", file=sys.stderr)

    # Use fused Triton for all shapes
    w = B_u8.reshape(n // 16, kp * 16)
    sc = B_scale_sh.view(torch.uint8)
    ws = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
    return gemm_a16wfp4_preshuffle(A, w, ws, prequant=True, dtype=_BF16)
