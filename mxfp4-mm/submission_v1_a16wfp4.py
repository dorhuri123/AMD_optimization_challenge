"""
Strategy A: Use gemm_a16wfp4_preshuffle — fused bf16 quant + FP4 GEMM in one kernel.

This eliminates the separate dynamic_mxfp4_quant + e8m0_shuffle overhead (~15us).
The kernel does inline bf16->MXFP4 quantization via _mxfp4_quant_op in registers,
then uses tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1") for the GEMM.

Requires MI355X (gfx950) with FP4 matrix core support.
"""
import torch
from task import input_t, output_t

try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _HAS_A16WFP4 = True
except ImportError:
    _HAS_A16WFP4 = False


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    return gemm_a16wfp4_preshuffle(
        A, B_shuffle, B_scale_sh,
        prequant=True, dtype=torch.bfloat16,
    )
