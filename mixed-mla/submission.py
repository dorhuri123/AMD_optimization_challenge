"""
Optimized MLA decode submission — v3.0

Pure Triton FP8 decode kernel — no AITER dependency.
Eliminates all AITER metadata overhead (~50μs saved on small configs).
Uses split-K flash-decoding with online softmax.
"""

import torch
from task import input_t, output_t
from triton_fp8_decode import triton_mla_decode_fp8


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp8, kv_scale = kv_data["fp8"]
    return triton_mla_decode_fp8(q, kv_fp8, kv_scale, kv_indptr, config)
