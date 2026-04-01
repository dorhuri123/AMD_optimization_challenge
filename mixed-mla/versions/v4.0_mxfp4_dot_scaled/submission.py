"""
Optimized MLA decode submission — v4.0 (MXFP4 dot_scaled)

Uses hardware tl.dot_scaled on MI355X (gfx950) for 2x throughput.
Q quantized to MXFP4 via aiter's dynamic_mxfp4_quant.
K scores via dot_scaled("e2m1", "e2m1").
V accumulated from bf16 KV via regular tl.dot.
"""

from task import input_t, output_t
from triton_mxfp4_decode import triton_mla_decode_mxfp4


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, _qo_indptr, kv_indptr, config = data
    kv_fp4, kv_scale = kv_data["mxfp4"]
    kv_bf16 = kv_data["bf16"]
    return triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, kv_bf16, kv_indptr, config)
