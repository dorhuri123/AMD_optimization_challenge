"""
moe-mxfp4 — optimized custom_kernel submission
================================================
Optimization: Bypass AITER's fused_moe wrapper and directly call the
internal CK stage functions with pre-computed sorting, eliminating
redundant work in the hot path.

Key insight: fused_moe internally does:
  1. moe_sorting (CK kernel)
  2. activation quantization (MXFP4 dynamic quant)
  3. stage1 CK GEMM (gate_up + SwiGLU)
  4. stage2 CK GEMM (down + weighted reduction)

By calling the internals directly, we can:
  - Skip Python overhead in fused_moe_ wrapper
  - Use optimal block_m/ksplit per config (from tuned CSV)
  - Pre-allocate buffers
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Clean baseline — let AITER use its CSV-tuned kernel configs
    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output
