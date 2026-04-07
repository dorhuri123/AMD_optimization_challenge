"""
MXFP4 MoE submission v2: Try ActivationType.Swiglu path.

Rationale:
- Silu activation with per_1x32 quant uses fp4x2 activations (dynamic quant overhead)
- Swiglu activation with per_1x32 quant uses bf16 activations on gfx950 (M < 512)
  or fp8 activations (M >= 512), both skipping separate MXFP4 quant kernel
- This goes through cktile_moe_stage1/stage2 path which may have better performance
- cktile also supports split_k for better CU utilization on small batches

The key code path in AITER fused_moe.py:
  if quant_type == QuantType.per_1x32:
      if activation == ActivationType.Swiglu:
          if get_gfx() != "gfx950" or M < bf16_fp8_bound:
              q_dtype_a = dtypes.bf16   # skip quant entirely!
          elif M >= bf16_fp8_bound:
              q_dtype_a = dtypes.fp8    # much cheaper than fp4x2 quant
      else:
          q_dtype_a = dtypes.fp4x2      # current default (expensive quant)
"""

import os
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Pre-cache constants
_SWIGLU = ActivationType.Swiglu
_PER1X32 = QuantType.per_1x32


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _, _, _, _,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        activation=_SWIGLU,
        quant_type=_PER1X32,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=config["d_hidden_pad"] - config["d_hidden"],
        intermediate_pad=config["d_expert_pad"] - config["d_expert"],
    )
