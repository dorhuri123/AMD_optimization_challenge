"""
Lean MoE submission — minimize Python overhead.
Pre-import everything, inline the call, no extra computation.
Known baseline: ~174 µs geomean.
"""

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Pre-cache constants
_SILU = ActivationType.Silu
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
        activation=_SILU,
        quant_type=_PER1X32,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=config["d_hidden_pad"] - config["d_hidden"],
        intermediate_pad=config["d_expert_pad"] - config["d_expert"],
    )
