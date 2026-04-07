"""
MXFP4 MoE — v9: opus sorting + doweight_stage1=True.

doweight_stage1=True moves routing weight multiplication from Stage 2 to Stage 1.
This may reduce Stage 2 work since the weighted reduction is pre-applied.

Note: BENCHMARKS.md says doweight_stage1=True caused "All tests FAIL" for
the old submission, but that used block_size_M override. Without override,
it may work differently.
"""

import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

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
        doweight_stage1=True,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=config["d_hidden_pad"] - config["d_hidden"],
        intermediate_pad=config["d_expert_pad"] - config["d_expert"],
    )
