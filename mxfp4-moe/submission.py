import torch
from typing import Dict, Optional
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def _get_optimal_block_m(M: int, n_experts: int, top_k: int, d_expert: int) -> Optional[int]:
    """Select block_size_M based on problem shape for better wave utilization."""
    tokens_per_expert = (M * top_k) // n_experts
    if tokens_per_expert <= 4:
        return 32
    elif tokens_per_expert <= 16:
        return 32
    elif tokens_per_expert <= 64:
        return 64
    else:
        return 128


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
    n_shared = config["n_shared_experts"]
    n_routed = config["n_routed_experts"]
    total_top_k = config["total_top_k"]
    routed_top_k = config["n_experts_per_token"]
    d_expert = config["d_expert"]
    M = hidden_states.shape[0]
    E_total = n_routed + n_shared

    block_m = _get_optimal_block_m(M, E_total, total_top_k, d_expert)

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
        block_size_M=block_m,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output
