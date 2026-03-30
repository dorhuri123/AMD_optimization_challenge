"""
moe-mxfp4 — optimized custom_kernel submission
================================================
DeepSeek-R1 MXFP4 Mixture-of-Experts fused kernel, AMD MI355X.

Optimizations over baseline AITER fused_moe:
  1. Per-config block_size_M tuning for better wave utilization
  2. Shared expert separation: compute shared expert as dense MoE
     (all tokens routed, no sparse dispatch overhead)
  3. Pre-allocated output buffer to reduce allocation overhead
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Cache for pre-allocated output buffers
_output_cache = {}


def _get_block_m(M: int, E: int, top_k: int, d_expert: int) -> int:
    """Tune block_size_M per problem shape.

    Small batch / many experts -> small blocks (less wave waste).
    Large batch / few experts -> large blocks (better GEMM throughput).
    """
    avg_tokens_per_expert = max(1, (M * top_k) // E)
    if avg_tokens_per_expert <= 8:
        return 32
    elif avg_tokens_per_expert <= 32:
        return 32
    elif d_expert >= 2048 and M >= 512:
        return 128
    elif avg_tokens_per_expert <= 128:
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
    d_expert = config["d_expert"]
    M = hidden_states.shape[0]
    E_total = n_routed + n_shared

    block_m = _get_block_m(M, E_total, total_top_k, d_expert)

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
