"""
MXFP4 MoE — v11: Monkey-patch moe_sorting to pre-allocate buffers.

Approach: Instead of rewriting the pipeline, we monkey-patch AITER's
internal _moe_sorting_impl to reuse pre-allocated buffers. This way:
1. We keep the DSv3 CK fallback kernels for optimal GEMM tile sizes
2. We eliminate the 5 tensor allocations per moe_sorting call
3. We use opus sorting for faster dispatch
"""

import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import aiter
from aiter import dtypes
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, BLOCK_SIZE_M
from task import input_t, output_t

# Pre-cache constants
_SILU = ActivationType.Silu
_PER1X32 = QuantType.per_1x32

# Buffer cache for sorting: keyed by (M, topk, E, model_dim, block_size)
_sort_buf_cache = {}


def _patched_moe_sorting_impl(
    topk_ids, topk_weights, num_experts, model_dim,
    moebuf_dtype, block_size, expert_mask, num_local_tokens,
    dispatch_policy, use_opus,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

    key = (M, topk, num_experts, model_dim, block_size)
    bufs = _sort_buf_cache.get(key)
    if bufs is None:
        sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
        sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
        sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
        num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
        moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        _sort_buf_cache[key] = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    else:
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = bufs

    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
    fwd_fn(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf, num_experts, int(block_size),
        expert_mask, num_local_tokens, dispatch_policy,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


# Monkey-patch the internal sorting function
import aiter.fused_moe as _fmoe_module
_fmoe_module._moe_sorting_impl = _patched_moe_sorting_impl


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
