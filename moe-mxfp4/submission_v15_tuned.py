"""
MXFP4 MoE — v15: Per-shape CK kernel injection + selective ksplit + NT control.

Based on competitor intelligence:
1. Opus sorting (~10% for E=257)
2. AITER_USE_NT=0 (disable non-temporal loads, ~2-4%)
3. Selective ksplit=2 for d_expert<2048 (skips quant, ~28% saving!)
4. Per-shape block_m tuning based on est_m = token*topk/expert
5. Force token_num_quant_moe_sort_switch higher for fused quant+sort
"""

import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"
os.environ["AITER_USE_NT"] = "0"

import functools
import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, get_block_size_M, get_ksplit, use_nt
import aiter.fused_moe as _fmoe_module
from task import input_t, output_t

# Pre-cache constants
_SILU = ActivationType.Silu
_PER1X32 = QuantType.per_1x32

# --- Override get_block_size_M with per-shape tuning ---
_orig_get_block_size_M = _fmoe_module.get_block_size_M.__wrapped__ if hasattr(_fmoe_module.get_block_size_M, '__wrapped__') else None

@functools.lru_cache(maxsize=2048)
def _tuned_block_size_M(token, topk, expert, inter_dim):
    est_m = token * topk // expert
    if expert >= 257:
        if est_m >= 15:
            return 128
        elif est_m >= 5:
            return 64
        else:
            return 32
    else:  # E<=64
        if est_m < 50:
            return 32
        elif inter_dim >= 2048 and est_m >= 100:
            return 128
        else:
            return 64

_fmoe_module.get_block_size_M = _tuned_block_size_M

# --- Override get_ksplit to enable selective split-K ---
@functools.lru_cache(maxsize=2048)
def _tuned_ksplit(token, topk, expert, inter_dim, model_dim):
    # ksplit=2 for small d_expert triggers cktile path which skips quant
    # But for d>=2048, cktile is much slower
    if inter_dim < 2048:
        return 2
    return 0

_fmoe_module.get_ksplit = _tuned_ksplit

# --- Override use_nt to always disable non-temporal loads ---
@functools.lru_cache(maxsize=2048)
def _tuned_use_nt(token, topk, e):
    return False

_fmoe_module.use_nt = _tuned_use_nt

# --- Force fused quant+sort for all token counts ---
_fmoe_module.token_num_quant_moe_sort_switch = 8192


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
