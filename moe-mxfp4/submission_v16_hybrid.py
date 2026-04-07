"""
MXFP4 MoE — v16: Hybrid approach - selective ksplit + opus sorting.

Key findings:
- ksplit=2 for E=33, d=512, small batch: 33% faster (skips quant, uses cktile)
- ksplit=2 for E=33, d=512, bs=512: 19% SLOWER (cktile block_m=64 suboptimal)
- ksplit=0 for E=257: DSv3 CK fallback kernels are optimal
- ksplit=0 for d=2048: cktile much slower than CK 2-stage

Strategy: Only override ksplit for specific shapes where it helps.
Don't touch block_size_M or use_nt — let AITER pick from DSv3 config.
"""

import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import functools
import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fmoe_module
from task import input_t, output_t

_SILU = ActivationType.Silu
_PER1X32 = QuantType.per_1x32

# --- Override get_ksplit: enable ksplit=2 ONLY for small-batch E=33 d<2048 ---
@functools.lru_cache(maxsize=2048)
def _selective_ksplit(token, topk, expert, inter_dim, model_dim):
    # ksplit=2 triggers cktile path that skips quantization
    # Only beneficial when:
    # 1. inter_dim < 2048 (d=512 or d=256) — cktile is fast here
    # 2. token is small enough that cktile tile sizes work well
    # 3. expert is small (E=33) — not E=257 which has DSv3 tuned configs
    est_m = token * topk // max(expert, 1)
    if inter_dim < 2048 and expert < 100 and est_m < 50:
        return 2
    return 0

_fmoe_module.get_ksplit = _selective_ksplit


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
