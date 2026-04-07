"""
MXFP4 MoE — v13: Per-shape kernel selection + FlyDSL config.

Combines:
1. DSv3 FP4 tuned config for FlyDSL kernels (E=257 shapes)
2. Selective split-K for E=33 small-batch shapes via AITER_KSPLIT
3. Opus sorting for faster dispatch

For E=33 with d_expert=2048: the default heuristic gives ksplit=0 because
token*topk > expert. But forcing ksplit=2 could help for the K=7168 dimension
in the stage1 GEMM (7168 / 2 = 3584, 3584 % 256 == 0).

However, the cktile split-K path requires (ksplit > 1 AND is_shuffled), which
is true for our data. The question is whether it helps.
"""

import os
import importlib.util

# Discover aiter root BEFORE importing aiter
_spec = importlib.util.find_spec("aiter")
if _spec and _spec.submodule_search_locations:
    _aiter_pkg_dir = list(_spec.submodule_search_locations)[0]
elif _spec and _spec.origin:
    _aiter_pkg_dir = os.path.dirname(_spec.origin)
else:
    _aiter_pkg_dir = None

if _aiter_pkg_dir:
    _dsv3_cfg = os.path.join(_aiter_pkg_dir, "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv")
    if os.path.exists(_dsv3_cfg):
        os.environ["AITER_CONFIG_FMOE"] = _dsv3_cfg

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

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

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
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )
