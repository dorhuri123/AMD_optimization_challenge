"""
MXFP4 MoE — v11: DSv3 FP4 tuned config with FlyDSL kernels.

Uses AITER_CONFIG_FMOE pointing to dsv3_fp4_tuned_fmoe.csv which has
FlyDSL-optimized kernel selections for E=257 shapes. For E=33, falls
through to default AITER heuristics since no E=33 entries exist.

Expected gains for E=257 shapes: 1.4-1.7x over reference.
E=33 shapes: same as reference (no tuned config).
"""

import os
import importlib.util

# --- Discover aiter root BEFORE importing aiter ---
_spec = importlib.util.find_spec("aiter")
if _spec and _spec.submodule_search_locations:
    _aiter_pkg_dir = list(_spec.submodule_search_locations)[0]
    _dsv3_cfg = os.path.join(_aiter_pkg_dir, "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv")
    if os.path.exists(_dsv3_cfg):
        os.environ["AITER_CONFIG_FMOE"] = _dsv3_cfg
elif _spec and _spec.origin:
    _aiter_pkg_dir = os.path.dirname(_spec.origin)
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
