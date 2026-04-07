"""
MXFP4 MoE submission v3: DSv3 config + Swiglu activation hybrid.

Combines:
1. DSv3 FP4 tuned config (FlyDSL kernels for E=257 shapes)
2. Swiglu activation (cktile bf16 path for E=33 shapes)

The Swiglu activation path uses cktile which supports split_k and bf16 activations,
avoiding the expensive MXFP4 dynamic quantization kernel.
"""

import os
import sys

# Set DSv3 config before importing aiter
def _set_dsv3_config():
    try:
        import aiter
        aiter_root = os.path.dirname(os.path.dirname(aiter.__file__))
        dsv3_path = os.path.join(aiter_root, "aiter", "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv")
        if os.path.exists(dsv3_path):
            os.environ["AITER_CONFIG_FMOE"] = dsv3_path
            return dsv3_path
        for candidate in [
            os.path.join(os.path.dirname(aiter.__file__), "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv"),
            os.path.expanduser("~/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"),
        ]:
            if os.path.exists(candidate):
                os.environ["AITER_CONFIG_FMOE"] = candidate
                return candidate
    except Exception:
        pass
    return None

_dsv3_path = _set_dsv3_config()

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_SILU = ActivationType.Silu
_SWIGLU = ActivationType.Swiglu
_PER1X32 = QuantType.per_1x32


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _, _, _, _,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    n_routed = config["n_routed_experts"]
    n_shared = config["n_shared_experts"]
    E = n_routed + n_shared

    # For E=257 (DSv3 config has FlyDSL tuned entries): use Silu (fp4x2 path)
    # For E=33 (no DSv3 entries): try Swiglu (cktile bf16 path, skip quant)
    if E > 100:
        activation = _SILU
    else:
        activation = _SWIGLU

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        activation=activation,
        quant_type=_PER1X32,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=config["d_hidden_pad"] - config["d_hidden"],
        intermediate_pad=config["d_expert_pad"] - config["d_expert"],
    )
