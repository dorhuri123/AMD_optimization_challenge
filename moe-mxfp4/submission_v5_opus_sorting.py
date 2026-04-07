"""
MXFP4 MoE submission v5: DSv3 config + opus MoE sorting.

AITER has two sorting implementations:
- moe_sorting_fwd (default)
- moe_sorting_opus_fwd (enabled by AITER_USE_OPUS_MOE_SORTING=1)

The opus sorting may have better performance for our specific shapes.
Combined with the DSv3 FP4 tuned config for optimal GEMM kernels.
"""

import os

# Enable opus MoE sorting
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

# Set DSv3 config
try:
    import aiter as _aiter_pre
    _aiter_root = os.path.dirname(os.path.dirname(_aiter_pre.__file__))
    for _candidate in [
        os.path.join(_aiter_root, "aiter", "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv"),
        os.path.join(os.path.dirname(_aiter_pre.__file__), "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv"),
        os.path.expanduser("~/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"),
    ]:
        if os.path.exists(_candidate):
            os.environ["AITER_CONFIG_FMOE"] = _candidate
            break
except Exception:
    pass

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
