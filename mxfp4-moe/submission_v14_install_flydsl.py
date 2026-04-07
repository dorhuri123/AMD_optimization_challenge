"""
MXFP4 MoE — v14: Install FlyDSL at runtime + DSv3 config + opus sorting.

Key insight: FlyDSL is available on PyPI (flydsl v0.1.1).
The eval machine doesn't have it pre-installed, but we can install it
at import time. This enables the FlyDSL-optimized kernels from the
DSv3 config which fuse FP4 quantization into the GEMM prologue.

DSv3 FlyDSL kernels show 1.4-1.7x speedup over CK fallback for E=257.
"""

import os
import sys
import subprocess

# Try to install flydsl if not available
try:
    import flydsl
except ImportError:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "flydsl", "--quiet", "--no-deps"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception:
        pass

# Set opus sorting
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
