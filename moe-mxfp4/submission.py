"""
moe-mxfp4 — custom_kernel submission
=======================================
DeepSeek-R1 MXFP4 Mixture-of-Experts fused kernel, AMD MI355X.

Interface (from task.py) — 12-element tuple:
  data = (
    hidden_states,                # [M, d_hidden]                           bf16
    gate_up_weight,               # [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (raw)
    down_weight,                  # [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (raw)
    gate_up_weight_scale,         # [E, 2*d_expert_pad, d_hidden_pad//32]   e8m0   (raw)
    down_weight_scale,            # [E, d_hidden_pad, d_expert_pad//32]     e8m0   (raw)
    gate_up_weight_shuffled,      # [E, 2*d_expert_pad, d_hidden_pad//2]    fp4x2  (pre-shuffled for CK)
    down_weight_shuffled,         # [E, d_hidden_pad, d_expert_pad//2]      fp4x2  (pre-shuffled for CK)
    gate_up_weight_scale_shuffled,# [padded, flat]                          e8m0   (pre-shuffled)
    down_weight_scale_shuffled,   # [padded, flat]                          e8m0   (pre-shuffled)
    topk_weights,                 # [M, total_top_k]                        float32
    topk_ids,                     # [M, total_top_k]                        int32
    config,                       # dict — see below
  )
  Output: [M, d_hidden] bfloat16

config keys:
  d_hidden, d_expert, d_hidden_pad, d_expert_pad,
  n_routed_experts, n_shared_experts, n_experts_per_token, total_top_k, bs

DeepSeek-R1 MoE (EP-off benchmark config):
  E = 257 (256 routed + 1 shared), top_k = 9 (8+1), d_hidden = 7168, d_expert = 256

Kernel flow:
  (1) Quant activations: hidden_states → MXFP4 (per_1x32 dynamic)
  (2) Stage 1: gate_up GEMM (a4w4) + SwiGLU activation
  (3) Stage 2: down GEMM (a4w4)
  (4) Weighted reduction across top_k experts → output [M, d_hidden]

Reference: AITER fused_moe with QuantType.per_1x32 (MXFP4 block scaling)
"""

import torch
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 MoE forward pass using AITER fused_moe (matches reference).

    TODO optimizations to beat the reference AITER CK kernel:
      1. Fuse activation quantization into Stage 1 GEMM prologue
         → eliminates one global mem round-trip for quantized activations
      2. Fuse Stage 1 + Stage 2 into a single kernel
         → eliminates intermediate buffer write/read between stages
      3. Shared expert fusion: shared expert is always selected for all tokens
         → compute as a dense GEMM (no routing overhead) and merge with reduction
      4. Split-K for large M (EP-on, bs=1024, E=33, d_expert=2048)
         → large GEMMs per expert benefit from split-K parallelism
      5. Custom expert dispatch: with 257 experts but only 9 active per token,
         most expert slots are empty — compact dispatch reduces wasted wavefronts
    """
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

    hidden_pad      = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,     # pre-shuffled (16,16) tile-coalesced layout
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,  # MXFP4 block scaling, block_size=32
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output
