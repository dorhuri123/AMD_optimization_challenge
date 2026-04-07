"""
MXFP4 MoE — v10: Pre-allocated buffers + direct kernel calls.

Bypasses Python overhead in AITER's fused_moe by:
1. Pre-allocating ALL intermediate buffers (sorted_ids, sorted_weights, etc.)
2. Caching metadata/config lookups from get_2stage_cfgs
3. Calling lower-level AITER functions directly
4. Using opus sorting for faster token dispatch
"""

import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    get_2stage_cfgs,
    get_inter_dim,
    get_padded_M,
    BLOCK_SIZE_M,
)
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils
from task import input_t, output_t

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SILU = ActivationType.Silu
_PER1X32 = QuantType.per_1x32

# ---------------------------------------------------------------------------
# Per-config cache: keyed by (M, E, model_dim, inter_dim, hidden_pad, intermediate_pad)
# Stores pre-allocated buffers and cached metadata.
# ---------------------------------------------------------------------------
_CACHE = {}


class _PreallocContext:
    """Holds pre-allocated buffers and cached metadata for a specific config."""
    __slots__ = (
        "sorted_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids",
        "moe_buf", "a2", "metadata", "block_size_M", "topk", "E",
        "model_dim", "inter_dim", "isG1U1", "q_dtype_a", "token_num",
        "hidden_pad", "intermediate_pad",
    )

    def __init__(self):
        pass


def _get_or_create_context(
    token_num, topk, E, model_dim, inter_dim, isG1U1,
    hidden_pad, intermediate_pad, is_shuffled, device, dtype,
):
    """Get cached context or create a new one with pre-allocated buffers."""
    key = (token_num, E, model_dim, inter_dim, hidden_pad, intermediate_pad)
    ctx = _CACHE.get(key)
    if ctx is not None:
        return ctx

    ctx = _PreallocContext()
    ctx.token_num = token_num
    ctx.topk = topk
    ctx.E = E
    ctx.model_dim = model_dim
    ctx.inter_dim = inter_dim
    ctx.isG1U1 = isG1U1
    ctx.hidden_pad = hidden_pad
    ctx.intermediate_pad = intermediate_pad

    # Determine q_dtype_a: for Silu + per_1x32, it's fp4x2
    ctx.q_dtype_a = dtypes.fp4x2

    # Cache metadata (this does the expensive CSV lookup + config resolution)
    padded_M = get_padded_M(token_num)
    ctx.metadata = get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk, dtype,
        ctx.q_dtype_a, dtypes.fp4x2, _PER1X32,
        isG1U1, _SILU, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )

    ctx.block_size_M = ctx.metadata.block_m
    if ctx.block_size_M is not None:
        ctx.block_size_M = int(ctx.block_size_M)

    # Pre-allocate sorting buffers
    M = token_num
    max_num_tokens_padded = M * topk + E * ctx.block_size_M - topk
    max_num_m_blocks = (max_num_tokens_padded + ctx.block_size_M - 1) // ctx.block_size_M

    ctx.sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    ctx.sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    ctx.sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    ctx.num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    ctx.moe_buf = torch.empty((M, model_dim), dtype=dtype, device=device)

    # Pre-allocate a2 buffer for 2-stage path
    if not ctx.metadata.run_1stage:
        ctx.a2 = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=device,
        )
    else:
        ctx.a2 = None

    _CACHE[key] = ctx
    return ctx


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _, _, _, _,
        w1, w2,
        w1_scale, w2_scale,
        topk_weights, topk_ids, config,
    ) = data

    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    is_shuffled = getattr(w1, "is_shuffled", False)
    dtype = hidden_states.dtype
    device = hidden_states.device
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Get or create pre-allocated context
    ctx = _get_or_create_context(
        M, topk, E, model_dim, inter_dim, isG1U1,
        hidden_pad, intermediate_pad, is_shuffled, device, dtype,
    )

    metadata = ctx.metadata
    block_size_M = ctx.block_size_M

    # --- Step 1: moe_sorting with pre-allocated buffers ---
    aiter.moe_sorting_opus_fwd(
        topk_ids,
        topk_weights,
        ctx.sorted_ids,
        ctx.sorted_weights,
        ctx.sorted_expert_ids,
        ctx.num_valid_ids,
        ctx.moe_buf,
        E,
        block_size_M,
        None,   # expert_mask
        None,   # num_local_tokens
        0,      # dispatch_policy
    )

    sorted_ids = ctx.sorted_ids
    sorted_weights = ctx.sorted_weights
    sorted_expert_ids = ctx.sorted_expert_ids
    num_valid_ids = ctx.num_valid_ids
    moe_out = ctx.moe_buf

    # --- 1-stage path ---
    if metadata.run_1stage:
        return metadata.stage1(
            hidden_states, w1, w2, topk,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_out, isG1U1, block_size_M,
            q_dtype_a=ctx.q_dtype_a,
            q_dtype_w=dtypes.fp4x2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            num_local_tokens=None,
            M=M,
            device=device,
            doweight_stage1=False,
        )

    # --- 2-stage path ---
    token_num = M

    # Quantize hidden_states: fused MXFP4 quant + moe sort
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=1,
        block_size=block_size_M,
    )

    # Stage 1 GEMM
    w1_scale_view = w1_scale.view(dtypes.fp8_e8m0)

    _fuse_fp4 = getattr(metadata, 'fuse_fp4_quant', False)
    if _fuse_fp4:
        a2_out = None
    else:
        a2_out = ctx.a2

    a2 = metadata.stage1(
        a1, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        a2_out,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=w1_scale_view,
        sorted_weights=None,  # doweight_stage1 is False
    )

    # Handle fused fp4 quant output from stage1
    if _fuse_fp4 and isinstance(a2, tuple):
        a2_raw, a2_scale = a2[0], a2[1]
        _fp4_bytes = token_num * topk * (inter_dim // 2)
        a2 = (
            a2_raw.view(-1)
            .view(torch.uint8)[:_fp4_bytes]
            .view(dtypes.fp4x2)
            .reshape(token_num, topk, -1)
        )
    else:
        # Quantize stage1 output: fused MXFP4 quant + moe sort for a2
        a2_flat = a2.view(-1, inter_dim)
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2_flat,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M,
        )
        a2 = a2.view(token_num, topk, -1)

    # Stage 2 GEMM
    w2_scale_view = w2_scale.view(dtypes.fp8_e8m0)

    metadata.stage2(
        a2, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_out,
        topk,
        w2_scale=w2_scale_view,
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights,  # doweight_stage1=False, so weights go to stage2
    )

    return moe_out
