"""
MXFP4 MoE — v12: Hybrid FlyDSL/CK config + pre-allocated buffers + direct calls.

Strategy:
1. For E=257: Use DSv3 FP4 tuned config (FlyDSL kernels) via AITER_CONFIG_FMOE
2. For E=33: Call AITER 2-stage directly with pre-allocated buffers, skip Python overhead
3. Pre-allocate ALL intermediate buffers per config shape
4. Use opus sorting with pre-allocated output buffers
5. Cache metadata lookups

The key insight: the DSv3 config has FlyDSL-tuned kernels for E=257 that are
1.4-1.7x faster than the CK reference. For E=33, we minimize overhead by
pre-allocating buffers and calling lower-level APIs.
"""

import os
import importlib.util

# Discover aiter root BEFORE importing aiter
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

import functools
import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_moe_2stages,
    get_2stage_cfgs,
    get_inter_dim,
    get_padded_M,
    moe_sorting,
)
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils
from task import input_t, output_t

_SILU = ActivationType.Silu
_PER1X32 = QuantType.per_1x32

# Cache for pre-allocated buffers per (M, E) config
_CTX_CACHE = {}


class _Ctx:
    __slots__ = (
        "sorted_ids", "sorted_weights", "sorted_expert_ids",
        "num_valid_ids", "moe_buf", "a2",
        "metadata", "block_size_M", "topk", "E",
        "model_dim", "inter_dim", "token_num",
    )


def _get_ctx(M, topk, E, model_dim, inter_dim, isG1U1,
             hidden_pad, intermediate_pad, is_shuffled, device, dtype):
    key = (M, E, model_dim, inter_dim)
    ctx = _CTX_CACHE.get(key)
    if ctx is not None:
        return ctx

    ctx = _Ctx()
    ctx.token_num = M
    ctx.topk = topk
    ctx.E = E
    ctx.model_dim = model_dim
    ctx.inter_dim = inter_dim

    # Cache metadata (expensive CSV lookup)
    padded_M = get_padded_M(M)
    ctx.metadata = get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk, dtype,
        dtypes.fp4x2, dtypes.fp4x2, _PER1X32,
        isG1U1, _SILU, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )

    ctx.block_size_M = int(ctx.metadata.block_m) if ctx.metadata.block_m is not None else 32

    # Pre-allocate sorting buffers
    bm = ctx.block_size_M
    max_num_tokens_padded = M * topk + E * bm - topk
    max_num_m_blocks = (max_num_tokens_padded + bm - 1) // bm

    ctx.sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    ctx.sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    ctx.sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    ctx.num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    ctx.moe_buf = torch.empty((M, model_dim), dtype=dtype, device=device)

    # Pre-allocate a2 for 2-stage
    if not ctx.metadata.run_1stage:
        ctx.a2 = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)
    else:
        ctx.a2 = None

    _CTX_CACHE[key] = ctx
    return ctx


def _direct_2stage(hidden_states, w1, w2, w1_scale, w2_scale,
                   topk_weights, topk_ids, config, ctx):
    """Direct 2-stage call with pre-allocated buffers, bypassing fused_moe overhead."""
    M = ctx.token_num
    topk = ctx.topk
    E = ctx.E
    block_size_M = ctx.block_size_M
    inter_dim = ctx.inter_dim
    metadata = ctx.metadata

    # Sorting with pre-allocated buffers
    aiter.moe_sorting_opus_fwd(
        topk_ids, topk_weights,
        ctx.sorted_ids, ctx.sorted_weights,
        ctx.sorted_expert_ids, ctx.num_valid_ids,
        ctx.moe_buf,
        E, block_size_M,
        None, None, 0,
    )

    sorted_ids = ctx.sorted_ids
    sorted_weights = ctx.sorted_weights
    sorted_expert_ids = ctx.sorted_expert_ids
    num_valid_ids = ctx.num_valid_ids
    moe_out = ctx.moe_buf

    # Quantize activations
    if M <= 1024:
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=M,
            topk=1,
            block_size=block_size_M,
        )
    else:
        from aiter import get_hip_quant as get_quant
        quant_func = get_quant(_PER1X32)
        a1, a1_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=dtypes.fp4x2,
            num_rows=None,
        )
        a1_scale = fp4_utils.moe_mxfp4_sort(
            a1_scale,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=M,
            block_size=block_size_M,
        )

    # Stage 1
    w1_scale_view = w1_scale.view(dtypes.fp8_e8m0)

    if metadata.fuse_fp4_quant:
        a2_out = None
    else:
        a2_out = ctx.a2

    a2 = metadata.stage1(
        a1, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        a2_out, topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=w1_scale_view,
        sorted_weights=None,
    )

    # Handle stage1 output quantization
    if metadata.fuse_fp4_quant and isinstance(a2, tuple):
        a2_raw, a2_scale = a2[0], a2[1]
        _fp4_bytes = M * topk * (inter_dim // 2)
        a2 = (
            a2_raw.view(-1)
            .view(torch.uint8)[:_fp4_bytes]
            .view(dtypes.fp4x2)
            .reshape(M, topk, -1)
        )
    else:
        # Quantize intermediate
        a2_flat = a2.view(-1, inter_dim)
        if M <= 1024:
            a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
                a2_flat,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=M,
                topk=topk,
                block_size=block_size_M,
            )
        else:
            from aiter import get_hip_quant as get_quant
            quant_func = get_quant(_PER1X32)
            a2, a2_scale = quant_func(
                a2_flat,
                scale=None,
                quant_dtype=dtypes.fp4x2,
                num_rows=None,
                num_rows_factor=topk,
            )
            a2_scale = fp4_utils.moe_mxfp4_sort(
                a2_scale[:M * topk, :].view(M, topk, -1),
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=M,
                block_size=block_size_M,
            )
        a2 = a2.view(M, topk, -1)

    # Stage 2
    w2_scale_view = w2_scale.view(dtypes.fp8_e8m0)

    metadata.stage2(
        a2, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_out, topk,
        w2_scale=w2_scale_view,
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights,
    )

    return moe_out


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

    ctx = _get_ctx(
        M, topk, E, model_dim, inter_dim, isG1U1,
        hidden_pad, intermediate_pad, is_shuffled, device, dtype,
    )

    # For E=257, the DSv3 config has FlyDSL kernels.
    # For E=33, the default heuristic is used.
    # In both cases, use pre-allocated buffers and direct 2-stage calls.
    if not ctx.metadata.run_1stage:
        return _direct_2stage(
            hidden_states, w1, w2, w1_scale, w2_scale,
            topk_weights, topk_ids, config, ctx,
        )
    else:
        # 1-stage path: just use fused_moe (simpler)
        return fused_moe(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            activation=_SILU, quant_type=_PER1X32,
            w1_scale=w1_scale, w2_scale=w2_scale,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
