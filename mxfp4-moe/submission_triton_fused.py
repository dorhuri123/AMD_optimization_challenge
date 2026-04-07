"""
MXFP4 MoE — Custom Triton 2-stage kernel with bf16 activations.

Strategy: Use tl.dot_scaled(bf16, None, "bf16", fp4x2, e8m0, "e2m1")
to avoid MXFP4 activation quantization entirely. Uses RAW (non-shuffled)
weight layouts which are directly compatible with tl.dot_scaled on gfx950.

Stage 1: bf16_hidden x e2m1_gate_up_weights + SwiGLU -> bf16 intermediate
Stage 2: bf16_inter  x e2m1_down_weights + weighted reduction -> bf16 output

Input tuple:
  hidden_states,                  # [M, d_hidden]                     bf16
  gate_up_weight,                 # [E, 2*d_expert_pad, d_hidden_pad//2]  fp4x2 (raw)
  down_weight,                    # [E, d_hidden_pad, d_expert_pad//2]    fp4x2 (raw)
  gate_up_weight_scale,           # [E, 2*d_expert_pad, d_hidden_pad//32] e8m0  (raw)
  down_weight_scale,              # [E, d_hidden_pad, d_expert_pad//32]   e8m0  (raw)
  gate_up_weight_shuffled,        # [E, 2*d_expert_pad, d_hidden_pad//2]  fp4x2 (shuffled)
  down_weight_shuffled,           # [E, d_hidden_pad, d_expert_pad//2]    fp4x2 (shuffled)
  gate_up_weight_scale_shuffled,  # [padded, flat]                        e8m0  (shuffled)
  down_weight_scale_shuffled,     # [padded, flat]                        e8m0  (shuffled)
  topk_weights,                   # [M, 9] float32
  topk_ids,                       # [M, 9] int32
  config                          # dict
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting, fused_moe


# =====================================================================
# Triton helpers
# =====================================================================

@triton.jit
def _remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@triton.jit
def _pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


# =====================================================================
# Stage 1 Kernel: bf16 hidden x e2m1 gate_up weights + SwiGLU
#
# Weight B layout (raw): [E, N, K//2] where N = 2*d_expert_pad, K = d_hidden_pad
#   B is stored as (expert, output_dim, packed_input_dim)
#   stride_be = B.stride(0), stride_bn = B.stride(1), stride_bk = B.stride(2)
#   B_scale layout (raw): [E, N, K//32] e8m0
#   stride_bmxe = Bs.stride(0), stride_bmxn = Bs.stride(1), stride_bmxk = Bs.stride(2)
#
# For SwiGLU: N = 2*d_expert. We interleave gate and up columns so each
# BLOCK_N tile has paired gate/up elements. After GEMM, split and apply
# silu(gate) * up. Output dimension is d_expert = N//2.
# =====================================================================

@triton.jit
def _moe_stage1_kernel(
    a_ptr,              # [M, K] bf16 hidden_states
    b_ptr,              # [E, N, K//2] uint8 fp4x2 (raw, NOT shuffled)
    b_scale_ptr,        # [E, N, K//32] uint8 e8m0 (raw, NOT shuffled)
    c_ptr,              # [M*topk, N//2] bf16 output after SwiGLU
    sorted_ids_ptr,
    expert_ids_ptr,
    num_valid_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn, stride_bsk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    top_k: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    PACK: tl.constexpr = 2            # fp4x2 packing factor
    MX_BS: tl.constexpr = 32          # MXFP4 block size for scales
    PK: tl.constexpr = BLOCK_K // PACK
    SK: tl.constexpr = BLOCK_K // MX_BS
    HALF_N: tl.constexpr = BLOCK_N // 2

    pid = tl.program_id(0)
    ntp = tl.load(num_valid_ptr)
    npm = tl.cdiv(ntp, BLOCK_M)
    npn = tl.cdiv(N, BLOCK_N)
    GRID = npm * npn
    if pid >= GRID:
        return
    pid = _remap_xcd(pid, GRID, 8)
    pm, pn = _pid_grid(pid, npm, npn, GROUP_M)

    # Token/expert
    offs_tid = pm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_tok = tl.load(sorted_ids_ptr + offs_tid)
    tmask = offs_tok < num_valid_tokens
    exp_id = tl.load(expert_ids_ptr + pm).to(tl.int64)

    if exp_id == -1:
        offs_cn = pn * HALF_N + tl.arange(0, HALF_N)
        cp = c_ptr + stride_cm * offs_tok[:, None] + stride_cn * offs_cn[None, :]
        cm = tmask[:, None] & (offs_cn[None, :] < N // 2)
        tl.store(cp, tl.zeros((BLOCK_M, HALF_N), dtype=tl.bfloat16), mask=cm)
        return

    # Interleave gate/up N-offsets for SwiGLU fusion
    i = tl.arange(0, BLOCK_N).to(tl.int64)
    i2 = i // 2
    h = ((pn * HALF_N + i2) % (N // 2)).to(tl.int64)
    offs_bn = ((h + (i % 2) * (N // 2)) % N).to(tl.int64)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    # B weight pointers: [E, N, K//2]
    # We need b_ptr[exp_id, offs_bn, :] — but B is [E, N, K//2]
    # So b_ptr + exp_id*stride_be + offs_bn*stride_bn + offs_bk*stride_bk
    offs_bk = tl.arange(0, PK)
    bp = (b_ptr + exp_id * stride_be
          + offs_bn[None, :] * stride_bn    # N is dim 1 (rows of output)
          + offs_bk[:, None] * stride_bk)   # K//2 is dim 2

    # B scale pointers: [E, N, K//32] -> load tile as [N, SK]
    offs_sk = tl.arange(0, SK)
    bsp = (b_scale_ptr + exp_id * stride_bse
           + offs_bn.to(tl.int64)[:, None] * stride_bsn
           + offs_sk.to(tl.int64)[None, :] * stride_bsk)

    # A pointers: [M, K] bf16
    offs_ak = tl.arange(0, BLOCK_K)
    ap = a_ptr + offs_tok[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_i in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(ap, mask=tmask[:, None], other=0.0)
            b = tl.load(bp)
            bs = tl.load(bsp)
        else:
            kr = K - k_i * BLOCK_K
            a = tl.load(ap, mask=tmask[:, None] & (offs_ak[None, :] < kr), other=0.0)
            b = tl.load(bp, mask=offs_bk[:, None] < (kr // PACK), other=0)
            bs = tl.load(bsp, mask=offs_sk[None, :] < (kr // MX_BS), other=0)

        # dot_scaled: bf16 A (no microscale) x e2m1 B (with e8m0 microscale)
        # A: [BLOCK_M, BLOCK_K] bf16
        # B: [BLOCK_K//2, BLOCK_N] uint8 (fp4x2)
        # B_scale: [BLOCK_N, BLOCK_K//32] uint8 (e8m0) — matches AITER layout
        acc = tl.dot_scaled(
            a, None, "bf16",
            b, bs, "e2m1",
            acc=acc, fast_math=True,
        )

        ap += BLOCK_K * stride_ak
        bp += PK * stride_bk
        bsp += SK * stride_bsk

    # SwiGLU: split interleaved gate/up, apply silu(gate) * up
    gate, up = acc.reshape(BLOCK_M, HALF_N, 2).split()
    result = (_silu(gate) * up).to(tl.bfloat16)

    offs_cn = pn * HALF_N + tl.arange(0, HALF_N)
    cp = c_ptr + stride_cm * offs_tok[:, None] + stride_cn * offs_cn[None, :]
    cm = tmask[:, None] & (offs_cn[None, :] < N // 2)
    tl.store(cp, result, mask=cm)


# =====================================================================
# Stage 2 Kernel: bf16 intermediate x e2m1 down weights + weighted reduce
#
# Weight B layout (raw): [E, N, K//2] where N = d_hidden_pad, K = d_expert_pad
#   B_scale layout (raw): [E, N, K//32]
# =====================================================================

@triton.jit
def _moe_stage2_kernel(
    a_ptr,              # [M*topk, K] bf16 intermediate
    b_ptr,              # [E, N, K//2] uint8 fp4x2
    b_scale_ptr,        # [E, N, K//32] uint8 e8m0
    c_ptr,              # [M, N] bf16 output (atomic add)
    sorted_ids_ptr,
    expert_ids_ptr,
    num_valid_ptr,
    sorted_weights_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn, stride_bsk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    top_k: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    PACK: tl.constexpr = 2
    MX_BS: tl.constexpr = 32
    PK: tl.constexpr = BLOCK_K // PACK
    SK: tl.constexpr = BLOCK_K // MX_BS

    pid = tl.program_id(0)
    ntp = tl.load(num_valid_ptr)
    npm = tl.cdiv(ntp, BLOCK_M)
    npn = tl.cdiv(N, BLOCK_N)
    GRID = npm * npn
    if pid >= GRID:
        return
    pid = _remap_xcd(pid, GRID, 8)
    pm, pn = _pid_grid(pid, npm, npn, GROUP_M)

    offs_tid = pm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_tok = tl.load(sorted_ids_ptr + offs_tid)
    tmask = offs_tok < num_valid_tokens
    exp_id = tl.load(expert_ids_ptr + pm).to(tl.int64)

    if exp_id == -1:
        return

    offs_bn = (pn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    offs_bk = tl.arange(0, PK)
    bp = (b_ptr + exp_id * stride_be
          + offs_bn[None, :] * stride_bn
          + offs_bk[:, None] * stride_bk)

    # B scale pointers: [E, N, K//32] -> load tile as [N, SK]
    offs_sk = tl.arange(0, SK)
    bsp = (b_scale_ptr + exp_id * stride_bse
           + offs_bn.to(tl.int64)[:, None] * stride_bsn
           + offs_sk.to(tl.int64)[None, :] * stride_bsk)

    offs_ak = tl.arange(0, BLOCK_K)
    ap = a_ptr + offs_tok[:, None] * stride_am + offs_ak[None, :] * stride_ak

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_i in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(ap, mask=tmask[:, None], other=0.0)
            b = tl.load(bp)
            bs = tl.load(bsp)
        else:
            kr = K - k_i * BLOCK_K
            a = tl.load(ap, mask=tmask[:, None] & (offs_ak[None, :] < kr), other=0.0)
            b = tl.load(bp, mask=offs_bk[:, None] < (kr // PACK), other=0)
            bs = tl.load(bsp, mask=offs_sk[None, :] < (kr // MX_BS), other=0)

        acc = tl.dot_scaled(
            a, None, "bf16",
            b, bs, "e2m1",
            acc=acc, fast_math=True,
        )

        ap += BLOCK_K * stride_ak
        bp += PK * stride_bk
        bsp += SK * stride_bsk

    # Weighted reduction: multiply by routing weight, atomic add to output
    w = tl.load(sorted_weights_ptr + offs_tid, mask=tmask, other=0)
    acc = acc * w[:, None]
    result = acc.to(tl.bfloat16)

    offs_cn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    orig = (offs_tok // top_k).to(tl.int64)
    cp = c_ptr + stride_cm * orig[:, None] + stride_cn * offs_cn[None, :]
    cm = tmask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(cp, result, mask=cm)


# =====================================================================
# Python dispatch
# =====================================================================

_buf_cache = {}


def _get_block_m(M, topk, E):
    tpe = M * topk // E
    if tpe <= 4:
        return 16
    elif tpe <= 32:
        return 32
    else:
        return 64


def _triton_moe(hidden_states, w1, w2, w1_scale, w2_scale,
                topk_weights, topk_ids, config):
    """
    Custom 2-stage MoE using bf16 activations + raw MXFP4 weights.
    Uses raw (non-shuffled) weights with standard [E, N, K//2] layout.
    """
    M, topk = topk_ids.shape
    E = w1.shape[0]
    d_hidden = config["d_hidden"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert = config["d_expert"]
    d_expert_pad = config["d_expert_pad"]
    device = hidden_states.device

    # w1 raw: [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
    # w1_scale raw: [E, 2*d_expert_pad, d_hidden_pad//32] e8m0
    N1 = w1.shape[1]          # 2*d_expert_pad
    K1 = w1.shape[2] * 2      # d_hidden_pad (fp4x2 packed)

    # w2 raw: [E, d_hidden_pad, d_expert_pad//2] fp4x2
    # w2_scale raw: [E, d_hidden_pad, d_expert_pad//32] e8m0
    N2 = w2.shape[1]          # d_hidden_pad
    K2 = w2.shape[2] * 2      # d_expert_pad (fp4x2 packed)

    block_m = _get_block_m(M, topk, E)

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = \
        moe_sorting(topk_ids, topk_weights, E, d_hidden_pad, torch.bfloat16, block_m)

    # Intermediate buffer: [M*topk, d_expert_pad]
    ikey = (M, topk, N1 // 2)
    if ikey not in _buf_cache:
        _buf_cache[ikey] = torch.empty(M * topk, N1 // 2, dtype=torch.bfloat16, device=device)
    intermediate = _buf_cache[ikey]

    # Zero output for atomic reduction
    moe_buf.zero_()

    # ---- Stage 1: gate_up GEMM + SwiGLU ----
    EM = sorted_ids.shape[0]
    if M < block_m:
        EM = min(EM, M * topk * block_m)

    BN1 = 128
    BK1 = 128
    grid1 = (triton.cdiv(EM, block_m) * triton.cdiv(N1, BN1),)

    _moe_stage1_kernel[grid1](
        hidden_states,
        w1, w1_scale,
        intermediate,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        N1, K1, M * topk,
        hidden_states.stride(0), hidden_states.stride(1),
        w1.stride(0), w1.stride(1), w1.stride(2),
        w1_scale.stride(0), w1_scale.stride(1), w1_scale.stride(2),
        intermediate.stride(0), intermediate.stride(1),
        BLOCK_M=block_m, BLOCK_N=BN1, BLOCK_K=BK1,
        GROUP_M=8, top_k=topk,
        EVEN_K=(K1 % BK1 == 0),
    )

    # ---- Stage 2: down GEMM + weighted reduction ----
    BN2 = 128
    BK2 = 128
    grid2 = (triton.cdiv(EM, block_m) * triton.cdiv(N2, BN2),)

    _moe_stage2_kernel[grid2](
        intermediate,
        w2, w2_scale,
        moe_buf,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        sorted_weights,
        N2, K2, M * topk,
        intermediate.stride(0), intermediate.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        w2_scale.stride(0), w2_scale.stride(1), w2_scale.stride(2),
        moe_buf.stride(0), moe_buf.stride(1),
        BLOCK_M=block_m, BLOCK_N=BN2, BLOCK_K=BK2,
        GROUP_M=8, top_k=topk,
        EVEN_K=(K2 % BK2 == 0),
    )

    return moe_buf[:, :d_hidden]


# =====================================================================
# AITER fallback (for comparison / safety)
# =====================================================================

_SILU = ActivationType.Silu
_PER1X32 = QuantType.per_1x32


def _aiter_fallback(data):
    (
        hidden_states, _, _, _, _,
        w1s, w2s, w1ss, w2ss,
        topk_weights, topk_ids, config,
    ) = data
    return fused_moe(
        hidden_states, w1s, w2s, topk_weights, topk_ids,
        activation=_SILU, quant_type=_PER1X32,
        w1_scale=w1ss, w2_scale=w2ss,
        hidden_pad=config["d_hidden_pad"] - config["d_hidden"],
        intermediate_pad=config["d_expert_pad"] - config["d_expert"],
    )


# =====================================================================
# Entry point
# =====================================================================

USE_TRITON = True


def custom_kernel(data: input_t) -> output_t:
    if USE_TRITON:
        (
            hidden_states,
            gate_up_weight,          # raw fp4x2
            down_weight,             # raw fp4x2
            gate_up_weight_scale,    # raw e8m0
            down_weight_scale,       # raw e8m0
            _, _,                    # shuffled (unused)
            _, _,                    # shuffled scales (unused)
            topk_weights, topk_ids, config,
        ) = data
        try:
            return _triton_moe(
                hidden_states,
                gate_up_weight, down_weight,
                gate_up_weight_scale, down_weight_scale,
                topk_weights, topk_ids, config,
            )
        except Exception:
            return _aiter_fallback(data)
    return _aiter_fallback(data)
