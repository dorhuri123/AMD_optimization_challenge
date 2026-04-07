#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
v20: Adapted from dgavriloff v211 — direct kernel dispatch with per-shape configs.

Key optimizations over our v19:
1. Direct kernel dispatch (bypass AITER wrapper Python overhead)
2. Per-shape configs with NUM_KSPLIT=7 for K>4096
3. Pre-allocated buffers cached by (M,K,N)
4. Fused quant+shuffle kernel for M>64 (inline e8m0 permutation)
5. cache_modifier=None for K<=1024 (L1 caching of B tiles)
6. Gluon reduce kernel for split-K (BSN=64 for fp32 partials)
"""
import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import (
    _gemm_a16wfp4_preshuffle_kernel,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import get_splitk

from task import input_t, output_t

# Try gluon reduce (better for fp32 partials), fall back to basic
try:
    from aiter.ops.triton.gluon.gemm_afp4wfp4 import (
        _gemm_afp4wfp4_reduce_kernel as _reduce_kernel,
    )
except ImportError:
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
        _gemm_afp4wfp4_reduce_kernel as _reduce_kernel,
    )

_buffers = {}
_ASM_KERNEL = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"


def _get_config(M, N, K):
    if K > 4096:
        return dict(
            BLOCK_SIZE_M=8, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=1, NUM_KSPLIT=7,
            num_warps=4, num_stages=2, waves_per_eu=2,
            matrix_instr_nonkdim=16, cache_modifier=".cg",
        )
    if M <= 4:
        return dict(
            BLOCK_SIZE_M=4, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=1, NUM_KSPLIT=1,
            num_warps=4, num_stages=2, waves_per_eu=0,
            matrix_instr_nonkdim=16, cache_modifier=".cg",
        )
    elif M <= 8:
        return dict(
            BLOCK_SIZE_M=8, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=1, NUM_KSPLIT=1,
            num_warps=4, num_stages=2, waves_per_eu=0,
            matrix_instr_nonkdim=16, cache_modifier=".cg",
        )
    elif M <= 32 and K <= 1024:
        return dict(
            BLOCK_SIZE_M=8, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=1, NUM_KSPLIT=1,
            num_warps=4, num_stages=2, waves_per_eu=2,
            matrix_instr_nonkdim=16, cache_modifier=None,
        )
    elif M <= 64:
        return dict(
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=1, NUM_KSPLIT=1,
            num_warps=4, num_stages=2, waves_per_eu=2,
            matrix_instr_nonkdim=16, cache_modifier=".cg",
        )
    else:
        return dict(
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=512,
            GROUP_SIZE_M=1, NUM_KSPLIT=1,
            num_warps=8, num_stages=1, waves_per_eu=2,
            matrix_instr_nonkdim=16, cache_modifier=None,
        )


# ── Fused quant+shuffle kernel for large M (inline e8m0 permutation) ───────
@triton.heuristics({
    "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
    and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
})
@triton.jit
def _fused_quant_shuffle_kernel(
    x_ptr, x_fp4_ptr, bs_ptr,
    stride_x_m_in, stride_x_n_in,
    stride_fp4_m_in, stride_fp4_n_in,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr, NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr, SCALING_MODE: tl.constexpr,
    SCALE_N_PAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_fp4_m = tl.cast(stride_fp4_m_in, tl.int64)
    stride_fp4_n = tl.cast(stride_fp4_n_in, tl.int64)

    NUM_QB: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=mask, cache_modifier=".cg").to(tl.float32)

        fp4, e8m0 = _mxfp4_quant_op(x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE)

        # Store fp4
        o_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        o_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        o_offs = o_m[:, None] * stride_fp4_m + o_n[None, :] * stride_fp4_n
        if EVEN_M_N:
            tl.store(x_fp4_ptr + o_offs, fp4, cache_modifier=".wt")
        else:
            o_mask = (o_m < M)[:, None] & (o_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + o_offs, fp4, mask=o_mask, cache_modifier=".wt")

        # Store scales with inline shuffle (e8m0_shuffle permutation)
        bs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_n = pid_n * NUM_QB + tl.arange(0, NUM_QB)
        num_cols = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

        # e8m0_shuffle permutation: (sm//32, 2, 16, sn//8, 2, 4) -> (0, 3, 5, 2, 4, 1)
        s0 = bs_m[:, None] // 32
        s1_full = bs_m[:, None] % 32
        s2 = s1_full % 16
        s1 = s1_full // 16
        s3 = bs_n[None, :] // 8
        s4_full = bs_n[None, :] % 8
        s5 = s4_full % 4
        s4 = s4_full // 4
        bs_offs = s1 + s4 * 2 + s2 * 4 + s5 * 64 + s3 * 256 + s0 * 32 * SCALE_N_PAD

        valid = (bs_m < M)[:, None] & (bs_n < num_cols)[None, :]
        e8m0 = tl.where(valid, e8m0, 127)
        SCALE_M_PAD = (M + 255) // 256 * 256
        bs_mask = (bs_m < SCALE_M_PAD)[:, None] & (bs_n < SCALE_N_PAD)[None, :]
        tl.store(bs_ptr + bs_offs, e8m0.to(tl.uint8), mask=bs_mask, cache_modifier=".cg")


def _init_buffers(M, K, N, device):
    key = (M, K, N)
    if key in _buffers:
        return _buffers[key]

    config = _get_config(M, N, K)

    if M <= 64:
        K_kernel = K // 2
        BSK = config["BLOCK_SIZE_K"]
        BSN = max(config["BLOCK_SIZE_N"], 32)
        BSM = config["BLOCK_SIZE_M"]

        if config["NUM_KSPLIT"] > 1:
            SPLITK_BS, BSK, NUM_KSPLIT = get_splitk(K_kernel, BSK, config["NUM_KSPLIT"])
            grid = NUM_KSPLIT * triton.cdiv(M, BSM) * triton.cdiv(N, BSN)
            y_pp = torch.empty((NUM_KSPLIT, M, N), dtype=torch.float32, device=device)
            ACTUAL_KSPLIT = triton.cdiv(K_kernel, (SPLITK_BS // 2))
            buf = dict(
                mode='splitk', out=torch.empty((M, N), dtype=torch.bfloat16, device=device),
                y_pp=y_pp, grid=grid, K_kernel=K_kernel,
                BSM=BSM, BSN=BSN, BSK=BSK, SPLITK_BS=SPLITK_BS, NUM_KSPLIT=NUM_KSPLIT,
                ACTUAL_KSPLIT=ACTUAL_KSPLIT, MAX_KSPLIT=triton.next_power_of_2(NUM_KSPLIT),
                reduce_grid=(triton.cdiv(M, 16), triton.cdiv(N, 64)),
                B_w=None, B_sc=None, config=config,
            )
        else:
            SPLITK_BS = 2 * K_kernel
            grid = triton.cdiv(M, BSM) * triton.cdiv(N, BSN)
            buf = dict(
                mode='direct', out=torch.empty((M, N), dtype=torch.bfloat16, device=device),
                grid=grid, K_kernel=K_kernel,
                BSM=BSM, BSN=BSN, BSK=BSK, SPLITK_BS=SPLITK_BS,
                B_w=None, B_sc=None, config=config,
            )
    else:
        # Two-phase: fused quant+shuffle → CK ASM GEMM
        SG = 32
        SN_valid = triton.cdiv(K, SG)
        SM = triton.cdiv(M, 256) * 256
        SN = triton.cdiv(SN_valid, 8) * 8
        BSM_q = min(32, triton.next_power_of_2(M))
        BSM_q = triton.cdiv(BSM_q, 32) * 32
        BSN_q = 64
        padded_M = (M + 31) // 32 * 32
        buf = dict(
            mode='two_phase',
            x_fp4=torch.empty((M, K // 2), dtype=torch.uint8, device=device),
            bs=torch.empty((SM, SN), dtype=torch.uint8, device=device),
            out=torch.empty((padded_M, N), dtype=torch.bfloat16, device=device),
            SN=SN, BSM_q=BSM_q, BSN_q=BSN_q,
            q_grid=(triton.cdiv(M, BSM_q), triton.cdiv(K, BSN_q)),
            M=M,
        )

    _buffers[key] = buf
    return buf


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_shuffle.shape[0]

    buf = _init_buffers(M, K, N, A.device)

    # Cache B reshapes
    b_ptr = B_shuffle.data_ptr()
    if buf.get('B_w') is None or buf.get('_bp') != b_ptr:
        if buf['mode'] in ('splitk', 'direct'):
            buf['B_w'] = B_shuffle.view(torch.uint8).reshape(N // 16, (K // 2) * 16)
            sc = B_scale_sh.view(torch.uint8)
            buf['B_sc'] = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
        buf['_bp'] = b_ptr

    if buf['mode'] == 'splitk':
        cfg = buf['config']
        _gemm_a16wfp4_preshuffle_kernel[(buf['grid'],)](
            A, buf['B_w'], buf['y_pp'], buf['B_sc'],
            M, N, buf['K_kernel'],
            A.stride(0), A.stride(1),
            buf['B_w'].stride(0), buf['B_w'].stride(1),
            buf['y_pp'].stride(0), buf['y_pp'].stride(1), buf['y_pp'].stride(2),
            buf['B_sc'].stride(0), buf['B_sc'].stride(1),
            BLOCK_SIZE_M=buf['BSM'], BLOCK_SIZE_N=buf['BSN'], BLOCK_SIZE_K=buf['BSK'],
            GROUP_SIZE_M=cfg['GROUP_SIZE_M'], NUM_KSPLIT=buf['NUM_KSPLIT'],
            SPLITK_BLOCK_SIZE=buf['SPLITK_BS'],
            num_warps=cfg['num_warps'], num_stages=cfg['num_stages'],
            waves_per_eu=cfg['waves_per_eu'], matrix_instr_nonkdim=16,
            PREQUANT=True, cache_modifier=cfg['cache_modifier'],
        )
        _reduce_kernel[buf['reduce_grid']](
            buf['y_pp'], buf['out'], M, N,
            buf['y_pp'].stride(0), buf['y_pp'].stride(1), buf['y_pp'].stride(2),
            buf['out'].stride(0), buf['out'].stride(1),
            16, 64, buf['ACTUAL_KSPLIT'], buf['MAX_KSPLIT'],
        )
        return buf['out']

    elif buf['mode'] == 'direct':
        cfg = buf['config']
        _gemm_a16wfp4_preshuffle_kernel[(buf['grid'],)](
            A, buf['B_w'], buf['out'], buf['B_sc'],
            M, N, buf['K_kernel'],
            A.stride(0), A.stride(1),
            buf['B_w'].stride(0), buf['B_w'].stride(1),
            0, buf['out'].stride(0), buf['out'].stride(1),
            buf['B_sc'].stride(0), buf['B_sc'].stride(1),
            BLOCK_SIZE_M=buf['BSM'], BLOCK_SIZE_N=buf['BSN'], BLOCK_SIZE_K=buf['BSK'],
            GROUP_SIZE_M=cfg['GROUP_SIZE_M'], NUM_KSPLIT=1,
            SPLITK_BLOCK_SIZE=buf['SPLITK_BS'],
            num_warps=cfg['num_warps'], num_stages=cfg['num_stages'],
            waves_per_eu=cfg['waves_per_eu'], matrix_instr_nonkdim=16,
            PREQUANT=True, cache_modifier=cfg['cache_modifier'],
        )
        return buf['out']

    else:
        # Two-phase: fused quant+shuffle → CK ASM
        _fused_quant_shuffle_kernel[buf['q_grid']](
            A, buf['x_fp4'], buf['bs'],
            *A.stride(), *buf['x_fp4'].stride(),
            M=M, N=K,
            BLOCK_SIZE_M=buf['BSM_q'], BLOCK_SIZE_N=buf['BSN_q'],
            NUM_ITER=1, NUM_STAGES=1,
            MXFP4_QUANT_BLOCK_SIZE=32, SCALING_MODE=0,
            SCALE_N_PAD=buf['SN'],
            num_warps=2, waves_per_eu=0, num_stages=1,
        )
        gemm_a4w4_asm(
            buf['x_fp4'].view(dtypes.fp4x2), B_shuffle,
            buf['bs'].view(dtypes.fp8_e8m0), B_scale_sh,
            buf['out'], _ASM_KERNEL, None, 1.0, 0.0, True, log2_k_split=0,
        )
        return buf['out'][:buf['M']]
