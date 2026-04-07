"""
MXFP4-MM: Self-contained Triton kernel with fused bf16->mxfp4 quant + GEMM.

Uses tl.dot_scaled("e2m1", "e2m1") for hardware-accelerated FP4 matrix multiply on MI355X.
Inline _mxfp4_quant_op eliminates separate quant kernel launch.

Works with pre-shuffled B weights (B_shuffle) and pre-shuffled B scales (B_scale_sh).
The preshuffle format matches AITER's shuffle_weight(layout=(16,16)) and e8m0_shuffle().
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

_BF16 = torch.bfloat16


# ── Inline MXFP4 quantization (from AITER _mxfp4_quant_op) ────────────────
@triton.jit
def _mxfp4_quant(
    x,  # [BLOCK_M, BLOCK_K] bf16/f32
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    QGROUP: tl.constexpr,  # 32
):
    NUM_BLK: tl.constexpr = BLOCK_K // QGROUP
    x = x.reshape(BLOCK_M, NUM_BLK, QGROUP)

    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_exp = tl.log2(amax).floor() - 2
    scale_exp = tl.clamp(scale_exp, min=-127, max=127)

    bs_e8m0 = scale_exp.to(tl.uint8) + 127
    qx = x * tl.exp2(-scale_exp)
    qx = qx.to(tl.uint32, bitcast=True)

    s = qx & 0x80000000
    qx = qx ^ s
    qf = qx.to(tl.float32, bitcast=True)

    sat = qf >= 6
    den = (not sat) & (qf < 1)
    nor = not (sat | den)

    # Denormal
    denorm_x = qf + tl.cast(149 << 23, tl.float32, bitcast=True)
    denorm_x = denorm_x.to(tl.uint32, bitcast=True) - (149 << 23)
    denorm_x = denorm_x.to(tl.uint8)

    # Normal
    nor_x = qx
    mant_odd = (nor_x >> 22) & 1
    nor_x += ((-126 << 23) + (1 << 21) - 1)
    nor_x += mant_odd
    nor_x = nor_x >> 22
    nor_x = nor_x.to(tl.uint8)

    e2m1 = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1 = tl.where(nor, nor_x, e2m1)
    e2m1 = tl.where(den, denorm_x, e2m1)
    sign_lp = (s >> 28).to(tl.uint8)
    e2m1 = e2m1 | sign_lp

    e2m1 = tl.reshape(e2m1, [BLOCK_M, NUM_BLK, QGROUP // 2, 2])
    evens, odds = tl.split(e2m1)
    fp4 = evens | (odds << 4)
    fp4 = fp4.reshape(BLOCK_M, BLOCK_K // 2)

    return fp4, bs_e8m0.reshape(BLOCK_M, NUM_BLK)


# ── GEMM kernel: bf16 A x preshuffle-fp4 B ─────────────────────────────────
@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] // 2) == 0,
})
@triton.jit
def _fused_gemm_preshuffle(
    a_ptr, b_ptr, c_ptr, bs_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fused bf16→mxfp4 quant + A4W4 GEMM with pre-shuffled B.

    A: [M, K] bf16
    B: pre-shuffled fp4 layout from shuffle_weight(layout=(16,16))
    B_scales: pre-shuffled e8m0 from e8m0_shuffle()
    C: [M, N] bf16 output
    """
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for L2 reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # A pointers: [BLOCK_M, BLOCK_K] in bf16
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_k_bf16 = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k_bf16[None, :] * stride_ak

    # B pointers: pre-shuffled layout — each "row" is (N//16) blocks, each block is (K//2)*16 bytes
    offs_k_shuffle = tl.arange(0, (BLOCK_K // 2) * 16)
    offs_bn = (pid_n * (BLOCK_N // 16) + tl.arange(0, BLOCK_N // 16)) % N
    b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk

    # B scale pointers: pre-shuffled e8m0
    offs_bsn = (pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP * 32)
    bs_ptrs = bs_ptr + offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    for _ in range(num_k_iters):
        # Load and un-shuffle B scales
        b_scales = (
            tl.load(bs_ptrs)
            .reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP // 8, 4, 16, 2, 2, 1)
            .permute(0, 5, 3, 1, 4, 2, 6)
            .reshape(BLOCK_N, BLOCK_K // SCALE_GROUP)
        )

        # Load A as bf16 and B as pre-shuffled fp4
        a_bf16 = tl.load(a_ptrs)
        b_raw = tl.load(b_ptrs)

        # Un-shuffle B tile
        b = (
            b_raw
            .reshape(1, BLOCK_N // 16, BLOCK_K // 64, 2, 16, 16)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(BLOCK_N, BLOCK_K // 2)
            .trans(1, 0)
        )

        # Inline quantize A to MXFP4
        a_fp4, a_scales = _mxfp4_quant(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP)

        # Hardware-accelerated scaled dot product
        acc += tl.dot_scaled(a_fp4, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * 16 * stride_bk
        bs_ptrs += BLOCK_K * stride_bsk

    # Write output
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ── Non-preshuffle variant: simpler B layout ───────────────────────────────
@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] // 2) == 0,
})
@triton.jit
def _fused_gemm_raw(
    a_ptr, b_ptr, c_ptr, bs_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Fused bf16→mxfp4 quant + A4W4 GEMM with raw (non-shuffled) B.

    A: [M, K] bf16
    B: [K//2, N] fp4 packed (transposed raw layout)
    B_scales: [N, K//32] e8m0 raw
    C: [M, N] bf16 output
    """
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k_bf16 = tl.arange(0, BLOCK_K)
    offs_k_fp4 = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k_bf16[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k_fp4[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptrs = bs_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    for _ in range(num_k_iters):
        b_scales = tl.load(bs_ptrs)
        a_bf16 = tl.load(a_ptrs)
        b_fp4 = tl.load(b_ptrs)

        a_fp4, a_scales = _mxfp4_quant(a_bf16, BLOCK_K, BLOCK_M, SCALE_GROUP)
        acc += tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        bs_ptrs += (BLOCK_K // SCALE_GROUP) * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ── Config tuning ───────────────────────────────────────────────────────────
def _get_config(m, n, k):
    """Tile size selection tuned for MI355X benchmark shapes."""
    # All benchmark shapes: m in {4,16,32,64,256}, n in {2112,2880,3072,4096,7168}, k in {512,1536,2048,7168}
    if m <= 16:
        # Small M: memory-bound, wider N tiles to amortize B loads
        return dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=2)
    elif m <= 64:
        return dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=2)
    else:
        # Large M: compute-bound, balance tile sizes
        return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                    num_warps=8, num_stages=3)


# ── Lean CK fallback ───────────────────────────────────────────────────────
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _dyn_quant
from aiter.utility.fp4_utils import e8m0_shuffle as _e8m0_shuffle
_gemm_a4w4 = aiter.gemm_a4w4


def _lean_ck(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_sc = _dyn_quant(A)
    return _gemm_a4w4(
        A_fp4.view(_dt.fp4x2), B_shuffle,
        _e8m0_shuffle(A_sc).view(_dt.fp8_e8m0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ── Main entry ──────────────────────────────────────────────────────────────
_active_fn = None


def custom_kernel(data: input_t) -> output_t:
    global _active_fn

    if _active_fn is not None:
        return _active_fn(data)

    A, _, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B_q.shape[0]

    # Strategy 1: Try AITER's fused preshuffle wrapper
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
        result = gemm_a16wfp4_preshuffle(A, B_shuffle, B_scale_sh, prequant=True, dtype=_BF16)
        _active_fn = lambda d: gemm_a16wfp4_preshuffle(d[0], d[3], d[4], prequant=True, dtype=_BF16)
        return result
    except Exception:
        pass

    # Strategy 2: Custom Triton preshuffle kernel
    try:
        cfg = _get_config(m, n, k)
        nw = cfg.pop("num_warps")
        ns = cfg.pop("num_stages")

        C = torch.empty((m, n), dtype=_BF16, device=A.device)
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

        _fused_gemm_preshuffle[grid](
            A, B_shuffle, C, B_scale_sh,
            m, n, k,
            A.stride(0), A.stride(1),
            B_shuffle.stride(0), B_shuffle.stride(1),
            C.stride(0), C.stride(1),
            B_scale_sh.stride(0), B_scale_sh.stride(1),
            num_warps=nw, num_stages=ns,
            **cfg,
        )

        def _triton_preshuffle(d):
            A_, _, _, B_sh_, B_sc_ = d
            m_, k_ = A_.shape
            n_ = B_sh_.shape[0] * 16  # preshuffle packs 16 columns
            c_ = _get_config(m_, n_, k_)
            nw_ = c_.pop("num_warps")
            ns_ = c_.pop("num_stages")
            C_ = torch.empty((m_, n_), dtype=_BF16, device=A_.device)
            g = lambda META: (triton.cdiv(m_, META["BLOCK_M"]) * triton.cdiv(n_, META["BLOCK_N"]),)
            _fused_gemm_preshuffle[g](
                A_, B_sh_, C_, B_sc_,
                m_, n_, k_,
                A_.stride(0), A_.stride(1),
                B_sh_.stride(0), B_sh_.stride(1),
                C_.stride(0), C_.stride(1),
                B_sc_.stride(0), B_sc_.stride(1),
                num_warps=nw_, num_stages=ns_,
                **c_,
            )
            return C_

        _active_fn = _triton_preshuffle
        return C
    except Exception:
        pass

    # Strategy 3: Custom Triton raw kernel (B in raw [K//2, N] layout)
    try:
        B_t = B_q.view(torch.uint8).T.contiguous()
        # We need raw (non-shuffled) B scales — re-quantize B to get them
        _, B_scale_raw = _dyn_quant(data[1])  # data[1] = B bf16
        B_scale_raw = B_scale_raw.view(torch.uint8)

        cfg = _get_config(m, n, k)
        nw = cfg.pop("num_warps")
        ns = cfg.pop("num_stages")
        C = torch.empty((m, n), dtype=_BF16, device=A.device)
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

        _fused_gemm_raw[grid](
            A, B_t, C, B_scale_raw,
            m, n, k,
            A.stride(0), A.stride(1),
            B_t.stride(0), B_t.stride(1),
            C.stride(0), C.stride(1),
            B_scale_raw.stride(0), B_scale_raw.stride(1),
            num_warps=nw, num_stages=ns,
            **cfg,
        )

        def _triton_raw(d):
            A_, B_bf16, B_q_, _, _ = d
            m_, k_ = A_.shape
            n_ = B_q_.shape[0]
            B_t_ = B_q_.view(torch.uint8).T.contiguous()
            _, B_sc_ = _dyn_quant(B_bf16)
            B_sc_ = B_sc_.view(torch.uint8)
            c_ = _get_config(m_, n_, k_)
            nw_ = c_.pop("num_warps")
            ns_ = c_.pop("num_stages")
            C_ = torch.empty((m_, n_), dtype=_BF16, device=A_.device)
            g = lambda META: (triton.cdiv(m_, META["BLOCK_M"]) * triton.cdiv(n_, META["BLOCK_N"]),)
            _fused_gemm_raw[g](
                A_, B_t_, C_, B_sc_,
                m_, n_, k_,
                A_.stride(0), A_.stride(1),
                B_t_.stride(0), B_t_.stride(1),
                C_.stride(0), C_.stride(1),
                B_sc_.stride(0), B_sc_.stride(1),
                num_warps=nw_, num_stages=ns_,
                **c_,
            )
            return C_

        _active_fn = _triton_raw
        return C
    except Exception:
        pass

    # Strategy 4: Lean CK fallback (always works)
    _active_fn = _lean_ck
    return _lean_ck(data)
