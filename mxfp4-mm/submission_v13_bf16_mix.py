"""
v13: Try bf16×fp4 mixed precision for large M.

For m > 64, use tl.dot_scaled(A_bf16, None, "bf16", B_fp4, B_scales, "e2m1")
which skips A quantization entirely. If the result is within rtol=1e-2 of the
fp4×fp4 reference, this eliminates ALL A-side overhead.

For m ≤ 64, use fused preshuffle (proven fast).
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

_BF16 = torch.bfloat16

# ── Fused preshuffle for small M ───────────────────────────────────────────
_preshuffle = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _preshuffle = gemm_a16wfp4_preshuffle
except Exception:
    pass

# Also try the non-preshuffle variant (bf16 A, no quant option)
_a16wfp4 = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    _a16wfp4 = gemm_a16wfp4
except Exception:
    pass

# ── CK fallback ────────────────────────────────────────────────────────────
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4
_FP4X2 = _dt.fp4x2
_E8M0 = _dt.fp8_e8m0


# ── Custom bf16 × fp4 GEMM kernel ──────────────────────────────────────────
@triton.jit
def _bf16_x_e2m1_kernel(
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
):
    """bf16 A × e2m1 B via tl.dot_scaled — NO A quantization."""
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
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_kp[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    bs_ptrs = bs_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)        # [BLOCK_M, BLOCK_K] bf16
        b = tl.load(b_ptrs)        # [BLOCK_K//2, BLOCK_N] uint8 (e2m1 packed)
        b_sc = tl.load(bs_ptrs)    # [BLOCK_N, BLOCK_K//32] uint8 (e8m0)

        # Mixed precision: bf16 × e2m1 with hardware scale MFMA
        acc += tl.dot_scaled(a, None, "bf16", b, b_sc, "e2m1")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        bs_ptrs += (BLOCK_K // SCALE_GROUP) * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _bf16_mix_kernel(data):
    """bf16 A × fp4 B — no A quant at all."""
    A, _, B_q, _, B_scale_sh = data
    m, k = A.shape
    n = B_q.shape[0]

    # Need raw (non-shuffled) B scales for this kernel
    B_u8 = B_q.view(torch.uint8)
    B_t = B_u8.T.contiguous()  # [K//2, N]

    # Re-derive raw B scales
    _, B_scale_raw = _quant(data[1])  # B bf16
    B_sc = B_scale_raw.view(torch.uint8)[:n, :k // 32].contiguous()

    C = torch.empty((m, n), dtype=_BF16, device=A.device)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

    _bf16_x_e2m1_kernel[grid](
        A, B_t, C, B_sc,
        m, n, k,
        A.stride(0), A.stride(1),
        B_t.stride(0), B_t.stride(1),
        C.stride(0), C.stride(1),
        B_sc.stride(0), B_sc.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
        num_warps=8, num_stages=2,
    )
    return C


def _fused_preshuffle(data):
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, kp = B_u8.shape
    w = B_u8.reshape(n // 16, kp * 16)
    sc = B_scale_sh.view(torch.uint8)
    ws = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
    return _preshuffle(A, w, ws, prequant=True, dtype=_BF16)


def _lean_ck(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_sc = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


_triton_ok = None
_bf16_ok = None
_M_THRESHOLD = 64


def custom_kernel(data: input_t) -> output_t:
    global _triton_ok, _bf16_ok

    A = data[0]
    m = A.shape[0]

    # Large M: try bf16×fp4 mixed (no A quant)
    if m > _M_THRESHOLD:
        if _bf16_ok is None:
            try:
                result = _bf16_mix_kernel(data)
                _bf16_ok = True
                return result
            except Exception:
                _bf16_ok = False
        elif _bf16_ok:
            return _bf16_mix_kernel(data)
        return _lean_ck(data)

    # Small M: fused preshuffle
    if _preshuffle is None or _triton_ok is False:
        return _lean_ck(data)

    if _triton_ok is None:
        try:
            result = _fused_preshuffle(data)
            _triton_ok = True
            return result
        except Exception:
            _triton_ok = False
            return _lean_ck(data)

    return _fused_preshuffle(data)
