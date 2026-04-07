"""
MXFP4-MM v5: Mixed-precision bf16 A × MXFP4 B via tl.dot_scaled("bf16", "e2m1").

Inspired by dgavriloff's MLA kernel which uses dot_scaled with bf16 query and
MXFP4 keys. Skips A quantization entirely — zero overhead, one kernel.

Risk: output may not match fp4×fp4 reference within rtol=1e-2, atol=1e-2.
If it fails correctness, fall back to fused fp4×fp4 or CK path.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

_BF16 = torch.bfloat16


@triton.jit
def _gemm_bf16_x_fp4_kernel(
    a_ptr, b_ptr, c_ptr, bs_ptr,
    M, N, K,  # K = actual K (not packed)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """bf16 A × MXFP4 B GEMM using tl.dot_scaled("bf16", "e2m1").
    No A quantization needed — hardware handles mixed-precision.

    A: [M, K] bf16
    B: [N, K//2] fp4 packed (uint8), row-major
    B_scales: [N, K//32] e8m0 (uint8)
    C: [M, N] bf16
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

    # A pointers: [BLOCK_M, BLOCK_K] bf16
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B pointers: [BLOCK_K//2, BLOCK_N] uint8 (packed fp4), transposed from [N, K//2]
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    b_ptrs = b_ptr + offs_k_packed[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # B scale pointers: [BLOCK_N, BLOCK_K//32] uint8
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)
    bs_ptrs = bs_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    for _ in range(num_k_iters):
        a_tile = tl.load(a_ptrs)  # [BLOCK_M, BLOCK_K] bf16
        b_tile = tl.load(b_ptrs)  # [BLOCK_K//2, BLOCK_N] uint8
        b_scales = tl.load(bs_ptrs)  # [BLOCK_N, BLOCK_K//32] uint8

        # Mixed-precision dot: bf16 A, MXFP4 B
        acc += tl.dot_scaled(a_tile, None, "bf16", b_tile, b_scales, "e2m1")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        bs_ptrs += (BLOCK_K // SCALE_GROUP) * stride_bsk

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _get_config(m, n, k):
    if m <= 16:
        return dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=2)
    elif m <= 64:
        return dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=2)
    else:
        return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
                    num_warps=8, num_stages=3)


# ── Fallback imports ────────────────────────────────────────────────────────
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4

_active_fn = None


def _bf16_x_fp4_kernel(data):
    """Zero-quant path: bf16 A × fp4 B using mixed-precision dot_scaled."""
    A, _, B_q, _, B_scale_sh = data
    m, k = A.shape
    n = B_q.shape[0]

    # B_q: [N, K//2] fp4 packed → view as uint8, transpose for kernel
    B_u8 = B_q.view(torch.uint8)
    B_t = B_u8.T.contiguous()  # [K//2, N]

    # B_scale_sh: shuffled e8m0, but we need raw scales for this kernel
    # We need un-shuffled [N, K//32] scales.
    # Since we don't have raw scales, re-derive from B (original bf16 weights)
    B_bf16 = data[1]  # B original
    _, B_scale_raw = _quant(B_bf16)
    B_scale_raw = B_scale_raw.view(torch.uint8)
    # Trim to actual shape [N, K//32]
    B_scale_raw = B_scale_raw[:n, :k // 32].contiguous()

    cfg = _get_config(m, n, k)
    nw = cfg.pop("num_warps")
    ns = cfg.pop("num_stages")

    C = torch.empty((m, n), dtype=_BF16, device=A.device)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

    _gemm_bf16_x_fp4_kernel[grid](
        A, B_t, C, B_scale_raw,
        m, n, k,
        A.stride(0), A.stride(1),
        B_t.stride(0), B_t.stride(1),
        C.stride(0), C.stride(1),
        B_scale_raw.stride(0), B_scale_raw.stride(1),
        num_warps=nw, num_stages=ns,
        **cfg,
    )
    return C


def _ck_kernel(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_scale_raw = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_dt.fp4x2), B_shuffle,
        _shuffle(A_scale_raw).view(_dt.fp8_e8m0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


def custom_kernel(data: input_t) -> output_t:
    global _active_fn

    if _active_fn is not None:
        return _active_fn(data)

    # Try bf16×fp4 mixed precision (fastest if it passes correctness)
    try:
        result = _bf16_x_fp4_kernel(data)
        _active_fn = _bf16_x_fp4_kernel
        return result
    except Exception:
        pass

    _active_fn = _ck_kernel
    return _ck_kernel(data)
