"""
MXFP4-MM v8: Triton small M + CK with splitK tuning for large M.

Try different splitK values for the CK gemm_a4w4_blockscale to find
the optimal parallelism for large M shapes.
"""
import torch
from task import input_t, output_t

_BF16 = torch.bfloat16

# ── Fused Triton ────────────────────────────────────────────────────────────
_preshuffle = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _preshuffle = gemm_a16wfp4_preshuffle
except Exception:
    pass

# ── CK imports ──────────────────────────────────────────────────────────────
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_FP4X2 = _dt.fp4x2
_E8M0 = _dt.fp8_e8m0

# Try to import blockscale directly for splitK control
_blockscale = None
try:
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale
    _blockscale = gemm_a4w4_blockscale
except Exception:
    pass

_gemm_a4w4 = aiter.gemm_a4w4
_M_THRESHOLD = 64


def _fused_triton(data):
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, kp = B_u8.shape
    w = B_u8.reshape(n // 16, kp * 16)
    sc = B_scale_sh.view(torch.uint8)
    ws = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
    return _preshuffle(A, w, ws, prequant=True, dtype=_BF16)


def _ck_splitk(data, splitK=0):
    """CK with explicit splitK control."""
    A, _, _, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B_shuffle.view(torch.uint8).shape[0]

    A_fp4, A_sc = _quant(A)
    A_q = A_fp4.view(_FP4X2).view(m, k // 2)
    A_sc_sh = _shuffle(A_sc).view(_E8M0)

    # Pad M to multiple of 32 for CK
    m_padded = (m + 31) // 32 * 32
    out = torch.empty((m_padded, n), dtype=_BF16, device=A.device)

    if _blockscale is not None:
        _blockscale(A_q, B_shuffle, A_sc_sh, B_scale_sh, out, splitK=splitK)
    else:
        return _gemm_a4w4(
            A_q, B_shuffle, A_sc_sh, B_scale_sh,
            bpreshuffle=True, dtype=_dt.bf16,
        )

    return out[:m]


def _lean_ck(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_sc = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


_triton_ok = None


def custom_kernel(data: input_t) -> output_t:
    global _triton_ok

    A = data[0]
    m = A.shape[0]

    # Large M: CK with splitK=2 for more parallelism
    if m > _M_THRESHOLD:
        try:
            return _ck_splitk(data, splitK=2)
        except Exception:
            return _lean_ck(data)

    # Small M: fused Triton
    if _preshuffle is None or _triton_ok is False:
        return _lean_ck(data)

    if _triton_ok is None:
        try:
            result = _fused_triton(data)
            _triton_ok = True
            return result
        except Exception:
            _triton_ok = False
            return _lean_ck(data)

    return _fused_triton(data)
