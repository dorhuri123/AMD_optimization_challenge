"""
v11: Try gemm_a16wfp4 (non-preshuffle) for large M shapes.

This variant takes raw B_q (not shuffled) and raw B scales.
We need to re-derive raw scales from B (original bf16) on first call,
then cache them. The non-preshuffle kernel may have better configs
for compute-bound shapes.

For small M, use fused preshuffle (proven fast).
"""
import torch
import sys
from task import input_t, output_t

_BF16 = torch.bfloat16

# ── Imports ─────────────────────────────────────────────────────────────────
_preshuffle = None
_a16wfp4 = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle, gemm_a16wfp4
    _preshuffle = gemm_a16wfp4_preshuffle
    _a16wfp4 = gemm_a16wfp4
except Exception as e:
    print(f"[v11] import error: {e}", file=sys.stderr)

import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4
_FP4X2 = _dt.fp4x2
_E8M0 = _dt.fp8_e8m0

_M_THRESHOLD = 64


def _fused_preshuffle(data):
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, kp = B_u8.shape
    w = B_u8.reshape(n // 16, kp * 16)
    sc = B_scale_sh.view(torch.uint8)
    ws = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
    return _preshuffle(A, w, ws, prequant=True, dtype=_BF16)


def _raw_a16wfp4(data):
    """Non-preshuffle: bf16 A × raw fp4 B with raw e8m0 scales."""
    A, B_bf16, B_q, _, _ = data
    # B_q is [N, K//2] raw fp4 — need raw (non-shuffled) scales
    # Re-derive from B_bf16
    _, B_scale_raw = _quant(B_bf16)
    B_scale_raw_u8 = B_scale_raw.view(torch.uint8)
    n = B_q.shape[0]
    k_scale = A.shape[1] // 32
    B_scale_raw_u8 = B_scale_raw_u8[:n, :k_scale].contiguous()
    B_q_u8 = B_q.view(torch.uint8)
    return _a16wfp4(A, B_q_u8, B_scale_raw_u8, dtype=_BF16)


def _lean_ck(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_sc = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_FP4X2), B_shuffle,
        _shuffle(A_sc).view(_E8M0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


_triton_ok = None
_a16w_ok = None


def custom_kernel(data: input_t) -> output_t:
    global _triton_ok, _a16w_ok

    A = data[0]
    m = A.shape[0]

    # Large M: try non-preshuffle a16wfp4 (may have better configs)
    if m > _M_THRESHOLD and _a16wfp4 is not None:
        if _a16w_ok is None:
            try:
                result = _raw_a16wfp4(data)
                _a16w_ok = True
                print(f"[v11] a16wfp4 raw path OK for m={m}", file=sys.stderr)
                return result
            except Exception as e:
                _a16w_ok = False
                print(f"[v11] a16wfp4 raw FAILED: {e}", file=sys.stderr)
        elif _a16w_ok:
            return _raw_a16wfp4(data)

    # Large M fallback: CK
    if m > _M_THRESHOLD:
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
