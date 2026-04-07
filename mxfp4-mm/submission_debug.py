"""Debug: log which path was taken and any errors from fused path."""
import torch
import sys
import traceback
from task import input_t, output_t

_BF16 = torch.bfloat16

# ── Log fused import attempt ────────────────────────────────────────────────
_fused_import_error = None
_preshuffle = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _preshuffle = gemm_a16wfp4_preshuffle
    print("[DEBUG] gemm_a16wfp4_preshuffle imported OK", file=sys.stderr)
except Exception as e:
    _fused_import_error = str(e)
    print(f"[DEBUG] fused import FAILED: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# Also try direct kernel import
_direct_kernel = None
try:
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import _gemm_a16wfp4_preshuffle_kernel
    _direct_kernel = _gemm_a16wfp4_preshuffle_kernel
    print("[DEBUG] _gemm_a16wfp4_preshuffle_kernel imported OK", file=sys.stderr)
except Exception as e:
    print(f"[DEBUG] direct kernel import FAILED: {e}", file=sys.stderr)

# Also try non-preshuffle
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    print("[DEBUG] gemm_a16wfp4 (non-preshuffle) imported OK", file=sys.stderr)
except Exception as e:
    print(f"[DEBUG] gemm_a16wfp4 import FAILED: {e}", file=sys.stderr)

# Check if dot_scaled is available in triton
try:
    import triton.language as tl
    print(f"[DEBUG] triton.language imported OK, has dot_scaled: {hasattr(tl, 'dot_scaled')}", file=sys.stderr)
except Exception as e:
    print(f"[DEBUG] triton import FAILED: {e}", file=sys.stderr)

# Check arch info
try:
    import aiter.ops.triton.utils._triton.arch_info as arch_info
    print(f"[DEBUG] is_fp4_avail: {arch_info.is_fp4_avail()}", file=sys.stderr)
except Exception as e:
    print(f"[DEBUG] arch_info FAILED: {e}", file=sys.stderr)

# ── CK fallback ────────────────────────────────────────────────────────────
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4

_active_fn = None


def _fused_kernel(data):
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, k_packed = B_u8.shape
    w = B_u8.reshape(n // 16, k_packed * 16)
    sc = B_scale_sh.view(torch.uint8)
    w_scales = sc.reshape(sc.shape[0] // 32, sc.shape[1] * 32)
    print(f"[DEBUG] Calling preshuffle: A={A.shape}, w={w.shape}, w_scales={w_scales.shape}", file=sys.stderr)
    return _preshuffle(A, w, w_scales, prequant=True, dtype=_BF16)


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

    if _preshuffle is not None:
        try:
            result = _fused_kernel(data)
            print("[DEBUG] FUSED PATH SUCCESS!", file=sys.stderr)
            _active_fn = _fused_kernel
            return result
        except Exception as e:
            print(f"[DEBUG] Fused path RUNTIME ERROR: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print("[DEBUG] Using CK fallback", file=sys.stderr)
    _active_fn = _ck_kernel
    return _ck_kernel(data)
