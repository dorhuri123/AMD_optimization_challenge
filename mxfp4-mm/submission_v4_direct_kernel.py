"""
MXFP4-MM: Direct Triton kernel call bypassing AITER wrappers.

Calls _gemm_a16wfp4_preshuffle_kernel directly to avoid any Python wrapper
overhead or dtype canonicalization issues. Handles config selection and
splitK reduction inline.

Falls back to CK path if Triton import fails.
"""
import torch
import triton
from task import input_t, output_t

_BF16 = torch.bfloat16

# ── Try direct kernel imports ───────────────────────────────────────────────
_has_kernel = False
try:
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import (
        _gemm_a16wfp4_preshuffle_kernel,
        _get_config,
        get_splitk,
    )
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
        _gemm_afp4wfp4_reduce_kernel,
    )
    _has_kernel = True
except Exception:
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle as _preshuffle_wrapper
        _has_kernel = "wrapper"
    except Exception:
        pass

# ── CK fallback ────────────────────────────────────────────────────────────
import aiter
from aiter import dtypes as _dt
from aiter.ops.triton.quant import dynamic_mxfp4_quant as _quant
from aiter.utility.fp4_utils import e8m0_shuffle as _shuffle
_gemm_a4w4 = aiter.gemm_a4w4

# ── Config cache ───��────────────────────────────────────────────────────────
_config_cache = {}


def _get_cached_config(M, N, K):
    key = (M, N, K)
    if key not in _config_cache:
        cfg, _ = _get_config(M, N, K, shuffle=True)
        _config_cache[key] = cfg
    return dict(_config_cache[key])  # return copy


def _direct_kernel(data):
    """Call the Triton preshuffle kernel directly, no wrapper overhead."""
    A, _, _, B_shuffle, B_scale_sh = data
    m, k = A.shape

    # Reshape B for preshuffle format
    B_u8 = B_shuffle.view(torch.uint8)
    n = B_u8.shape[0]
    k_packed = B_u8.shape[1]
    w = B_u8.reshape(n // 16, k_packed * 16)

    # Reshape scales for preshuffle format
    sc_u8 = B_scale_sh.view(torch.uint8)
    sc_r, sc_c = sc_u8.shape
    w_scales = sc_u8.reshape(sc_r // 32, sc_c * 32)

    # Get config
    N_actual = n
    K_actual = k_packed  # k // 2 in the wrapper's convention
    config = _get_cached_config(m, N_actual, K_actual)

    # Handle splitK
    NUM_KSPLIT = config.get("NUM_KSPLIT", 1)
    BLOCK_K = config["BLOCK_SIZE_K"]

    if NUM_KSPLIT > 1:
        SPLITK_BLOCK_SIZE, BLOCK_K, NUM_KSPLIT = get_splitk(K_actual, BLOCK_K, NUM_KSPLIT)
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

    if BLOCK_K >= 2 * K_actual:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K_actual)
        config["SPLITK_BLOCK_SIZE"] = 2 * K_actual
        config["NUM_KSPLIT"] = 1
    config["BLOCK_SIZE_N"] = max(config.get("BLOCK_SIZE_N", 32), 32)

    NUM_KSPLIT = config["NUM_KSPLIT"]
    use_splitk = NUM_KSPLIT > 1

    if use_splitk:
        y_pp = torch.empty((NUM_KSPLIT, m, N_actual), dtype=torch.float32, device=A.device)
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K_actual
        y_pp = None

    y = torch.empty((m, N_actual), dtype=_BF16, device=A.device)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]

    grid = (
        NUM_KSPLIT
        * triton.cdiv(m, BLOCK_M)
        * triton.cdiv(N_actual, BLOCK_N),
    )

    out_tensor = y if y_pp is None else y_pp

    _gemm_a16wfp4_preshuffle_kernel[grid](
        A,
        w,
        out_tensor,
        w_scales,
        m,
        N_actual,
        K_actual,
        A.stride(0),
        A.stride(1),
        w.stride(0),
        w.stride(1),
        0 if y_pp is None else y_pp.stride(0),
        y.stride(0) if y_pp is None else y_pp.stride(1),
        y.stride(1) if y_pp is None else y_pp.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        PREQUANT=True,
        **config,
    )

    if use_splitk:
        ACTUAL_KSPLIT = triton.cdiv(K_actual, (config["SPLITK_BLOCK_SIZE"] // 2))
        grid_reduce = (
            triton.cdiv(m, 16),
            triton.cdiv(N_actual, 64),
        )
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp, y, m, N_actual,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1),
            16, 64, ACTUAL_KSPLIT,
            triton.next_power_of_2(NUM_KSPLIT),
        )

    return y


def _wrapper_kernel(data):
    """Use AITER wrapper with proper reshape."""
    A, _, _, B_shuffle, B_scale_sh = data
    B_u8 = B_shuffle.view(torch.uint8)
    n, k_packed = B_u8.shape
    w = B_u8.reshape(n // 16, k_packed * 16)
    sc_u8 = B_scale_sh.view(torch.uint8)
    sc_r, sc_c = sc_u8.shape
    w_scales = sc_u8.reshape(sc_r // 32, sc_c * 32)
    return _preshuffle_wrapper(A, w, w_scales, prequant=True, dtype=_BF16)


def _ck_kernel(data):
    A, _, _, B_shuffle, B_scale_sh = data
    A_fp4, A_scale_raw = _quant(A)
    return _gemm_a4w4(
        A_fp4.view(_dt.fp4x2), B_shuffle,
        _shuffle(A_scale_raw).view(_dt.fp8_e8m0), B_scale_sh,
        bpreshuffle=True, dtype=_dt.bf16,
    )


# ── Dispatch ────────────────────────────────────────────────────────────────
_active_fn = None


def custom_kernel(data: input_t) -> output_t:
    global _active_fn

    if _active_fn is not None:
        return _active_fn(data)

    # Strategy 1: Direct kernel call
    if _has_kernel is True:
        try:
            result = _direct_kernel(data)
            _active_fn = _direct_kernel
            return result
        except Exception:
            pass

    # Strategy 2: Wrapper call with reshape
    if _has_kernel == "wrapper":
        try:
            result = _wrapper_kernel(data)
            _active_fn = _wrapper_kernel
            return result
        except Exception:
            pass

    # Strategy 3: CK fallback
    _active_fn = _ck_kernel
    return _ck_kernel(data)
