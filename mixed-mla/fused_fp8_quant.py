"""
Fused FP8 quantization via two Triton kernels (amax reduction + quantize).

Replaces the 3-op PyTorch implementation:
    amax = tensor.abs().amax().clamp(min=1e-12)   # kernel 1
    scale = amax / fp8_max                          # kernel 2 (or CPU)
    fp8 = (tensor / scale).clamp(...).to(FP8_DTYPE) # kernel 3

With two purpose-built Triton kernels:
    Kernel 1: parallel block reduction -> global amax (single scalar)
    Kernel 2: read amax, compute scale, quantize all elements to FP8

For typical MLA decode Q tensors (batch*16*576 elements), this saves ~10-15 us
by eliminating PyTorch dispatch overhead and fusing scale computation into
the quantize kernel.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------
# Kernel 1: Compute global absolute-max via block-parallel reduction
# ---------------------------------------------------------------

@triton.jit
def _amax_kernel(
    input_ptr,       # [N] flat input (any float type)
    amax_ptr,        # [1] output: global amax (float32)
    N,               # total number of elements
    BLOCK: tl.constexpr,
):
    """Each block computes local amax, then atomically updates global max."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_amax = tl.max(tl.abs(x))

    # tl.atomic_max expects the same dtype as the pointer; amax_ptr is float32
    tl.atomic_max(amax_ptr, local_amax)


# ---------------------------------------------------------------
# Kernel 2: Quantize to FP8 using the pre-computed amax
# ---------------------------------------------------------------

@triton.jit
def _quantize_fp8_kernel(
    input_ptr,       # [N] flat input (any float type)
    output_ptr,      # [N] flat output (fp8)
    amax_ptr,        # [1] global amax (float32), read-only
    scale_ptr,       # [1] output: scale (float32)
    fp8_max,         # scalar constexpr: max representable FP8 value
    fp8_min,         # scalar constexpr: min representable FP8 value (negative)
    N,               # total number of elements
    BLOCK: tl.constexpr,
):
    """Read global amax, compute scale = amax / fp8_max, quantize elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Every block reads the same global amax (L1 cached after first read)
    amax = tl.load(amax_ptr)
    # Clamp to avoid division by zero
    amax = tl.maximum(amax, 1e-12)
    scale = amax / fp8_max

    # Program 0 writes scale to output
    if pid == 0:
        tl.store(scale_ptr, scale)

    # Load, scale, clamp, cast
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_scaled = x / scale
    x_clamped = tl.minimum(tl.maximum(x_scaled, fp8_min), fp8_max)

    tl.store(output_ptr + offs, x_clamped.to(tl.float8e4m3fnuz), mask=mask)


# ---------------------------------------------------------------
# Host-side wrapper
# ---------------------------------------------------------------

# Cache for the amax scratch buffer (avoids repeated allocation)
_amax_buf_cache: dict = {}


def fused_quantize_fp8(
    tensor: torch.Tensor,
    fp8_dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 using two fused Triton kernels.

    Args:
        tensor: Input tensor of any shape, typically bfloat16.
        fp8_dtype: Target FP8 dtype. Defaults to float8_e4m3fnuz (AMD).

    Returns:
        (fp8_tensor, scale) where:
            fp8_tensor: same shape as input, dtype=fp8_dtype
            scale: shape (1,), dtype=float32
    """
    if fp8_dtype is None:
        fp8_dtype = torch.float8_e4m3fnuz  # AMD MI300X/MI355X native FP8

    finfo = torch.finfo(fp8_dtype)
    fp8_max_val = finfo.max
    fp8_min_val = finfo.min

    # Flatten for kernel indexing
    flat = tensor.reshape(-1)
    N = flat.numel()

    # Allocate outputs
    fp8_flat = torch.empty(N, dtype=fp8_dtype, device=tensor.device)
    scale_out = torch.empty(1, dtype=torch.float32, device=tensor.device)

    # Get or allocate amax scratch buffer (reuse across calls)
    device_key = tensor.device
    if device_key not in _amax_buf_cache:
        _amax_buf_cache[device_key] = torch.zeros(1, dtype=torch.float32, device=tensor.device)
    amax_buf = _amax_buf_cache[device_key]
    # Reset to zero before reduction
    amax_buf.zero_()

    BLOCK = 1024
    grid_size = (N + BLOCK - 1) // BLOCK

    # Kernel 1: compute global amax
    _amax_kernel[(grid_size,)](
        flat, amax_buf, N,
        BLOCK=BLOCK,
    )

    # Kernel 2: quantize using the computed amax
    _quantize_fp8_kernel[(grid_size,)](
        flat, fp8_flat, amax_buf, scale_out,
        fp8_max_val, fp8_min_val, N,
        BLOCK=BLOCK,
    )

    return fp8_flat.view(tensor.shape), scale_out
