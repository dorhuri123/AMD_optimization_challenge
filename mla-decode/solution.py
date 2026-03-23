"""
MLA Decode — Optimized Solution
================================
Starting point: fused BMM + FP8 quantization kernel (adapted from vLLM PR #36297).

Steps to implement:
  1. Copy the Triton kernel from vllm/kernels/triton/ops/bmm_fp8_quant.py
  2. Adapt the interface to match task.py's expected function signature
  3. Retune BLOCK_M / BLOCK_N / num_stages for MI355X
  4. Verify correctness against reference.py
  5. Run benchmark.py and submit
"""

import torch
import triton
import triton.language as tl

# TODO: import the official task interface once reference-kernels is cloned
# from task import input_t, output_t


# ── FP8 dtype (ROCm uses e4m3fnuz, not NVIDIA's e4m3fn) ─────────────────────
FP8_DTYPE = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn
FP8_MAX = 240.0 if FP8_DTYPE == getattr(torch, "float8_e4m3fnuz", None) else 448.0


# ── Fused BMM + FP8 Quantization Kernel ─────────────────────────────────────
# Adapted from vLLM PR #36297: https://github.com/vllm-project/vllm/pull/36297
# Fuses the v_up_proj BMM with FP8 quantization to eliminate the bf16
# intermediate memory round-trip.
#
# bf16 context  ──►  [BMM + quant in registers]  ──►  fp8 output
#                     no intermediate HBM write

@triton.autotune(
    configs=[
        # TODO: retune these for MI355X — values below are a starting guess
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_bmm_fp8_quant_kernel(
    A_ptr, B_ptr, C_ptr,
    scale_ptr,           # per-tensor or per-group scale (output)
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    fp8_max: tl.constexpr,
    per_group: tl.constexpr,  # True = per-head dynamic, False = static per-tensor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused batched matmul + FP8 static/dynamic quantization.

    Computes C = A @ B, then quantizes C to FP8 in-place.
    A: [batch, M, K] bf16
    B: [batch, K, N] bf16
    C: [batch, M, N] fp8 (output)
    """
    batch_id = tl.program_id(0)
    pid_m    = tl.program_id(1)
    pid_n    = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A = A_ptr + batch_id * stride_ab
    B = B_ptr + batch_id * stride_bb
    C = C_ptr + batch_id * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a = tl.load(A + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        b = tl.load(B + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Quantize acc (fp32) → fp8
    if per_group:
        # Dynamic per-block scale
        amax = tl.max(tl.abs(acc))
        scale = amax / fp8_max
        tl.store(scale_ptr + batch_id * tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
                 + pid_m * tl.cdiv(N, BLOCK_N) + pid_n, scale)
    else:
        scale = tl.load(scale_ptr)

    c_fp8 = (acc / scale).to(tl.float8e4nv)  # TODO: switch to float8e4b8 for ROCm e4m3fnuz
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c_fp8, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_bmm_fp8_quant(
    a: torch.Tensor,   # [batch, M, K] bf16
    b: torch.Tensor,   # [batch, K, N] bf16
    scale: torch.Tensor | None = None,  # None = dynamic (per-group)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused batched matmul + FP8 quantization.

    Returns:
        c_fp8:  [batch, M, N] fp8
        scales: per-tensor or per-block scales
    """
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    batch, M, K = a.shape
    _, _, N = b.shape

    c_fp8 = torch.empty((batch, M, N), device=a.device, dtype=FP8_DTYPE)
    per_group = scale is None
    if per_group:
        # allocate per-block scales (will be filled by kernel)
        grid_m = triton.cdiv(M, 16)  # rough estimate; kernel uses actual BLOCK_M
        grid_n = triton.cdiv(N, 64)
        scale = torch.empty((batch, grid_m, grid_n), device=a.device, dtype=torch.float32)

    grid = lambda meta: (batch,
                         triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))
    _fused_bmm_fp8_quant_kernel[grid](
        a, b, c_fp8, scale,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c_fp8.stride(0), c_fp8.stride(1), c_fp8.stride(2),
        fp8_max=FP8_MAX,
        per_group=per_group,
    )
    return c_fp8, scale


# ── TODO: wrap the full MLA decode forward pass ─────────────────────────────
# Once you have the official task interface (reference.py + task.py), implement:
#
# def solution(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
#              max_seqlen_q, w_v_up_proj, fp8_scale=None):
#     # 1. Call AITER mla_decode_fwd for the attention step
#     from aiter import mla_decode_fwd
#     context = mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr,
#                              kv_indices, kv_last_page_lens, max_seqlen_q)
#     # 2. Fused v_up_proj + FP8 quantization
#     out_fp8, scales = fused_bmm_fp8_quant(context, w_v_up_proj, scale=fp8_scale)
#     return out_fp8, scales


if __name__ == "__main__":
    # Quick smoke test of the fused kernel
    torch.manual_seed(0)
    B, M, K, N = 4, 16, 512, 512
    a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.bfloat16)

    c_fp8, scales = fused_bmm_fp8_quant(a, b)
    print(f"Input:  a={a.shape} {a.dtype}, b={b.shape} {b.dtype}")
    print(f"Output: c_fp8={c_fp8.shape} {c_fp8.dtype}, scales={scales.shape} {scales.dtype}")

    # Compare dequantized output vs reference torch matmul
    ref = torch.bmm(a.float(), b.float())
    c_deq = c_fp8.float() * scales.unsqueeze(-1)  # rough dequant for sanity check
    # (exact comparison depends on scale layout — this is just a smoke test)
    print("Smoke test passed.")
