"""
mixed-mla — custom_kernel submission
=====================================
DeepSeek R1 forward_absorb MLA decode, optimized for AMD MI355X.

Interface (from task.py):
  data = (q, kv_data, qo_indptr, kv_indptr, config)
    q          : (total_q, 16, 576)    bfloat16   — absorbed query (kv_lora_rank + qk_rope_dim)
    kv_data    : dict with keys:
                   "bf16"  -> Tensor (total_kv, 1, 576)           bfloat16
                   "fp8"   -> (kv_fp8, scale)  fp8 + scalar float
                   "mxfp4" -> (kv_fp4x2, scale_e8m0)
    qo_indptr  : (batch+1,)  int32
    kv_indptr  : (batch+1,)  int32
    config     : dict with MLA parameters
  Output: (total_q, 16, 512)  bfloat16

Architecture (DeepSeek R1):
  num_heads     = 16
  num_kv_heads  = 1
  kv_lora_rank  = 512   (value dim, first 512 of kv_buffer)
  qk_rope_dim   = 64
  qk_head_dim   = 576   (kv_lora_rank + qk_rope_dim)
  sm_scale      = 1 / sqrt(576)

Reference performance (AITER a8w8 on MI355X):
  fp8 Q + fp8 KV: 1.4–2.3x over bf16

Optimization target:
  Use MXFP4 KV cache (4x bandwidth savings vs bf16) and/or fuse
  dequant into the attention kernel to exceed the a8w8 reference.
"""

import torch
import aiter
from task import input_t, output_t

# ── Architecture constants ────────────────────────────────────────────
NUM_HEADS    = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_DIM  = 64
QK_HEAD_DIM  = KV_LORA_RANK + QK_ROPE_DIM  # 576
SM_SCALE     = 1.0 / (QK_HEAD_DIM ** 0.5)

# ROCm FP8 format (e4m3fnuz, max = 240, not NVIDIA's e4m3fn max = 448)
FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX   = torch.finfo(FP8_DTYPE).max


def _quant_fp8_dynamic(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization (sglang style)."""
    scale = x.abs().max() / FP8_MAX
    return (x / scale).to(FP8_DTYPE), scale


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized MLA decode kernel.

    Current strategy: fp8 Q + fp8 KV via AITER persistent decode kernel
    (mirrors the reference a8w8 path).

    TODO optimizations to beat the reference:
      1. Use MXFP4 KV cache instead of FP8 — 4x bandwidth vs bf16, 2x vs fp8
         → uncomment USE_MXFP4 block below once AITER mxfp4 MLA path is verified
      2. Tune num_kv_splits for MI355X (currently auto-determined by AITER)
      3. Write a custom Triton kernel that fuses dequant + attention + output
         → key insight from vLLM PR #36297: fusing BMM + FP8 quant on the
           output side yields 5.2–6.6x on MI300X; apply same idea here for
           the kv-side dequantization
    """
    q, kv_data, qo_indptr, kv_indptr, config = data

    total_q  = q.shape[0]
    output   = torch.empty(
        (total_q, NUM_HEADS, KV_LORA_RANK), device=q.device, dtype=torch.bfloat16
    )
    kv_last_page_lens = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    kv_indices = torch.arange(kv_indptr[-1].item(), device=q.device, dtype=torch.int32)

    # ── Path A: MXFP4 KV (4x BW savings) ────────────────────────────
    # TODO: enable once AITER MXFP4 MLA decode API is confirmed
    # if "mxfp4" in kv_data:
    #     kv_buffer, kv_scale = kv_data["mxfp4"]
    #     q_fp8, q_scale = _quant_fp8_dynamic(q)
    #     aiter.mla_decode_fwd(
    #         q_fp8, kv_buffer, output, qo_indptr, kv_indptr,
    #         kv_indices, kv_last_page_lens, max_seqlen_q=1,
    #         sm_scale=SM_SCALE, kv_scale=kv_scale, q_scale=q_scale,
    #     )
    #     return output

    # ── Path B: FP8 Q + FP8 KV (a8w8, matches reference) ────────────
    if "fp8" in kv_data:
        kv_buffer, kv_scale = kv_data["fp8"]
        q_fp8, q_scale = _quant_fp8_dynamic(q)
        aiter.mla_decode_fwd(
            q=q_fp8,
            kv_buffer=kv_buffer,
            o=output,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            max_seqlen_q=1,
            sm_scale=SM_SCALE,
            kv_scale=kv_scale,
            q_scale=q_scale,
        )
        return output

    # ── Path C: BF16 fallback ────────────────────────────────────────
    kv_buffer = kv_data["bf16"]
    aiter.mla_decode_fwd(
        q=q,
        kv_buffer=kv_buffer,
        o=output,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        max_seqlen_q=1,
        sm_scale=SM_SCALE,
    )
    return output
