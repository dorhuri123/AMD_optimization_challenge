"""
MLA Decode — Benchmark: reference vs solution
Run: python benchmark.py
"""

import torch
import triton
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = "cuda"

# ── Configs to sweep ────────────────────────────────────────────────────────
BATCH_SIZES  = [1, 4, 8, 16, 32, 64, 128, 256]
SEQ_LENS     = [512, 1024, 2048, 4096, 8192]
NUM_HEADS    = 128
KV_LORA_RANK = 512
QK_ROPE_DIM  = 64
KV_HEAD_DIM  = KV_LORA_RANK + QK_ROPE_DIM   # 576


def make_inputs(batch: int, seq_len: int):
    q  = torch.randn(batch, NUM_HEADS, KV_LORA_RANK + QK_ROPE_DIM,
                     device=DEVICE, dtype=torch.bfloat16)
    # Simplified: use contiguous KV (replace with paged layout once task.py available)
    kv = torch.randn(batch, seq_len, KV_HEAD_DIM,
                     device=DEVICE, dtype=torch.bfloat16)
    w_v = torch.randn(NUM_HEADS, KV_LORA_RANK, KV_LORA_RANK,
                      device=DEVICE, dtype=torch.bfloat16)
    return q, kv, w_v


def reference_forward(q, kv, w_v):
    """Baseline: separate BMM + FP8 quant (no fusion)."""
    # Simulated attention output: [batch, heads, kv_lora_rank]
    context = torch.einsum("bhs,bls->bhl", q[..., :KV_LORA_RANK], kv[..., :KV_LORA_RANK])
    context = context / (KV_LORA_RANK ** 0.5)
    # Unfused output projection
    out_bf16 = torch.bmm(context, w_v.reshape(NUM_HEADS * KV_LORA_RANK, KV_LORA_RANK)
                         .T.unsqueeze(0).expand(context.shape[0], -1, -1))
    fp8_dtype = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn
    fp8_max = 240.0 if fp8_dtype == getattr(torch, "float8_e4m3fnuz", None) else 448.0
    scale = out_bf16.abs().max() / fp8_max
    out_fp8 = (out_bf16 / scale).to(fp8_dtype)
    return out_fp8


def solution_forward(q, kv, w_v):
    """Our solution: TODO — wire in fused kernel once task interface is known."""
    from solution import fused_bmm_fp8_quant
    context = torch.einsum("bhs,bls->bhl", q[..., :KV_LORA_RANK], kv[..., :KV_LORA_RANK])
    context = context / (KV_LORA_RANK ** 0.5)
    out_fp8, scales = fused_bmm_fp8_quant(
        context,
        w_v.reshape(NUM_HEADS * KV_LORA_RANK, KV_LORA_RANK)
           .T.unsqueeze(0).expand(context.shape[0], -1, -1)
    )
    return out_fp8


def bench(fn, *args, warmup=25, rep=100):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


print(f"\n{'Batch':>6} {'SeqLen':>8} {'Ref (ms)':>10} {'Sol (ms)':>10} {'Speedup':>9}")
print("-" * 50)

for bs in BATCH_SIZES:
    for sl in SEQ_LENS:
        q, kv, w_v = make_inputs(bs, sl)
        try:
            t_ref = bench(reference_forward, q, kv, w_v)
            t_sol = bench(solution_forward, q, kv, w_v)
            speedup = t_ref / t_sol
            flag = " ✓" if speedup >= 1.0 else " ✗"
            print(f"{bs:>6} {sl:>8} {t_ref:>10.3f} {t_sol:>10.3f} {speedup:>8.2f}x{flag}")
        except Exception as e:
            print(f"{bs:>6} {sl:>8}  ERROR: {e}")
