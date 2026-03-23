"""
MXFP4 MoE — Benchmark: reference vs solution
Run: python benchmark.py [--batch_size N]
"""

import torch
import triton
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEVICE      = "cuda"
HIDDEN_DIM  = 4096
FFN_DIM     = 14336
NUM_EXPERTS = 128
TOP_K       = 4
MXFP4_BLOCK = 32

# Token counts to sweep (simulates different batch sizes / sequence lengths)
TOKEN_COUNTS = [1, 8, 32, 128, 512, 2048, 4096]


def make_inputs(num_tokens):
    hidden  = torch.randn(num_tokens, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
    router  = torch.randn(num_tokens, NUM_EXPERTS, device=DEVICE)
    # Fake MXFP4 expert weights
    w1_fp4    = torch.randint(0, 256, (NUM_EXPERTS, HIDDEN_DIM // 2, FFN_DIM),
                               device=DEVICE, dtype=torch.uint8)
    w1_scales = torch.ones(NUM_EXPERTS, HIDDEN_DIM // MXFP4_BLOCK, FFN_DIM, device=DEVICE)
    w2_fp4    = torch.randint(0, 256, (NUM_EXPERTS, FFN_DIM // 2, HIDDEN_DIM),
                               device=DEVICE, dtype=torch.uint8)
    w2_scales = torch.ones(NUM_EXPERTS, FFN_DIM // MXFP4_BLOCK, HIDDEN_DIM, device=DEVICE)
    # Dense BF16 weights for reference
    w1_full = torch.randn(NUM_EXPERTS, HIDDEN_DIM, FFN_DIM, device=DEVICE, dtype=torch.bfloat16)
    w2_full = torch.randn(NUM_EXPERTS, FFN_DIM, HIDDEN_DIM, device=DEVICE, dtype=torch.bfloat16)
    return hidden, router, w1_fp4, w1_scales, w2_fp4, w2_scales, w1_full, w2_full


def reference_forward(hidden, router, w1_full, w2_full):
    """Baseline: Python loop over experts, bf16 GEMMs."""
    import torch.nn.functional as F
    from solution import compute_routing
    num_tokens = hidden.shape[0]
    expert_ids, expert_weights, tokens_per_expert = compute_routing(router)
    expert_order = expert_ids.reshape(-1).argsort(stable=True)
    token_indices = expert_order // TOP_K
    tokens_scattered = hidden[token_indices]
    sorted_expert_ids = expert_ids.reshape(-1)[expert_order]

    output = torch.zeros_like(hidden)
    offset = 0
    for e in range(NUM_EXPERTS):
        n = tokens_per_expert[e].item()
        if n == 0:
            continue
        t = tokens_scattered[offset:offset + n].float()
        h = F.silu(t @ w1_full[e].float())
        o = h @ w2_full[e].float()
        weights_e = expert_weights.reshape(-1)[expert_order][offset:offset + n]
        output.scatter_add_(0,
            token_indices[offset:offset + n].unsqueeze(1).expand(-1, HIDDEN_DIM),
            (o.bfloat16() * weights_e.unsqueeze(1)))
        offset += n
    return output


def solution_forward(hidden, router, w1_fp4, w1_scales, w2_fp4, w2_scales):
    from solution import moe_forward
    return moe_forward(hidden, router, w1_fp4, w1_scales, w2_fp4, w2_scales)


def bench(fn, *args, warmup=10, rep=50):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=None, help="single token count to test")
args = parser.parse_args()

counts = [args.batch_size] if args.batch_size else TOKEN_COUNTS

print(f"\n{'Tokens':>8} {'Ref (ms)':>10} {'Sol (ms)':>10} {'Speedup':>9}")
print("-" * 44)

for num_tokens in counts:
    hidden, router, w1_fp4, w1_scales, w2_fp4, w2_scales, w1_full, w2_full = make_inputs(num_tokens)
    try:
        t_ref = bench(reference_forward, hidden, router, w1_full, w2_full)
        t_sol = bench(solution_forward, hidden, router, w1_fp4, w1_scales, w2_fp4, w2_scales)
        speedup = t_ref / t_sol
        flag = " ✓" if speedup >= 1.0 else " ✗"
        print(f"{num_tokens:>8} {t_ref:>10.3f} {t_sol:>10.3f} {speedup:>8.2f}x{flag}")
    except Exception as e:
        print(f"{num_tokens:>8}  ERROR: {e}")
