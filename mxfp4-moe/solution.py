"""
MXFP4 MoE — Optimized Solution
=================================
Goal: optimize Mixture of Experts with MXFP4-quantized expert weights on MI355X.

Architecture (gpt-oss):
  - 128 experts total
  - Each token routed to top-4 experts
  - No shared expert
  - Expert weights in MXFP4

The two main kernels:
  1. Router + topK + sort (scatter tokens to experts)
  2. Grouped GEMM over experts (variable M per expert, fixed N, K)

Steps:
  1. Get the official baseline from reference-kernels and study task.yml
  2. Profile the reference to find the bottleneck (routing vs GEMM)
  3. Implement grouped GEMM that reuses the MXFP4 GEMM kernel from mxfp4-gemm/
  4. Optimize the routing sort
  5. Fuse where possible
"""

import torch
import triton
import triton.language as tl
import sys
from pathlib import Path

# Reuse the MXFP4 GEMM kernel from the sibling challenge
sys.path.insert(0, str(Path(__file__).parent.parent / "mxfp4-gemm"))
# from solution import mxfp4_gemm  # uncomment once mxfp4-gemm solution is ready

# TODO: import official task interface once reference-kernels is cloned
# from task import input_t, output_t

MXFP4_BLOCK_SIZE = 32
NUM_EXPERTS  = 128
TOP_K        = 4


# ── 1. Routing Kernel ────────────────────────────────────────────────────────

def compute_routing(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-K routing and sort tokens by expert.

    Args:
        router_logits: [num_tokens, num_experts]

    Returns:
        expert_ids:        [num_tokens, TOP_K]  — which experts each token goes to
        expert_weights:    [num_tokens, TOP_K]  — softmax weights
        tokens_per_expert: [num_experts]        — how many tokens each expert receives
    """
    num_tokens = router_logits.shape[0]
    # Softmax + topK
    probs = torch.softmax(router_logits, dim=-1)
    top_weights, top_expert_ids = torch.topk(probs, TOP_K, dim=-1)
    # Normalize weights
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

    # Count tokens per expert (for grouped GEMM dispatch)
    tokens_per_expert = torch.zeros(NUM_EXPERTS, device=router_logits.device, dtype=torch.int32)
    tokens_per_expert.scatter_add_(0,
        top_expert_ids.reshape(-1),
        torch.ones(num_tokens * TOP_K, device=router_logits.device, dtype=torch.int32))

    return top_expert_ids, top_weights, tokens_per_expert


@triton.jit
def _scatter_tokens_kernel(
    tokens_ptr,         # [num_tokens, hidden_dim]
    expert_ids_ptr,     # [num_tokens, TOP_K]
    output_ptr,         # [num_tokens * TOP_K, hidden_dim]  — tokens scattered by expert
    cum_expert_ptr,     # [num_experts + 1]  — prefix sum of tokens_per_expert
    dispatch_idx_ptr,   # [num_tokens * TOP_K]  — output write positions
    num_tokens, hidden_dim,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Scatter tokens from [num_tokens, hidden] to expert-sorted [total, hidden]."""
    tid = tl.program_id(0)   # token index
    kid = tl.program_id(1)   # top-k index

    expert_id = tl.load(expert_ids_ptr + tid * TOP_K + kid)
    # Atomically claim a slot in this expert's section
    slot = tl.atomic_add(cum_expert_ptr + expert_id, 1)
    write_idx = slot

    # Write token to output
    offs_h = tl.arange(0, BLOCK_H)
    for h in range(0, tl.cdiv(hidden_dim, BLOCK_H)):
        h_offs = h * BLOCK_H + offs_h
        x = tl.load(tokens_ptr + tid * hidden_dim + h_offs,
                    mask=h_offs < hidden_dim, other=0.0)
        tl.store(output_ptr + write_idx * hidden_dim + h_offs,
                 x, mask=h_offs < hidden_dim)


# ── 2. Grouped GEMM Dispatch ─────────────────────────────────────────────────

def grouped_mxfp4_gemm(
    tokens_by_expert: torch.Tensor,   # [total_tokens, hidden_dim] — scattered tokens
    expert_w1_fp4: torch.Tensor,      # [num_experts, hidden_dim//2, ffn_dim] — packed FP4
    expert_w1_scales: torch.Tensor,   # [num_experts, hidden_dim//MXFP4_BLOCK_SIZE, ffn_dim]
    tokens_per_expert: torch.Tensor,  # [num_experts] — how many tokens per expert
) -> torch.Tensor:
    """Apply the first FFN linear layer for each expert.

    This is a grouped GEMM where each expert has a variable number of input tokens
    but the same weight shape (hidden_dim × ffn_dim).

    TODO: replace with a proper grouped GEMM kernel once mxfp4-gemm solution is ready.
    For now, uses a Python loop over experts as a correctness reference.
    """
    num_experts = expert_w1_fp4.shape[0]
    _, hidden_dim = tokens_by_expert.shape
    ffn_dim = expert_w1_scales.shape[-1]

    outputs = []
    offset = 0
    for e in range(num_experts):
        n = tokens_per_expert[e].item()
        if n == 0:
            continue
        tokens_e = tokens_by_expert[offset:offset + n]  # [n, hidden_dim]
        w_fp4_e   = expert_w1_fp4[e]                    # [hidden_dim//2, ffn_dim]
        w_scale_e = expert_w1_scales[e]                  # [groups, ffn_dim]

        # TODO: call mxfp4_gemm(tokens_e, w_fp4_e, w_scale_e) once kernel is ready
        # For now: dequantize + torch.mm as placeholder
        w_dequant = w_fp4_e.float()  # placeholder dequant
        out_e = torch.mm(tokens_e.float(), w_dequant).bfloat16()
        outputs.append(out_e)
        offset += n

    return torch.cat(outputs, dim=0) if outputs else tokens_by_expert.new_empty(0, ffn_dim)


# ── 3. Full MoE Forward ───────────────────────────────────────────────────────

def moe_forward(
    hidden_states: torch.Tensor,     # [num_tokens, hidden_dim] bf16
    router_logits: torch.Tensor,     # [num_tokens, num_experts]
    expert_w1_fp4: torch.Tensor,     # [num_experts, hidden_dim//2, ffn_dim]
    expert_w1_scales: torch.Tensor,
    expert_w2_fp4: torch.Tensor,     # [num_experts, ffn_dim//2, hidden_dim]
    expert_w2_scales: torch.Tensor,
) -> torch.Tensor:
    """Full MoE forward pass (gpt-oss style, no shared expert)."""
    num_tokens = hidden_states.shape[0]

    # 1. Route tokens
    expert_ids, expert_weights, tokens_per_expert = compute_routing(router_logits)

    # 2. Scatter tokens to expert order (TODO: replace loop with _scatter_tokens_kernel)
    total_dispatched = TOP_K * num_tokens
    expert_order = expert_ids.reshape(-1).argsort(stable=True)  # sort by expert_id
    token_indices = expert_order // TOP_K
    tokens_scattered = hidden_states[token_indices]   # [total, hidden]
    sorted_expert_ids = expert_ids.reshape(-1)[expert_order]

    # Recompute tokens_per_expert in sorted order (already done above)

    # 3. GEMM1 + activation (SiLU for gated FFN, or GELU)
    ffn_hidden = grouped_mxfp4_gemm(
        tokens_scattered, expert_w1_fp4, expert_w1_scales, tokens_per_expert)
    ffn_hidden = torch.nn.functional.silu(ffn_hidden)  # SiLU activation

    # 4. GEMM2
    expert_out = grouped_mxfp4_gemm(
        ffn_hidden, expert_w2_fp4, expert_w2_scales, tokens_per_expert)

    # 5. Weighted gather back to [num_tokens, hidden_dim]
    output = torch.zeros_like(hidden_states)
    weights_flat = expert_weights.reshape(-1)[expert_order]
    output.scatter_add_(0,
        token_indices.unsqueeze(1).expand(-1, hidden_states.shape[1]),
        (expert_out * weights_flat.unsqueeze(1)).to(hidden_states.dtype))

    return output


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    num_tokens = 32
    hidden_dim = 4096
    ffn_dim    = 14336
    ne = NUM_EXPERTS

    hidden = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    router = torch.randn(num_tokens, ne, device="cuda")
    # Fake MXFP4 weights (packed uint8)
    w1_fp4    = torch.randint(0, 256, (ne, hidden_dim // 2, ffn_dim), device="cuda", dtype=torch.uint8)
    w1_scales = torch.ones(ne, hidden_dim // MXFP4_BLOCK_SIZE, ffn_dim, device="cuda")
    w2_fp4    = torch.randint(0, 256, (ne, ffn_dim // 2, hidden_dim), device="cuda", dtype=torch.uint8)
    w2_scales = torch.ones(ne, ffn_dim // MXFP4_BLOCK_SIZE, hidden_dim, device="cuda")

    out = moe_forward(hidden, router, w1_fp4, w1_scales, w2_fp4, w2_scales)
    print(f"input={hidden.shape}, output={out.shape}")
    print("Smoke test passed.")
