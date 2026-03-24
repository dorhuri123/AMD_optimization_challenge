"""
Custom Triton MLA decode kernel with MXFP4 KV cache.

This is the high-effort, high-reward optimization path.
AITER does NOT support MXFP4 KV in mla_decode_fwd — we must write our own.

Architecture:
  MLA decode: q_seq_len=1, variable kv_seq_len (up to 8k)
  - Q: (total_q, 16, 576) bf16 or fp8
  - KV: (total_kv, 1, 576) stored as fp4x2 (288 bytes) + e8m0 scales
  - Output: (total_q, 16, 512) bf16

Key insights:
  1. Decode is memory-bound: Q is tiny (1 token), KV is huge (up to 8k tokens)
  2. MQA: 1 KV head shared across 16 Q heads — load KV once, compute 16 scores
  3. MXFP4 KV = 2x less HBM traffic than fp8, 4x less than bf16
  4. Dequant fp4x2 → bf16 in registers, never write back to HBM

Algorithm (per batch element):
  Phase 1 — Score computation:
    For each KV tile of BLOCK_KV tokens:
      Load KV tile (fp4x2) from HBM → dequant to bf16 in registers
      Load E8M0 scales for this tile
      For each of 16 Q heads:
        score[head, kv_pos] += dot(Q[head, :576], K[kv_pos, :576])
    Apply softmax across kv_seq_len for each head

  Phase 2 — Value aggregation:
    For each KV tile of BLOCK_KV tokens:
      Load KV tile (fp4x2) → dequant first 512 dims only (values)
      For each of 16 Q heads:
        output[head, :512] += softmax_score[head, kv_pos] * V[kv_pos, :512]

Status: SKELETON — needs implementation + testing on MI355X
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Architecture constants
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576   # full dim for keys
V_HEAD_DIM = 512    # first 512 dims for values
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
MXFP4_BLOCK_SIZE = 32  # elements per E8M0 scale


@triton.jit
def _unpack_fp4x2(packed: tl.tensor) -> tuple:
    """
    Unpack fp4x2 byte into two FP4 E2M1 values.
    Low nibble = even index, high nibble = odd index.
    Returns (even_val, odd_val) as float32.

    E2M1 encoding (4 bits):
      sign(1) | exp(2) | mantissa(1)
      Values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    """
    # Extract nibbles
    lo = (packed & 0x0F).to(tl.float32)
    hi = ((packed >> 4) & 0x0F).to(tl.float32)

    # TODO: proper E2M1 decode — for now this is a placeholder
    # The actual decode needs: sign extraction, exponent bias, mantissa handling
    # Will use aiter's mxfp4_to_f32 as reference for the bit manipulation
    return lo, hi


@triton.jit
def _apply_e8m0_scale(vals: tl.tensor, scale_byte: tl.tensor) -> tl.tensor:
    """
    Apply E8M0 block scale: scale = 2^(scale_byte - 127)
    E8M0 is exponent-only: 8-bit unsigned int interpreted as power-of-2.
    """
    # E8M0: value = 2^(byte_val - 127)
    exponent = scale_byte.to(tl.float32) - 127.0
    scale = tl.math.exp2(exponent)
    return vals * scale


@triton.jit
def mla_decode_mxfp4_kernel(
    # Q tensor: (total_q, NUM_HEADS, QK_HEAD_DIM) bf16
    Q_ptr,
    # KV fp4x2: (total_kv, 1, QK_HEAD_DIM // 2) uint8
    KV_fp4_ptr,
    # KV scales: (total_kv * num_scale_blocks,) uint8  [flattened e8m0]
    KV_scale_ptr,
    # Output: (total_q, NUM_HEADS, V_HEAD_DIM) bf16
    O_ptr,
    # Indptrs
    qo_indptr_ptr,
    kv_indptr_ptr,
    # Dimensions
    total_q: tl.constexpr,
    num_heads: tl.constexpr,
    qk_dim: tl.constexpr,       # 576
    v_dim: tl.constexpr,         # 512
    sm_scale: tl.constexpr,
    # Block sizes
    BLOCK_KV: tl.constexpr,      # KV tokens per tile
    BLOCK_D: tl.constexpr,       # dimension tile size
):
    """
    MLA decode attention with MXFP4 KV cache.

    Grid: (total_q, num_heads)
    Each program instance handles one query token × one head.

    TODO: This is a skeleton — the actual implementation needs:
    1. Proper indptr-based variable-length batching
    2. Online softmax (numerically stable, streaming)
    3. Correct MXFP4 E2M1 unpacking
    4. Split-K for long sequences
    5. MQA optimization: share KV loads across heads in the same batch element
    """
    # Program indices
    pid_q = tl.program_id(0)    # which query token
    pid_h = tl.program_id(1)    # which head

    # TODO: look up batch element from qo_indptr to get kv_start, kv_end
    # For now this is a placeholder
    pass


def triton_mla_decode_mxfp4(
    q: torch.Tensor,           # (total_q, 16, 576) bf16
    kv_fp4: torch.Tensor,      # (total_kv, 1, 288) fp4x2/uint8
    kv_scale: torch.Tensor,    # (rows, blocks) e8m0/uint8
    qo_indptr: torch.Tensor,   # (batch+1,) int32
    kv_indptr: torch.Tensor,   # (batch+1,) int32
    config: dict,
) -> torch.Tensor:
    """
    Python wrapper for the Triton MLA decode with MXFP4 KV.
    """
    total_q = q.shape[0]
    o = torch.empty((total_q, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    # TODO: launch kernel with proper grid and block sizes
    # grid = (total_q, NUM_HEADS)
    # mla_decode_mxfp4_kernel[grid](...)

    raise NotImplementedError(
        "Triton MXFP4 MLA decode kernel not yet implemented. "
        "Run sweep_kv_splits.py first for quick wins, then implement this."
    )

    return o


# ---------------------------------------------------------------------------
# Integration with submission.py
# ---------------------------------------------------------------------------
def custom_kernel_mxfp4(data: input_t) -> output_t:
    """
    MLA decode using custom Triton kernel with MXFP4 KV cache.
    Drop-in replacement for custom_kernel in submission.py.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp4, kv_scale = kv_data["mxfp4"]
    return triton_mla_decode_mxfp4(q, kv_fp4, kv_scale, qo_indptr, kv_indptr, config)
