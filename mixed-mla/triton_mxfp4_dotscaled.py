"""
Triton MXFP4 MLA decode kernel using tl.dot_scaled (gfx950 only).

Uses hardware-accelerated MXFP4 dot product for QK scores.
V is dequantized from MXFP4 to bf16 and uses regular tl.dot.

Key design:
  - Q quantized to MXFP4 via dynamic_mxfp4_quant (AITER utility)
  - K in MXFP4 (from kv_data["mxfp4"])
  - tl.dot_scaled(Q_fp4, Q_scale, "e2m1", K_fp4, K_scale, "e2m1")
  - BLOCK_M = 16 (query heads — perfect 16×16 MFMA tile)
  - Split-K for decode parallelism
  - V manually dequantized to bf16, accumulated via tl.dot

Data layout for tl.dot_scaled:
  - Q (lhs): [BLOCK_M, PADDED_D//2] uint8 — fp4x2 packed
  - Q_scale: [BLOCK_M, D//32] uint8 — E8M0 per-block-of-32
  - K (rhs): [PADDED_D//2, BLOCK_N] uint8 — transposed! fp4x2 packed
  - K_scale: [BLOCK_N, D//32] uint8 — E8M0 per-block-of-32
"""

import torch
import triton
import triton.language as tl

NUM_HEADS = 16
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

# Pad to next power of 2 for MFMA compatibility
PADDED_QK_DIM = 1024  # 576 → 1024 (next power of 2)
SCALE_GROUP = 32
NUM_QK_SCALES = QK_HEAD_DIM // SCALE_GROUP  # 576/32 = 18
NUM_V_SCALES = V_HEAD_DIM // SCALE_GROUP    # 512/32 = 16
PADDED_QK_SCALES = PADDED_QK_DIM // SCALE_GROUP  # 32

# Split-K tuning
MXFP4_KV_SPLITS_MAP = {
    (4, 1024): 4,
    (4, 8192): 16,
    (32, 1024): 4,
    (32, 8192): 16,
    (64, 1024): 8,
    (64, 8192): 16,
    (256, 1024): 4,
    (256, 8192): 8,
}

_buf_cache = {}


@triton.jit
def _e2m1_dequant(nibble):
    """Decode 4-bit E2M1 to float32."""
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    mant = nibble & 1
    exp_f = exp.to(tl.float32)
    mant_f = mant.to(tl.float32)
    normal_val = tl.math.exp2(exp_f - 1.0) * (1.0 + 0.5 * mant_f)
    denorm_val = 0.5 * mant_f
    val = tl.where(exp == 0, denorm_val, normal_val)
    val = tl.where(sign != 0, -val, val)
    return val


@triton.jit
def _mla_mxfp4_stage1(
    Q_fp4_ptr,        # [batch, 16, QK_DIM//2] uint8 — fp4x2 packed
    Q_scale_ptr,      # [batch*16, QK_DIM//32] uint8 — E8M0 scales
    KV_fp4_ptr,       # [total_kv, QK_DIM//2] uint8 — fp4x2 packed
    KV_scale_ptr,     # [total_kv, QK_DIM//32] uint8 — E8M0 scales
    Partial_O_ptr,    # [batch, splits, heads, V_DIM] f32
    Partial_m_ptr,    # [batch, splits, heads] f32
    Partial_l_ptr,    # [batch, splits, heads] f32
    kv_indptr_ptr,    # [batch+1] i32
    stride_qf_batch,  # Q fp4 batch stride
    stride_qf_head,   # Q fp4 head stride
    stride_qs_row,    # Q scale row stride (batch*16 rows)
    stride_kvf_tok,   # KV fp4 token stride
    stride_kvs_tok,   # KV scale token stride
    stride_po_b, stride_po_s, stride_po_h,
    stride_ml_b, stride_ml_s, stride_ml_h,
    BLOCK_N: tl.constexpr,        # KV tokens per tile (e.g., 64)
    V_BLOCK_D: tl.constexpr,      # V dim tile (e.g., 64)
    NUM_KV_SPLITS: tl.constexpr,
    PADDED_D_HALF: tl.constexpr,  # PADDED_QK_DIM // 2 = 512
    ACTUAL_D_HALF: tl.constexpr,  # QK_HEAD_DIM // 2 = 288
    NUM_SCALES: tl.constexpr,     # QK_HEAD_DIM // 32 = 18
    PADDED_SCALES: tl.constexpr,  # PADDED_QK_DIM // 32 = 32
):
    """
    Stage 1 using tl.dot_scaled for QK scores.

    Grid: (batch * NUM_KV_SPLITS, 1)
    Each program:
      - Loads Q for all 16 heads: [16, PADDED_D_HALF] fp4x2
      - For each KV tile: dot_scaled → [16, BLOCK_N] scores
      - Online softmax
      - Dequant V tile from MXFP4 → bf16, accumulate via tl.dot
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    V_DIM: tl.constexpr = 512
    BLOCK_M: tl.constexpr = 16  # heads

    pid_bs = tl.program_id(0)
    pid_b = pid_bs // NUM_KV_SPLITS
    pid_s = pid_bs % NUM_KV_SPLITS

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    kv_len = kv_end - kv_start

    split_size = tl.cdiv(kv_len, NUM_KV_SPLITS)
    split_kv_start = pid_s * split_size
    split_kv_end = tl.minimum(split_kv_start + split_size, kv_len)

    # Load Q fp4x2 for all 16 heads: [16, PADDED_D_HALF]
    heads_offs = tl.arange(0, BLOCK_M)  # 0..15
    d_half_offs = tl.arange(0, PADDED_D_HALF)  # 0..511

    q_base = Q_fp4_ptr + pid_b * stride_qf_batch
    q_mask = d_half_offs[None, :] < ACTUAL_D_HALF
    q_fp4 = tl.load(
        q_base + heads_offs[:, None] * stride_qf_head + d_half_offs[None, :],
        mask=q_mask, other=0
    )

    # Load Q scales: [16, PADDED_SCALES]
    qs_offs = tl.arange(0, PADDED_SCALES)
    qs_mask = qs_offs[None, :] < NUM_SCALES
    q_scale = tl.load(
        Q_scale_ptr + (pid_b * BLOCK_M + heads_offs[:, None]) * stride_qs_row + qs_offs[None, :],
        mask=qs_mask, other=127  # 127 = scale of 1.0 in E8M0
    )

    # Initialize online softmax state
    m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    # V accumulator: [BLOCK_M, V_DIM] — process in V_BLOCK_D chunks
    # We'll use 8 separate [16, V_BLOCK_D] chunks stored sequentially
    NUM_V_BLOCKS: tl.constexpr = V_DIM // V_BLOCK_D

    acc0 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)
    acc7 = tl.zeros([BLOCK_M, V_BLOCK_D], dtype=tl.float32)

    num_tiles = tl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    kv_n_offs = tl.arange(0, BLOCK_N)

    for tile_idx in range(num_tiles):
        tile_start = split_kv_start + tile_idx * BLOCK_N
        kv_offsets = tile_start + kv_n_offs
        mask_kv = kv_offsets < split_kv_end
        kv_idx = kv_start + kv_offsets

        # Load K fp4x2 transposed: [PADDED_D_HALF, BLOCK_N]
        # K is stored row-major as [total_kv, D_HALF], we load transposed
        k_mask = (d_half_offs[:, None] < ACTUAL_D_HALF) & mask_kv[None, :]
        k_fp4 = tl.load(
            KV_fp4_ptr + kv_idx[None, :] * stride_kvf_tok + d_half_offs[:, None],
            mask=k_mask, other=0
        )

        # Load K scales: [BLOCK_N, PADDED_SCALES]
        k_scale_mask = mask_kv[:, None] & (qs_offs[None, :] < NUM_SCALES)
        k_scale = tl.load(
            KV_scale_ptr + kv_idx[:, None] * stride_kvs_tok + qs_offs[None, :],
            mask=k_scale_mask, other=127
        )

        # === Hardware MXFP4 dot: scores = Q[16, D] @ K[D, BLOCK_N] ===
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        scores = tl.dot_scaled(
            q_fp4, q_scale, "e2m1",
            k_fp4, k_scale, "e2m1",
            fast_math=True, acc=scores
        )

        # Mask invalid positions
        scores = tl.where(mask_kv[None, :], scores, float("-inf"))

        # NOTE: sm_scale is NOT applied here because dot_scaled already
        # dequantizes to true values. We need to apply sm_scale separately.
        # Actually: dot_scaled computes Q_true @ K_true where
        # Q_true = dequant(Q_fp4, Q_scale), K_true = dequant(K_fp4, K_scale)
        # So scores = Q_true @ K_true, and we need scores * sm_scale

        # --- Online softmax ---
        # Apply sm_scale via log2: exp2(scores * sm_scale * LOG2E)
        # = exp2((scores * LOG2E) * sm_scale)
        # For numerical stability with online softmax:
        m_ij = tl.max(scores, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_prev, m_ij)
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)

        # We need exp(scores * sm_scale - m_new * sm_scale)
        # But our m values track the raw scores (without sm_scale).
        # Let's track scaled scores instead:
        # Actually, let me rethink. The standard approach:
        # p = exp((score - m) * sm_scale) -- WRONG
        # p = exp(score * sm_scale - m)  where m = max(score * sm_scale)
        # So let's scale first:

        # Scale all scores by sm_scale
        # (this is fine since we haven't computed m yet per-tile... wait, we already computed m_ij above)
        # Let me redo this properly:

        m_ij_scaled = m_ij  # m_ij is max of raw scores, we track raw
        m_new = tl.maximum(m_prev, m_ij_scaled)
        alpha = tl.math.exp2((m_prev - m_new) * LOG2E)

        p = tl.math.exp2((scores - m_new[:, None]) * LOG2E)
        p = tl.where(mask_kv[None, :], p, 0.0)

        l_ij = tl.sum(p, axis=1)
        l_prev = l_prev * alpha + l_ij
        m_prev = m_new

        # Rescale accumulators
        acc0 *= alpha[:, None]; acc1 *= alpha[:, None]
        acc2 *= alpha[:, None]; acc3 *= alpha[:, None]
        acc4 *= alpha[:, None]; acc5 *= alpha[:, None]
        acc6 *= alpha[:, None]; acc7 *= alpha[:, None]

        # === Accumulate V: p[16, BLOCK_N] @ V[BLOCK_N, V_BLOCK_D] ===
        # Dequantize V from MXFP4 to bf16 per chunk
        p_bf16 = p.to(tl.bfloat16)

        for v_blk in tl.static_range(NUM_V_BLOCKS):
            vd_start = v_blk * V_BLOCK_D
            vd_half_start = vd_start // 2
            vd_scale_start = vd_start // SCALE_GROUP
            vd_half_offs = vd_half_start + tl.arange(0, V_BLOCK_D // 2)
            vd_scale_offs = vd_scale_start + tl.arange(0, V_BLOCK_D // SCALE_GROUP)

            # Load V fp4x2 packed: [BLOCK_N, V_BLOCK_D//2]
            v_packed = tl.load(
                KV_fp4_ptr + kv_idx[:, None] * stride_kvf_tok + vd_half_offs[None, :],
                mask=mask_kv[:, None], other=0
            )

            # Unpack fp4x2: lo nibble and hi nibble
            lo = _e2m1_dequant(v_packed & 0x0F)  # [BLOCK_N, V_BLOCK_D//2]
            hi = _e2m1_dequant((v_packed >> 4) & 0x0F)

            # Load V scales: [BLOCK_N, V_BLOCK_D//32]
            v_scales_raw = tl.load(
                KV_scale_ptr + kv_idx[:, None] * stride_kvs_tok + vd_scale_offs[None, :],
                mask=mask_kv[:, None], other=127
            )
            v_scale_f32 = tl.math.exp2(v_scales_raw.to(tl.float32) - 127.0)

            # Apply block-wise scales
            NUM_VSCALE_BLOCKS: tl.constexpr = V_BLOCK_D // SCALE_GROUP
            HALF_SCALE_GROUP: tl.constexpr = SCALE_GROUP // 2  # 16

            lo_blocked = tl.reshape(lo, [BLOCK_N, NUM_VSCALE_BLOCKS, HALF_SCALE_GROUP])
            hi_blocked = tl.reshape(hi, [BLOCK_N, NUM_VSCALE_BLOCKS, HALF_SCALE_GROUP])
            v_scale_exp = v_scale_f32[:, :, None]

            lo_scaled = tl.reshape(lo_blocked * v_scale_exp, [BLOCK_N, V_BLOCK_D // 2])
            hi_scaled = tl.reshape(hi_blocked * v_scale_exp, [BLOCK_N, V_BLOCK_D // 2])

            # Interleave lo and hi to reconstruct [BLOCK_N, V_BLOCK_D] bf16
            # lo has even indices, hi has odd indices
            # We need to create [BLOCK_N, V_BLOCK_D] from alternating lo/hi
            # Actually: fp4x2 packing is lo=byte&0xF, hi=byte>>4
            # In the original data: byte contains [elem_2i, elem_2i+1]
            # lo = elem_2i (even index), hi = elem_2i+1 (odd index)

            # Compute weighted sum using p: p[16, BLOCK_N] @ V[BLOCK_N, V_BLOCK_D]
            # We compute it as: p @ lo_scaled + p @ hi_scaled, properly indexed

            # For V, we need interleaved [BLOCK_N, V_BLOCK_D] where:
            # v[:, 0] = lo_scaled[:, 0], v[:, 1] = hi_scaled[:, 0], v[:, 2] = lo_scaled[:, 1], ...

            # Rather than fully interleaving, compute:
            # acc_even = p @ lo_scaled  — contributes to even output dims
            # acc_odd = p @ hi_scaled   — contributes to odd output dims
            weighted_even = tl.dot(p_bf16, lo_scaled.to(tl.bfloat16), out_dtype=tl.float32)
            weighted_odd = tl.dot(p_bf16, hi_scaled.to(tl.bfloat16), out_dtype=tl.float32)

            # TODO: need to interleave even/odd into the accumulator
            # For now, store even and odd separately and interleave at reduce stage
            # Actually this doesn't work with our accumulator layout...

            # Alternative: just do element-wise multiply-reduce (slower but correct)
            # p[:, None] * v_tile → sum over dim 0
            # This is the same approach as the original triton_mla_mxfp4.py

            # Full V tile in f32 (interleaved):
            # Since lo_scaled = even elements and hi_scaled = odd elements,
            # we need to reconstruct the full V_BLOCK_D-dim vector.
            # But tl.dot requires 2D inputs and can't easily interleave.

            # Simplest correct approach: element-wise
            # p_expanded = p[:, :, None]  — can't do 3D in triton easily
            # Let's use the vector reduce approach:
            # For each output dim d: acc[h, d] += sum_n(p[h, n] * v[n, d])
            # = p[h, :] @ v[:, d]

            # With lo_scaled[BLOCK_N, V_BLOCK_D//2] and hi_scaled[BLOCK_N, V_BLOCK_D//2]:
            # acc_even[16, V_BLOCK_D//2] += p @ lo_scaled
            # acc_odd[16, V_BLOCK_D//2] += p @ hi_scaled
            # Then interleave at the end

            # This works! Store as [16, V_BLOCK_D] where even/odd are interleaved
            # at store time.

            if v_blk == 0:
                acc0_even = weighted_even
                acc0_odd = weighted_odd
                # Actually, we already defined acc0..7 as [BLOCK_M, V_BLOCK_D]
                # Let me restructure...
                pass

            # ... This is getting too complex. Let me simplify.
            pass

    # This kernel needs restructuring for the V interleave issue.
    # TODO: implement properly
    pass


# --- Simpler approach: don't use dot_scaled for QK, just use MXFP4 data
# with manual dequant for 2x bandwidth savings ---

def mxfp4_decode_simple(q, kv_fp4, kv_scale, kv_indptr, config):
    """
    Placeholder for MXFP4 decode.
    Uses MXFP4 KV cache for 2x bandwidth savings.
    V is dequantized to bf16 for dot product.
    """
    # TODO: implement
    pass
