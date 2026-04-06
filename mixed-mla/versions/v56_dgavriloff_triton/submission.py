"""
v56: dgavriloff-style custom Triton flash-decode for ALL 8 configs.

Replicates dgavriloff's v114 architecture with our optimizations:
  - Stage 1: Per-head flash-decode kernel
    Grid: (bs * NSPLIT * 16,)
    - Q stays bf16, padded to QK_PAD=1024 (one load, no tile loop)
    - K is MXFP4, padded to PKD_PAD=512 packed / NSC_PAD=32 scales
    - dot_scaled: bf16 Q [1, QK_PAD] x e2m1 K [PKD_PAD, BN] -> [1, BN]
    - V from FP8 cache, full 512 dims loaded per tile
    - Online softmax, stores LSE = m + log(l)
    - Mid_v stored as bf16 (normalized) for bandwidth

  - Stage 2: Reduce across splits
    Grid: (bs, 16) — one program per (batch, head)
    Combines NSPLIT partial results via exp(lse - max_lse) weighting.

  - NSPLIT per-config tuning for optimal performance
  - Buffer caching to avoid allocation overhead
  - BN=64, num_warps=4, num_stages=2
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

# ===============================================================
# CONSTANTS
# ===============================================================

NUM_HEADS: int = 16
QK_HEAD_DIM: int = 576
V_HEAD_DIM: int = 512
SM_SCALE: float = 1.0 / (QK_HEAD_DIM ** 0.5)

# Padded dims for dot_scaled (must be powers of 2)
QK_PAD: int = 1024     # pad 576 -> 1024
PKD_PAD: int = 512      # pad 288 -> 512
NSC_PAD: int = 32       # pad 18 -> 32

# ===============================================================
# NSPLIT ROUTING — tuned per config
# ===============================================================

NSPLIT_TABLE = {
    (4, 1024):    8,
    (32, 1024):   8,
    (64, 1024):   8,
    (256, 1024):  8,
    (4, 8192):    16,
    (32, 8192):   16,
    (64, 8192):   16,
    (256, 8192):  16,
}

# ===============================================================
# CACHES
# ===============================================================

_buf_cache: dict = {}
_output_cache: dict = {}


# ===============================================================
# STAGE 1: Per-head flash-decode
# ===============================================================

@triton.jit
def _stage1(
    Q,              # [total_q, NUM_HEADS, QK_DIM] bf16
    KV_pk,          # [total_kv, PKD] uint8 packed MXFP4
    KV_sc,          # [total_kv, N_SC] uint8 e8m0 scales
    KV_fp8,         # [total_kv, FP8_STRIDE] fp8
    kv_indptr,      # [batch+1] int32
    qo_indptr,      # [batch+1] int32
    Mid_v,          # [total_q * NSPLIT * NHEADS * V_DIM] bf16 (flat)
    Mid_lse,        # [total_q * NSPLIT * NHEADS] float32 (flat)
    sm_scale,
    fp8_scale,
    NSPLIT: tl.constexpr,
    BN: tl.constexpr,
    NHEADS: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    FP8_STRIDE: tl.constexpr,
    PKD: tl.constexpr,
    N_SC: tl.constexpr,
    QK_P: tl.constexpr,
    PKD_P: tl.constexpr,
    NSC_P: tl.constexpr,
):
    pid = tl.program_id(0)
    hid = pid % NHEADS
    pid2 = pid // NHEADS
    sid = pid2 % NSPLIT
    bid = pid2 // NSPLIT

    kv_s = tl.load(kv_indptr + bid)
    kv_e = tl.load(kv_indptr + bid + 1)
    kv_len = kv_e - kv_s
    qi = tl.load(qo_indptr + bid)

    per_split = tl.cdiv(kv_len, NSPLIT)
    ss = sid * per_split
    se = tl.minimum(ss + per_split, kv_len)

    lse_off = (qi * NSPLIT + sid) * NHEADS + hid
    v_off = lse_off * V_DIM

    if ss >= kv_len:
        tl.store(Mid_lse + lse_off, float("-inf"))
        ov = tl.arange(0, V_DIM)
        tl.store(Mid_v + v_off + ov, tl.zeros([V_DIM], dtype=tl.bfloat16))
        return

    # Load Q for this head: [QK_P] bf16 (padded from QK_DIM=576 to QK_P=1024)
    ok = tl.arange(0, QK_P)
    q_vec = tl.load(Q + qi * NHEADS * QK_DIM + hid * QK_DIM + ok,
                     mask=ok < QK_DIM, other=0.0)

    m_i = float("-inf")
    l_i = 0.0
    acc_v = tl.zeros([V_DIM], dtype=tl.float32)

    for blk_s in range(ss, se, BN):
        bn = tl.minimum(BN, se - blk_s)
        on = tl.arange(0, BN)
        nm = on < bn
        kg = kv_s + blk_s + on

        # K packed [BN, PKD_P], K scale [BN, NSC_P]
        opk = tl.arange(0, PKD_P)
        kp = tl.load(KV_pk + kg[:, None] * PKD + opk[None, :],
                      mask=nm[:, None] & (opk[None, :] < PKD), other=0)
        osc = tl.arange(0, NSC_P)
        ks = tl.load(KV_sc + kg[:, None] * N_SC + osc[None, :],
                      mask=nm[:, None] & (osc[None, :] < N_SC), other=127)

        # Score: [1, QK_P] @ [PKD_P, BN] -> [1, BN]
        q_2d = q_vec[None, :]
        kpt = tl.trans(kp)
        scores_2d = tl.dot_scaled(q_2d.to(tl.bfloat16), None, "bf16",
                                   kpt, ks, "e2m1")
        scores = tl.reshape(scores_2d, [BN])
        scores = scores * sm_scale
        scores = tl.where(nm, scores, float("-inf"))

        # Online softmax
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc_v = acc_v * alpha

        # V from FP8: load [BN, V_DIM], dequant
        ov = tl.arange(0, V_DIM)
        fp8_offsets = (kg[:, None] * FP8_STRIDE + ov[None, :]).to(tl.int64)
        vfp8 = tl.load(KV_fp8 + fp8_offsets,
                        mask=nm[:, None], other=0.0)
        v_f32 = vfp8.to(tl.float32) * fp8_scale

        # V accum: p[BN] @ V[BN, V_DIM] -> [V_DIM]
        acc_v += tl.sum(p[:, None] * v_f32, axis=0)
        m_i = m_new

    inv_l = 1.0 / l_i
    tl.store(Mid_lse + lse_off, m_i + tl.log(l_i))

    ov = tl.arange(0, V_DIM)
    tl.store(Mid_v + v_off + ov, (acc_v * inv_l).to(tl.bfloat16))


# ===============================================================
# STAGE 2: Reduce across splits
# ===============================================================

@triton.jit
def _stage2(
    Mid_v, Mid_lse, O, qo_indptr,
    NSPLIT: tl.constexpr,
    NHEADS: tl.constexpr,
    V_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    qi = tl.load(qo_indptr + bid)

    mx = float("-inf")
    for s in tl.static_range(NSPLIT):
        lse = tl.load(Mid_lse + (qi * NSPLIT + s) * NHEADS + hid)
        mx = tl.maximum(mx, lse)

    ov = tl.arange(0, V_DIM)
    acc = tl.zeros([V_DIM], dtype=tl.float32)
    lsum = 0.0
    for s in tl.static_range(NSPLIT):
        lse = tl.load(Mid_lse + (qi * NSPLIT + s) * NHEADS + hid)
        w = tl.exp(lse - mx)
        lsum += w
        base = ((qi * NSPLIT + s) * NHEADS + hid) * V_DIM
        v_bf16 = tl.load(Mid_v + base + ov)
        acc += w * v_bf16.to(tl.float32)

    inv = 1.0 / lsum
    o_base = qi * NHEADS * V_DIM + hid * V_DIM
    tl.store(O + o_base + ov, (acc * inv).to(tl.bfloat16))


# ===============================================================
# BUFFER MANAGEMENT
# ===============================================================

def _get_buffers(total_q, nsplit, device):
    key = (total_q, nsplit)
    if key not in _buf_cache:
        n_mid = total_q * nsplit * NUM_HEADS
        _buf_cache[key] = {
            "mid_v": torch.empty(n_mid * V_HEAD_DIM, dtype=torch.bfloat16, device=device),
            "mid_lse": torch.empty(n_mid, dtype=torch.float32, device=device),
        }
    return _buf_cache[key]


def _get_output(total_q, device):
    if total_q not in _output_cache:
        _output_cache[total_q] = torch.empty(
            (total_q, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=device,
        )
    return _output_cache[total_q]


# ===============================================================
# ENTRY POINT
# ===============================================================

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    tq = q.shape[0]

    # MXFP4 for K (dot_scaled)
    kv_pk_raw, kv_sc_raw = kv_data["mxfp4"]
    kv_pk = kv_pk_raw.view(torch.uint8) if kv_pk_raw.dtype != torch.uint8 else kv_pk_raw
    kv_sc = kv_sc_raw.view(torch.uint8) if kv_sc_raw.dtype != torch.uint8 else kv_sc_raw
    if kv_pk.dim() > 2:
        kv_pk = kv_pk.reshape(kv_pk.shape[0], -1)
    if kv_sc.dim() > 2:
        kv_sc = kv_sc.reshape(kv_sc.shape[0], -1)

    # FP8 for V
    kv_fp8_raw, kv_fp8_scale = kv_data["fp8"]
    kv_fp8 = kv_fp8_raw.view(torch.float8_e4m3fnuz).reshape(kv_fp8_raw.shape[0], -1)
    fp8_scale_val = kv_fp8_scale.item() if kv_fp8_scale.numel() == 1 else 1.0

    PKD = kv_pk.shape[-1]
    N_SC = kv_sc.shape[-1]
    FP8_STRIDE = kv_fp8.shape[-1]

    # NSPLIT from tuned table
    kvlen = config["kv_seq_len"]
    NSPLIT = NSPLIT_TABLE.get((bs, kvlen), 16)
    BN = 64

    # Get cached buffers
    bufs = _get_buffers(tq, NSPLIT, q.device)
    o = _get_output(tq, q.device)

    _stage1[(bs * NSPLIT * NUM_HEADS,)](
        q.view(-1, NUM_HEADS, QK_HEAD_DIM), kv_pk, kv_sc, kv_fp8,
        kv_indptr, qo_indptr,
        bufs["mid_v"], bufs["mid_lse"],
        SM_SCALE, fp8_scale_val,
        NSPLIT=NSPLIT, BN=BN, NHEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM, V_DIM=V_HEAD_DIM,
        FP8_STRIDE=FP8_STRIDE,
        PKD=PKD, N_SC=N_SC,
        QK_P=QK_PAD, PKD_P=PKD_PAD, NSC_P=NSC_PAD,
        num_warps=4, num_stages=2,
    )

    _stage2[(bs, NUM_HEADS)](
        bufs["mid_v"], bufs["mid_lse"], o, qo_indptr,
        NSPLIT=NSPLIT, NHEADS=NUM_HEADS, V_DIM=V_HEAD_DIM,
        num_warps=4,
    )

    return o
