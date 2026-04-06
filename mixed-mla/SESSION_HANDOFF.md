# Mixed-MLA Session Handoff — 2026-04-05

## Current State
- **submission.py** = v24 (best leaderboard: **47.521μs** geomean)
- **Deadline**: April 7, 2026
- **Top 10 target**: ~33μs, #1 josusanmartin: 19.484μs
- **Team**: FP32, username: dorhuri123

## Versions Ready for Testing

### Practical (AITER-based, no compilation)
| Version | Key Change |
|---------|-----------|
| v24 (current) | 3-way hybrid: MXFP4+a16w8+a8w8, 47.5μs |
| v32 | v24 + env vars + splits=32 for large |
| v33 | v32 + kv_granularity=64 for large |

### Experimental (HIP MFMA)
| Version | Status |
|---------|--------|
| hip_mxfp4_v2 | Written, untested. Multi-wave + scalar V |
| hip_mxfp4_v3 | Being built. FP8 Q + MXFP4 K + bf16 MFMA PV |

## Key Findings
- kv_granularity=64 may help large config (kernelsanders insight)
- Custom HIP kernels risk 17-min compilation timeout on eval
- FP8 Q x MXFP4 K (cbsz=0, blgp=4) gives much better accuracy
- bf16 MFMA for PV phase is the key to making custom kernel fast

## Submit: `popcorn submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/submission.py`
