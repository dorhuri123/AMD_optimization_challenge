# Phase 4a Status

## MFMA Compilation: SUCCESS
The inline ASM `v_mfma_f32_16x16x128_f8f6f4` compiles and executes on MI355X (gfx950).

## Correctness: FAILS on small configs
- bs=4, kv=1024: 7409 mismatched elements (wrong values, wrong signs)
- bs=32, kv=1024: 57893 mismatched elements
- bs=64, kv=8192: PASSES (uses AITER fallback)
- bs=256, kv=8192: PASSES (uses AITER fallback)

## Root Cause: MFMA register layout bug
The data loading into A/B registers doesn't match the hardware's expected layout.
The thread-to-element mapping for v_mfma_f32_16x16x128_f8f6f4 needs to be verified
against the AMD matrix instruction calculator.

## What Works
- load_inline compiles HIP C++ on MI355X ✓
- v_mfma_f32_16x16x128_f8f6f4 inline ASM assembles ✓
- The instruction executes without crashing ✓
- AITER fallback works correctly ✓

## Next Steps
1. Study AMD matrix instruction calculator for exact register layout
2. Fix the A/B register loading in the MFMA kernel
3. Retest with corrected mapping
