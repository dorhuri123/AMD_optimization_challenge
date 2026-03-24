# AMD x GPU MODE — E2E Model Speedrun

Team submission for the [AMD x GPU MODE E2E Model Speedrun](https://lu.ma/cqq4mojz) competition.

- **Phase 1 deadline:** April 6, 2026
- **Hardware:** AMD Instinct MI355X (CDNA3)
- **Prize pool:** $1.1M total — top 10 Phase 1 teams advance to finals
- **Official challenge folder:** [gpu-mode/reference-kernels/problems/amd_202602](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602)

## Phase 1 Challenges

| Folder | Challenge | Model | Owner |
| --- | --- | --- | --- |
| [`mixed-mla/`](mixed-mla/) | MLA Decode (mixed FP8/MXFP4 KV) | DeepSeek R1 | Person A |
| [`mxfp4-mm/`](mxfp4-mm/) | MXFP4 Matrix Multiply | — | Person B |
| [`moe-mxfp4/`](moe-mxfp4/) | MXFP4 MoE Fused Kernel | DeepSeek R1 | Person B |

Each challenge: implement `custom_kernel(data)` in `submission.py` to beat the AITER reference.

## Quick Start

```bash
# 1. Clone this repo on your GPU machine
git clone git@github.com:dorhuri123/AMD_optimization_challenge.git
cd AMD_optimization_challenge

# 2. Run the setup script (installs ROCm deps, AITER, popcorn-cli, copies reference files)
bash setup.sh

# 3. Verify the environment
python scripts/verify_env.py

# 4. Work on a challenge
cd mixed-mla
python reference.py    # run the AITER baseline
python benchmark.py    # compare reference vs your submission.py
```

## Submission

```bash
popcorn login   # once

# Submit from inside a challenge folder
cd mixed-mla
popcorn submit submission.py --problem mixed-mla

cd mxfp4-mm
popcorn submit submission.py --problem mxfp4-mm

cd moe-mxfp4
popcorn submit submission.py --problem moe-mxfp4
```

## Repository Structure

```
AMD_optimization_challenge/
├── setup.sh                   # one-shot environment setup + copies reference files
├── eval.py                    # fetched by setup.sh — official eval harness
├── utils.py                   # fetched by setup.sh — shared test utilities
├── scripts/
│   ├── verify_env.py          # check ROCm, AITER, FP8, popcorn-cli
│   └── benchmark_all.py       # run all 3 challenge benchmarks
├── mixed-mla/
│   ├── README.md              # architecture details + optimization checklist
│   ├── reference.py           # fetched by setup.sh — AITER a8w8 baseline (do not modify)
│   ├── task.py                # fetched by setup.sh — input_t / output_t / TestSpec
│   ├── task.yml               # fetched by setup.sh — benchmark configs
│   ├── submission.py          # ← EDIT THIS — custom_kernel() implementation
│   └── benchmark.py           # ref vs submission latency table
├── mxfp4-mm/
│   ├── README.md
│   ├── reference.py           # fetched by setup.sh
│   ├── task.py                # fetched by setup.sh
│   ├── task.yml               # fetched by setup.sh
│   ├── submission.py          # ← EDIT THIS
│   └── benchmark.py
└── moe-mxfp4/
    ├── README.md
    ├── reference.py           # fetched by setup.sh
    ├── task.py                # fetched by setup.sh
    ├── task.yml               # fetched by setup.sh
    ├── submission.py          # ← EDIT THIS
    └── benchmark.py
```

> `reference.py`, `task.py`, `task.yml`, `eval.py`, `utils.py` are gitignored — fetched from
> [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) by `setup.sh`.

## Key Resources

- [Official challenge folder](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602)
- [ROCm/aiter](https://github.com/ROCm/aiter) — AMD's optimized AI operators (AITER)
- [AITER MLA decode tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)
- [vLLM ROCm attention backends](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)
- [MXFP4 quantization on AMD](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [vLLM PR #36297](https://github.com/vllm-project/vllm/pull/36297) — fused BMM+FP8 for MLA (5.2–6.6x on MI300X)
- GPU MODE Discord: `#amd-competition` channel

## Profiling Cheatsheet

```bash
# Quick kernel timing
rocprof --stats python submission.py

# Deep hardware counter analysis
omniperf profile --name run1 -- python submission.py
omniperf analyze -p workloads/run1/

# Memory bandwidth baseline
rocm-bandwidth-test
```
