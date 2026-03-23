# AMD x GPU MODE — E2E Model Speedrun

Team submission for the [AMD x GPU MODE E2E Model Speedrun](https://lu.ma/cqq4mojz) competition.

- **Phase 1 deadline:** April 6, 2026
- **Hardware:** AMD Instinct MI355X (CDNA3)
- **Prize pool:** $1.1M total — top 10 Phase 1 teams advance to finals

## Phase 1 Challenges

| Folder | Challenge | Owner |
|--------|-----------|-------|
| [`mla-decode/`](mla-decode/) | MLA Decode (Multi-Head Latent Attention) | Person A |
| [`mxfp4-gemm/`](mxfp4-gemm/) | MXFP4 GEMM | Person B |
| [`mxfp4-moe/`](mxfp4-moe/) | MXFP4 MoE (Mixture of Experts) | Person B |

## Quick Start

```bash
# 1. Clone this repo on your GPU machine
git clone <repo-url>
cd AMD_optimization_challenge

# 2. Run the setup script
bash setup.sh

# 3. Verify the environment
python scripts/verify_env.py

# 4. Navigate to a challenge and start working
cd mla-decode
python reference.py      # run the baseline
python solution.py       # run your solution
python benchmark.py      # compare performance
```

## Submission

```bash
# Authenticate once
popcorn login

# Submit a challenge (run from the challenge folder)
cd mla-decode
popcorn submit solution.py --problem mla-decode
```

## Repository Structure

```
AMD_optimization_challenge/
├── setup.sh                  # one-shot environment setup
├── scripts/
│   ├── verify_env.py         # check ROCm, AITER, torch are working
│   └── benchmark_all.py      # run all 3 challenge benchmarks
├── mla-decode/
│   ├── README.md
│   ├── reference.py          # unmodified baseline from gpu-mode/reference-kernels
│   ├── solution.py           # our optimized kernel
│   └── benchmark.py          # perf comparison script
├── mxfp4-gemm/
│   ├── README.md
│   ├── reference.py
│   ├── solution.py
│   └── benchmark.py
└── mxfp4-moe/
    ├── README.md
    ├── reference.py
    ├── solution.py
    └── benchmark.py
```

## Key Resources

- [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) — official baselines and task specs
- [ROCm/aiter](https://github.com/ROCm/aiter) — AMD's optimized AI operators
- [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) — CK library for MXFP4 GEMM
- [AITER MLA decode tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)
- [vLLM ROCm attention backends](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)
- [MXFP4 quantization on AMD](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- GPU MODE Discord: `#amd-competition` channel

## Profiling Cheatsheet

```bash
# Quick kernel timing
rocprof --stats python solution.py

# Deep hardware counter analysis
omniperf profile --name run1 -- python solution.py
omniperf analyze -p workloads/run1/

# Memory bandwidth test
rocm-bandwidth-test
```
