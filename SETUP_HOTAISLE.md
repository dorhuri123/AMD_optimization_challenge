# Hot Aisle MI300X — Quick Setup

Run this **every time** you spin up a fresh Hot Aisle VM. Takes ~8 minutes (mostly AITER JIT compilation).

**SSH:** `ssh hotaisle@<YOUR_IP> -i ~/.ssh/id_ed25519`

## One-Shot Setup Script

```bash
# ── 1. Install PyTorch for ROCm 7.2 (~2 min) ──
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2 --no-cache-dir
pip install ninja cmake triton einops numpy tqdm rich

# Verify GPU
python3 -c "import torch; print(f'torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')"

# ── 2. Clone repos (~30 sec) ──
cd ~
git clone https://github.com/dorhuri123/AMD_optimization_challenge.git
cd AMD_optimization_challenge && git checkout mixed-mla-dev

git clone https://github.com/gpu-mode/reference-kernels ~/reference-kernels

# ── 3. Copy official reference files ──
cp ~/reference-kernels/problems/amd_202602/eval.py .
cp ~/reference-kernels/problems/amd_202602/utils.py .
cp ~/reference-kernels/problems/amd_202602/mixed-mla/reference.py mixed-mla/
cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.py mixed-mla/
cp ~/reference-kernels/problems/amd_202602/mixed-mla/task.yml mixed-mla/

# ── 4. Install AITER (~1 min) ──
cd ~ && git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
export PATH=$HOME/.local/bin:$PATH
python3 setup.py develop 2>&1 | tail -5

# Verify AITER
python3 -c "from aiter.mla import mla_decode_fwd; print('AITER MLA: OK')"

# ── 5. Install popcorn-cli (~10 sec) ──
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
export PATH=$HOME/.local/bin:$PATH

# First time only: register
# popcorn-cli register github
# (opens browser link — authorize with GitHub)

# ── 6. Set environment (add to ~/.bashrc for persistence) ──
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$HOME/aiter:$HOME/AMD_optimization_challenge:$PYTHONPATH' >> ~/.bashrc
echo 'export POPCORN_API_URL="https://site--bot--dxfjds728w5v.code.run"' >> ~/.bashrc
source ~/.bashrc
```

## After Setup — Run / Test / Submit

```bash
cd ~/AMD_optimization_challenge/mixed-mla

# Run reference (first time: ~5 min for JIT, then instant)
python3 -u reference.py

# Run our benchmark
python3 -u benchmark.py

# Submit to leaderboard (runs on GPU MODE's MI355X, no local GPU needed)
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui submission.py      # correctness check
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard --no-tui submission.py  # ranked
```

## Notes

- **First AITER run** takes ~5 min — JIT compiles ASM kernels. Cached in `~/aiter/aiter/jit/build/`
- **If you change PyTorch version**, clear JIT cache: `find ~/aiter -name "*.so" -delete && rm -rf ~/aiter/aiter/jit/build`
- **popcorn submit works from anywhere** — it uploads your file and runs on their MI355X. The GPU is only needed for local dev/profiling
- **Host ROCm is 7.2.0** — PyTorch must be `+rocm7.2` to match

## Provider Info

| | Hot Aisle | RunPod |
|---|---|---|
| Type | Bare metal | SR-IOV virtualized |
| Works? | **Yes** | **No** — hipInit() hangs |
| Price | $1.99/GPU/hr | ~$3.5/GPU/hr |
| SSH | `hotaisle@<IP>` | via proxy |
