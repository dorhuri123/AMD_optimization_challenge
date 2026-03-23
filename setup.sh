#!/usr/bin/env bash
# AMD x GPU MODE Challenge — Environment Setup
# Run once on your GPU machine: bash setup.sh
set -euo pipefail

ROCM_MIN="6.3"
AITER_DIR="$HOME/libs/aiter"
REF_KERNELS_DIR="$HOME/libs/reference-kernels"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. Check ROCm ────────────────────────────────────────────────────────────
info "Checking ROCm..."
if ! command -v rocm-smi &>/dev/null; then
    error "rocm-smi not found. Install ROCm >= ${ROCM_MIN} first: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
fi
rocm-smi --showproductname 2>/dev/null || true

ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || rocminfo 2>/dev/null | grep -m1 "ROCm Version" | awk '{print $NF}' || echo "unknown")
info "ROCm version: ${ROCM_VER}"

# ── 2. Python dependencies ───────────────────────────────────────────────────
info "Installing Python dependencies..."

# Detect ROCm version for correct PyTorch wheel
ROCM_SHORT=$(echo "$ROCM_VER" | grep -oP '^\d+\.\d+' || echo "6.2")
TORCH_ROCM_TAG="rocm${ROCM_SHORT}"

pip install --upgrade pip setuptools wheel

# PyTorch for ROCm
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    warn "PyTorch not found or no GPU visible. Installing for ${TORCH_ROCM_TAG}..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_ROCM_TAG}"
else
    info "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
fi

# Triton (ROCm-compatible)
pip install triton

# General utilities
pip install numpy einops tqdm rich

# ── 3. AITER (AMD AI Tensor Engine for ROCm) ─────────────────────────────────
info "Setting up AITER..."
mkdir -p "$(dirname "$AITER_DIR")"
if [ ! -d "$AITER_DIR" ]; then
    git clone https://github.com/ROCm/aiter "$AITER_DIR"
else
    warn "AITER already cloned at ${AITER_DIR}, pulling latest..."
    git -C "$AITER_DIR" pull
fi
pip install -e "$AITER_DIR"

# ── 4. Popcorn CLI (submission tool) ─────────────────────────────────────────
info "Installing popcorn-cli (submission tool)..."
pip install popcorn-cli || warn "popcorn-cli install failed — try manually: pip install popcorn-cli"

# ── 5. Reference kernels (official baselines) ────────────────────────────────
info "Cloning gpu-mode/reference-kernels..."
mkdir -p "$(dirname "$REF_KERNELS_DIR")"
if [ ! -d "$REF_KERNELS_DIR" ]; then
    git clone https://github.com/gpu-mode/reference-kernels "$REF_KERNELS_DIR"
else
    warn "reference-kernels already cloned at ${REF_KERNELS_DIR}, pulling latest..."
    git -C "$REF_KERNELS_DIR" pull
fi

# ── 6. Write .env for PYTHONPATH / env vars ───────────────────────────────────
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/.env"
cat > "$ENV_FILE" <<EOF
# Source this file before working: source .env
export PYTHONPATH="${AITER_DIR}:\${PYTHONPATH:-}"

# Enable AITER operators in vLLM (if using vLLM)
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1

# Reference kernels path
export REF_KERNELS_DIR="${REF_KERNELS_DIR}"
EOF

info ".env written to ${ENV_FILE}"
info "Run 'source .env' before starting work in each session."

# ── 7. Verify ────────────────────────────────────────────────────────────────
info "Running environment verification..."
python "$(cd "$(dirname "$0")" && pwd)/scripts/verify_env.py" || warn "Verification had warnings — see above."

echo ""
info "Setup complete! Next steps:"
echo "  1. source .env"
echo "  2. popcorn login"
echo "  3. cd mla-decode && python reference.py"
