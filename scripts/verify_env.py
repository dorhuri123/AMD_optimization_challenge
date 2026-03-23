"""
verify_env.py — sanity-check the environment before starting work.
Run: python scripts/verify_env.py
"""

import sys
import subprocess

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"

checks_failed = 0


def check(label: str, fn):
    global checks_failed
    try:
        result = fn()
        print(f"[{PASS}] {label}{': ' + str(result) if result else ''}")
    except Exception as e:
        print(f"[{FAIL}] {label}: {e}")
        checks_failed += 1


def warn(label: str, fn):
    try:
        result = fn()
        print(f"[{PASS}] {label}{': ' + str(result) if result else ''}")
    except Exception as e:
        print(f"[{WARN}] {label}: {e} (optional)")


print("=" * 55)
print("  AMD Challenge — Environment Check")
print("=" * 55)

# ── Python version ────────────────────────────────────────
check("Python >= 3.10", lambda: (
    f"{sys.version_info.major}.{sys.version_info.minor}",
    None if sys.version_info >= (3, 10) else (_ for _ in ()).throw(
        RuntimeError(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old"))
)[0])

# ── PyTorch + ROCm ───────────────────────────────────────
check("torch importable", lambda: __import__("torch").__version__)

check("torch.cuda.is_available() (ROCm)", lambda: (
    "yes" if __import__("torch").cuda.is_available()
    else (_ for _ in ()).throw(RuntimeError("No GPU visible to PyTorch"))
))

check("GPU device name", lambda: __import__("torch").cuda.get_device_name(0))

check("GPU count", lambda: f"{__import__('torch').cuda.device_count()} device(s)")

# ── Triton ───────────────────────────────────────────────
check("triton importable", lambda: __import__("triton").__version__)

# ── AITER ────────────────────────────────────────────────
check("aiter importable", lambda: __import__("aiter").__version__ if hasattr(__import__("aiter"), "__version__") else "ok")

warn("aiter.mla_decode_fwd callable", lambda: (
    getattr(__import__("aiter"), "mla_decode_fwd"),
    "found"
)[1])

# ── Popcorn CLI ──────────────────────────────────────────
warn("popcorn-cli installed", lambda: (
    subprocess.check_output(["popcorn", "--version"], stderr=subprocess.DEVNULL)
    .decode().strip()
))

# ── rocm-smi ────────────────────────────────────────────
check("rocm-smi available", lambda: (
    subprocess.check_output(["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL)
    .decode().split("\n")[2].strip() if True else "ok"
))

# ── Quick tensor op on GPU ───────────────────────────────
def _gpu_tensor_check():
    import torch
    a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    return f"bf16 matmul {a.shape} OK"

check("BF16 matmul on GPU", _gpu_tensor_check)

# ── FP8 support ──────────────────────────────────────────
def _fp8_check():
    import torch
    a = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    # ROCm uses e4m3fnuz; NVIDIA uses e4m3fn
    fp8_dtype = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn
    a_fp8 = a.to(fp8_dtype)
    return f"{fp8_dtype} supported"

check("FP8 dtype available", _fp8_check)

# ── Summary ──────────────────────────────────────────────
print("=" * 55)
if checks_failed == 0:
    print("\033[92mAll checks passed. Environment is ready.\033[0m")
else:
    print(f"\033[91m{checks_failed} check(s) failed. Fix the issues above before proceeding.\033[0m")
    sys.exit(1)
