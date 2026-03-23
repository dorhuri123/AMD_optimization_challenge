"""
benchmark_all.py — run all 3 challenge benchmarks and print a summary table.
Run: python scripts/benchmark_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

CHALLENGES = [
    ("mla-decode",  "MLA Decode"),
    ("mxfp4-gemm",  "MXFP4 GEMM"),
    ("mxfp4-moe",   "MXFP4 MoE"),
]


def run_benchmark(folder: str) -> tuple[str, str]:
    bench_script = ROOT / folder / "benchmark.py"
    if not bench_script.exists():
        return "SKIP", "benchmark.py not found"
    try:
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(bench_script)],
            capture_output=True, text=True, timeout=300,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            # grab last non-empty line as summary
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            summary = lines[-1] if lines else f"done in {elapsed:.1f}s"
            return "PASS", summary
        else:
            err = result.stderr.splitlines()[-1] if result.stderr else "unknown error"
            return "FAIL", err
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "exceeded 5 min limit"
    except Exception as e:
        return "ERROR", str(e)


print("\n" + "=" * 65)
print("  AMD Challenge — All Benchmarks")
print("=" * 65)

results = []
for folder, name in CHALLENGES:
    print(f"\nRunning {name}...")
    status, detail = run_benchmark(folder)
    results.append((name, status, detail))

print("\n" + "=" * 65)
print(f"{'Challenge':<22} {'Status':<10} {'Detail'}")
print("-" * 65)
for name, status, detail in results:
    color = "\033[92m" if status == "PASS" else "\033[91m" if status == "FAIL" else "\033[93m"
    reset = "\033[0m"
    print(f"{name:<22} {color}{status:<10}{reset} {detail}")
print("=" * 65 + "\n")
