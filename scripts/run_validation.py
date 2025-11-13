#!/usr/bin/env python3
"""
Run the diffsol test matrix (unit/integration/gradient/memory/perf) with useful defaults.

Examples:
    python scripts/run_validation.py
    python scripts/run_validation.py --perf --perf-json perf.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import importlib


ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_PATH = str(ROOT / "bindings" / "diffsol-pytorch" / "python")
sys.path.insert(0, PYTHON_PACKAGE_PATH)
BASE_ENV = os.environ.copy()
BASE_ENV["PYTHONPATH"] = (
    f"{PYTHON_PACKAGE_PATH}{os.pathsep}{BASE_ENV['PYTHONPATH']}"
    if "PYTHONPATH" in BASE_ENV and BASE_ENV["PYTHONPATH"]
    else PYTHON_PACKAGE_PATH
)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"[run_validation] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env or BASE_ENV)


def pytest_select(expr: str, *extra: str) -> list[str]:
    return ["pytest", "bindings/diffsol-pytorch/tests", "-m", expr, *extra]


def ensure_extension_built() -> None:
    try:
        importlib.import_module("diffsol_pytorch")
    except ImportError:
        run(
            [
                "maturin",
                "develop",
                "--manifest-path",
                "bindings/diffsol-pytorch/Cargo.toml",
                "--features",
                "python",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf", action="store_true", help="Run performance benchmarks (torchdiffeq required).")
    parser.add_argument(
        "--perf-json",
        type=Path,
        default=ROOT / "benchmark-results.json",
        help="Where to store pytest-benchmark JSON output (default: %(default)s).",
    )
    args = parser.parse_args()

    ensure_extension_built()

    # Unit + integration + gradient => default suite with coverage
    run(pytest_select("unit or integration or gradient"))

    # Memory/leak checks (skipped automatically when psutil absent)
    run(pytest_select("memory"))

    # Rust leak harness (diagnostics feature)
    run(["cargo", "test", "-p", "diffsol-pytorch", "--features", "diagnostics", "--test", "memory_stress"])

    if args.perf:
        run(
            pytest_select("perf", "--run-perf", f"--benchmark-json={args.perf_json}"),
            env={**os.environ, "PYTEST_ADDOPTS": os.environ.get("PYTEST_ADDOPTS", "")},
        )


if __name__ == "__main__":
    main()
