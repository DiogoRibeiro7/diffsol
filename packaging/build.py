#!/usr/bin/env python
"""
Helper script to standardise wheel/conda builds for diffsol-pytorch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist"


def run(cmd: list[str]) -> None:
    print(f"[packaging] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_wheel(target: str | None, release: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "--strip",
        "--out",
        str(out_dir),
    ]
    if release:
        cmd.append("--release")
    if target:
        cmd.extend(["--target", target])
    run(cmd)


def build_conda(channel: str | None, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    cmd = [
        "conda-build",
        str(ROOT / "packaging" / "conda"),
        "--output-folder",
        str(output_folder),
    ]
    if channel:
        cmd.extend(["--channel", channel])
    run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_wheel(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target", help="Target triple passed to maturin build.")
        p.add_argument(
            "--release",
            action="store_true",
            help="Build wheels in release mode (default: debug).",
        )
        p.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_DIST / "wheels",
            help="Directory for wheel artifacts.",
        )

    wheel = sub.add_parser("wheel", help="Build wheels via maturin.")
    add_common_wheel(wheel)

    conda = sub.add_parser("conda", help="Build the Conda recipe.")
    conda.add_argument("--channel", help="Extra channel passed to conda-build.")
    conda.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_DIST / "conda",
        help="Directory for conda-build outputs.",
    )

    combo = sub.add_parser("all", help="Run both wheel and Conda builds.")
    add_common_wheel(combo)
    combo.add_argument("--channel", help="Extra channel passed to conda-build.")
    combo.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_DIST / "conda",
        help="Directory for conda-build outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "wheel":
        build_wheel(args.target, args.release, args.out_dir)
    elif args.command == "conda":
        build_conda(args.channel, args.output_folder)
    elif args.command == "all":
        build_wheel(args.target, args.release, args.out_dir)
        build_conda(args.channel, args.output_folder)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
