#!/usr/bin/env python3
"""
Download or validate LLVM + Enzyme toolchains for diffsol.

Usage examples:
  python tools/prepare_toolchain.py fetch --version 18.1.8
  python tools/prepare_toolchain.py check --prefix .deps/clang+llvm-18.1.8-x86_64-pc-windows-msvc
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

LLVM_RELEASE_BASE = "https://github.com/llvm/llvm-project/releases/download"

ARTIFACTS = {
    ("Windows", "AMD64"): "clang+llvm-{version}-x86_64-pc-windows-msvc.tar.xz",
    ("Linux", "x86_64"): "clang+llvm-{version}-x86_64-linux-gnu-ubuntu-22.04.tar.xz",
    ("Darwin", "arm64"): "clang+llvm-{version}-arm64-apple-darwin22.0.tar.xz",
    ("Darwin", "x86_64"): "clang+llvm-{version}-x86_64-apple-darwin.tar.xz",
}


def host_tuple() -> tuple[str, str]:
    return platform.system(), platform.machine()


def resolve_artifact(version: str, override: str | None) -> str:
    if override:
        return override
    key = host_tuple()
    try:
        pattern = ARTIFACTS[key]
    except KeyError as exc:
        raise SystemExit(f"Unsupported host triple {key}. Use --artifact to override.") from exc
    return pattern.format(version=version)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[toolchain] downloading {url}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)
    return dest


def _safe_extract(tf: tarfile.TarFile, dest_dir: Path) -> None:
    for member in tf.getmembers():
        member_path = dest_dir / member.name
        if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
            raise SystemExit(f"Unsafe path detected in archive: {member.name}")
    tf.extractall(dest_dir)


def extract(archive: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        top_level = tf.getmembers()[0].name.split("/")[0]
        target = dest_dir / top_level
        if target.exists():
            print(f"[toolchain] {target} already exists, skipping extraction")
            return target
        print(f"[toolchain] extracting to {dest_dir}")
        _safe_extract(tf, dest_dir)
        return target


def fetch(args: argparse.Namespace) -> None:
    artifact = resolve_artifact(args.version, args.artifact)
    url = f"{LLVM_RELEASE_BASE}/llvmorg-{args.version}/{artifact}"
    dest = Path(args.install_dir).resolve()
    tarball = dest / artifact
    if not tarball.exists() or args.force:
        download(url, tarball)
    prefix = extract(tarball, dest)
    print(f"[toolchain] LLVM unpacked to {prefix}")
    print(
        "[toolchain] export LLVM_SYS_181_PREFIX="
        f"{prefix if args.export_prefix else '<path-to-toolchain>'}"
    )


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def check(args: argparse.Namespace) -> None:
    prefix = Path(args.prefix or os.environ.get("LLVM_SYS_181_PREFIX", "")).expanduser()
    if not prefix:
        raise SystemExit("Pass --prefix or set LLVM_SYS_181_PREFIX")
    if not prefix.is_dir():
        raise SystemExit(f"{prefix} does not exist")
    llvm_config = prefix / "bin" / ("llvm-config.exe" if platform.system() == "Windows" else "llvm-config")
    if not llvm_config.exists():
        raise SystemExit(f"{llvm_config} not found")
    version = run_cmd([str(llvm_config), "--version"]).stdout.strip()
    major = int(version.split(".")[0])
    if major != 18 and not args.allow_any:
        raise SystemExit(
            f"Expected LLVM major 18, detected {version}. "
            "Pass --allow-any to bypass."
        )
    required = ["cmake"]
    if platform.system() == "Windows":
        required.append("ninja")
    for tool in required:
        try:
            run_cmd([tool, "--version"])
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise SystemExit(f"Required build tool `{tool}` missing: {exc}") from exc
    print(f"[toolchain] {prefix} OK (LLVM {version})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Download and extract a prebuilt LLVM toolchain")
    fetch_p.add_argument("--version", default="18.1.8", help="LLVM version tag (default: 18.1.8)")
    fetch_p.add_argument(
        "--install-dir",
        default=".deps",
        help="Directory to place the downloaded toolchain (default: .deps)",
    )
    fetch_p.add_argument("--artifact", help="Override artifact name")
    fetch_p.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive even if it already exists",
    )
    fetch_p.add_argument(
        "--export-prefix",
        action="store_true",
        help="Print a ready-to-use LLVM_SYS_181_PREFIX export line",
    )
    fetch_p.set_defaults(func=fetch)

    check_p = sub.add_parser("check", help="Validate an existing LLVM installation")
    check_p.add_argument("--prefix", help="Path to LLVM install (defaults to LLVM_SYS_181_PREFIX)")
    check_p.add_argument(
        "--allow-any",
        action="store_true",
        help="Allow LLVM majors other than 18",
    )
    check_p.set_defaults(func=check)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
