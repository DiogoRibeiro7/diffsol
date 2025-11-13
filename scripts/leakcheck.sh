#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LLVM_SYS_181_PREFIX:-}" ]]; then
  python3 "$REPO_ROOT/tools/prepare_toolchain.py" fetch --version 18.1.8 --install-dir "$REPO_ROOT/.deps"
  LLVM_DIR="$(ls -d "$REPO_ROOT"/.deps/clang+llvm-18.1.8-* | head -n 1)"
  export LLVM_SYS_181_PREFIX="$LLVM_DIR"
fi

cargo test -p diffsol-pytorch --features diagnostics --test memory_stress
cargo build -p diffsol-pytorch --bin leak_harness --features "mem-profiling llvm"

valgrind --error-exitcode=1 --leak-check=full --track-origins=yes "$REPO_ROOT/target/debug/leak_harness"
