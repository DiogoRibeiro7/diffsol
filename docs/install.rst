Installation Guide
===================

This document collects the platform-specific steps required to build ``diffsol-pytorch``
from source. The bindings always compile the Rust core locally, so you need a working
Rust toolchain plus a compatible LLVM/Enzyme distribution when enabling autodiff.

Common prerequisites
--------------------

* Python 3.9+ (recommended: 3.11 or 3.12)
* Rust (via ``rustup``) with the MSVC toolchain on Windows
* CMake 3.24+, Ninja, and a recent C/C++ compiler (``clang``/``gcc``/MSVC)
* ``pip install maturin`` for building the PyO3 extension
* Optional: CUDA/ROCm/MPS runtimes when exercising GPU-backed workflows

Linux (Ubuntu/Debian)
---------------------

.. code-block:: bash

   sudo apt update
   sudo apt install build-essential cmake ninja-build python3 python3-venv python3-pip \
        libssl-dev pkg-config curl
   curl https://sh.rustup.rs -sSf | sh  # install Rust

Download LLVM/Enzyme 18.1.x and point ``LLVM_SYS_181_PREFIX`` at the extracted directory::

   python tools/prepare_toolchain.py fetch --version 18.1.8 --install-dir .deps --export-prefix
   export LLVM_SYS_181_PREFIX=$PWD/.deps/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-22.04

macOS (Intel/Apple Silicon)
---------------------------

Install Xcode command-line tools and Homebrew dependencies::

   xcode-select --install
   brew install cmake ninja python@3.12 rustup-init
   rustup-init -y

Use the helper to fetch LLVM/Enzyme and set the environment variable::

   python tools/prepare_toolchain.py fetch --version 18.1.8 --install-dir .deps --export-prefix
   export LLVM_SYS_181_PREFIX=$PWD/.deps/clang+llvm-18.1.8-arm64-apple-darwin22.0  # adjust for Intel vs Apple Silicon

Metal (MPS) requires macOS 13+ and Xcode 14+. CUDA/ROCm are not available on macOS.

Windows 10/11 (MSVC)
--------------------

1. Install Visual Studio Build Tools or VS 2022 with the Desktop C++ workload.
2. Install Python 3.11+, CMake, and Ninja (e.g., via Chocolatey).
3. Install Rust with ``rustup-init.exe`` (choose the *MSVC* toolchain).
4. Fetch LLVM/Enzyme::

       python tools\prepare_toolchain.py fetch --version 18.1.8 --install-dir .deps --export-prefix
       setx LLVM_SYS_181_PREFIX C:\path\to\diffsol\.deps\clang+llvm-18.1.8-x86_64-pc-windows-msvc

CUDA users should also install the matching CUDA Toolkit; ROCm is currently Linux-only.

Verifying the setup
-------------------

Create a virtual environment and install the project in editable mode::

   python -m venv .venv
   . .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
   pip install maturin
   maturin develop --manifest-path bindings/diffsol-pytorch/Cargo.toml --features python

Run the validation suite to ensure autodiff, tests, and leak checks succeed::

   python scripts/run_validation.py

If LLVM autodiff support is optional in your workflow, you can temporarily skip it by
setting ``DIFFSOL_SKIP_LLVM_VERSION_CHECK=1``, but gradient tests will be skipped.
