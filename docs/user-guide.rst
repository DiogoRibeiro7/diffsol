User Guide
==========

This guide explains how to install ``diffsol-pytorch`` on common platforms, migrate from ``torchdiffeq``, and get the best runtime behaviour when solving large ODE/DAE systems. It closes with troubleshooting tips for the issues we see most often when pairing scientific solvers with machine-learning workloads.

For detailed platform setup instructions, see :doc:`install`.

Installation
------------

LLVM/Enzyme prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~

All LLVM-enabled builds (Linux, macOS, Windows) need a matching toolchain on disk. Run the helper once per machine:

.. code-block:: bash

   python tools/prepare_toolchain.py fetch --version 18.1.8 --install-dir .deps --export-prefix
   export LLVM_SYS_181_PREFIX=$PWD/.deps/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-22.04

On Windows use ``setx`` (or your preferred profile) instead of ``export`` and ensure ``cmake`` plus ``ninja`` are on ``PATH``. Re-run ``python tools/prepare_toolchain.py check`` if you upgrade LLVM manually; the command validates ``llvm-config`` plus the host build tools before ``maturin`` starts compiling PyO3 bindings.

The build scripts respect two opt-out flags for power users:

* ``DIFFSOL_ALLOW_UNSUPPORTED_LLVM=1`` – attempt to build against an LLVM major other than 18.
* ``DIFFSOL_SKIP_LLVM_VERSION_CHECK=1`` / ``DIFFSOL_SKIP_TOOLCHAIN_CHECK=1`` – silence the automatic validation when you know the environment is already wired up.

Windows
~~~~~~~

1. Install **Python 3.9+** and **PyTorch >= 1.11** (CPU or CUDA) from https://pytorch.org.
2. Install the **Rust toolchain** (https://rustup.rs). On Windows the MSVC toolchain is required.
3. Run ``python tools\prepare_toolchain.py fetch --export-prefix`` and set ``LLVM_SYS_181_PREFIX`` to the printed directory (typically ``.deps\clang+llvm-18.1.8-x86_64-pc-windows-msvc``).
4. (Optional, for GPU benchmarks) install the CUDA toolkit and ensure ``nvcc`` is on your ``PATH``.
5. Create a virtual environment and install build requirements::

      python -m venv .venv
      .\.venv\Scripts\activate
      pip install maturin torch pytest numpy

6. Build and install the bindings::

      cd bindings\diffsol-pytorch
      maturin develop  # uses PyO3 to build the extension in-place

Linux / macOS
~~~~~~~~~~~~~

1. Install Python and PyTorch as above. The recommended path on Linux is via ``pip`` or ``conda``; on macOS use the CPU wheels.
2. Fetch the LLVM toolchain with ``python tools/prepare_toolchain.py fetch --version 18.1.8 --install-dir .deps`` and export ``LLVM_SYS_181_PREFIX`` to the extracted directory.
3. Install build prerequisites: ``sudo apt install build-essential python3-dev ninja-build`` (Linux) or Xcode command-line tools + Homebrew ``ninja`` (macOS).
4. Install Rust with ``curl https://sh.rustup.rs -sSf | sh``.
5. Build the bindings with ``maturin develop`` exactly as on Windows.

Conda Environments
~~~~~~~~~~~~~~~~~~

``diffsol-pytorch`` builds cleanly inside conda as long as the environment owns the Python and PyTorch installations. Activate the environment before running ``maturin`` so it links against the right interpreter, e.g.:

.. code-block:: bash

   conda activate ode-dev
   maturin develop

Migration from ``torchdiffeq``
------------------------------

The PyTorch API deliberately mirrors the ``torchdiffeq`` ideas, but the control surface is slightly different:

.. list-table::
   :header-rows: 1

   * - ``torchdiffeq``
     - ``diffsol-pytorch``
     - Notes
   * - ``torchdiffeq.odeint(func, y0, t)``
     - ``DiffsolModule(code).solve_dense(params, t.tolist())``
     - Replace Python RHS functions with DiffSL modules. Dense output is returned column-major; reshape with ``np.array(flat).reshape(nout, nt)``.
   * - ``adjoint`` backward pass
     - ``diffsol_pytorch.reverse_mode(code, params, times, grad_output)``
     - Adjoint gradients are computed outside autograd. ``grad_output`` is shaped ``nout x nt`` in column-major order.
   * - ``odeint(..., options={'rtol':1e-4, 'atol':1e-5})``
     - ``OdeBuilder`` options baked into the DiffSL module
     - Set tolerances when building the DiffSL problem (``rtol``, ``atol``). Exposed today via the Rust builder; Python bindings honour whatever the DiffSL file encodes.

Typical migration steps:

1. Translate the RHS into DiffSL (see ``examples/scientific``). Each ``state`` becomes an entry in ``u_i`` with initial conditions.
2. Instantiate ``DiffsolModule`` once per distinct system and reuse it across batches to amortise compilation.
3. Replace autograd recording of PyTorch ops with explicit calls to ``reverse_mode`` (adjoint) or ``forward_mode`` (sensitivities), then feed the resulting gradients into your PyTorch optimiser.

Performance Tuning
------------------

* **Choose the right solver**: The default BDF integrator excels on stiff problems and DAEs. For purely non-stiff neural ODEs you can swap to the ERK (Tsitouras 4/5) solver at the Rust level to reduce cost.
* **Dense grid vs sampled output**: ``solve_dense`` interpolates to every time in ``times``. If you only care about a handful of points, keep the grid short and let the solver adapt internally.
* **Tolerance strategy**: Start with ``rtol=1e-6`` / ``atol=1e-9`` for scientific workloads. For ML pipelines you can often relax to ``1e-4`` to cut step counts drastically.
* **Adjoint checkpointing**: Reverse-mode uses checkpointing under the hood. You can supply coarser ``times`` for the backward pass (e.g. only observation points) to limit memory pressure.
* **Batching**: Create one module per unique RHS and loop over parameter sets; the compiled kernel is cached so parameter sweeps are cheap.
* **GPU**: The PyTorch bindings keep data on host memory. For GPU-heavy training, move inputs/outputs via ``torch.tensor(flat, device=device)`` and rely on diffsol for the heavy lifting; the solver itself currently runs on CPU.

Troubleshooting
---------------

``RuntimeError: module does not support sens autograd``
   You built without LLVM/Enzyme (Cranelift-only). Install LLVM (v18) and rebuild ``diffsol`` with the ``diffsl-llvm`` feature (set ``LLVM_SYS_181_PREFIX`` before running ``maturin``).

``pyo3_runtime.PanicException: gradient output length must match number of time points``
   ``grad_output`` must have ``nout * len(times)`` entries in column-major order. Build it with ``grad.reshape(nout, nt)`` then flatten with ``.reshape(-1).tolist()``.

Solver stalls or ``DIFFSOL: stepsize underflow``
   Usually resolved by loosening tolerances or supplying consistent initial conditions for algebraic components. Inspect ``examples/scientific/dae_index1.py`` for a template on handling constraints.

``pip install diffsol-pytorch`` fails on Windows
   Build from source using ``maturin build``. Wheels for Windows aren't published yet because the project requires a system LLVM. The user guide above walks through compiling locally.

PyTorch/Diffsol version mismatch
   Ensure the Python running ``maturin`` is the same interpreter you plan to import from. Virtual environments avoid this entire class of errors.

Error taxonomy & recovery
-------------------------

Runtime messages now include the failing stage plus a suggested fix:

* **Build failure** – DiffSL parsing/compilation or solver initialisation (suggestion: rerun ``diffsl`` locally or rebuild with ``--features diffsl-llvm``).
* **Autodiff error** – forward/reverse setup or checkpoint replay (suggestion: ensure LLVM/Enzyme is installed, pass gradients with shape ``nout × len(times)``).
* **Runtime error** – dense solves, interpolation, Python bridge (suggestion: provide non-empty time grids, loosen tolerances, or double-check event functions).

Shape issues become Python ``ValueError`` exceptions; other errors map to ``RuntimeError`` while preserving the ``Help: …`` footer so notebook users know what to try next. On the Rust side, match on the ``TorchDiffsolError`` variants to implement custom recovery logic in host applications.

Debugging & Logging
-------------------

- Call ``diffsol_pytorch.enable_logging("debug")`` (or another level) at process start to initialise the Rust logger. This wraps ``env_logger`` behind a Python helper.
- Set ``RUST_LOG=diffsol=info`` (or ``trace``) when you want per-module filtering beyond the helper.
- Use ``DIFFSOL_SAVE_CHECKPOINTS=1`` to dump forward checkpoints into ``/tmp/diffsol-checkpoints`` for post-mortem analysis.
- The Python bindings expose ``diffsol_pytorch.testing`` utilities (finite differences, sensitivity checks) to help isolate gradient discrepancies.

Device placement
----------------

``diffsol_pytorch.device`` helps you keep tensors on the "right" device while still running the solver
on the CPU:

.. code-block:: python

   from diffsol_pytorch import DiffsolModule, device
   module = DiffsolModule(code)
   params = torch.tensor([0.7], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
   times = torch.linspace(0.0, 1.0, steps=128, device=params.device)
   sol = device.solve_dense_tensor(module, params, times)
   grad = device.reverse_mode_tensor(code, params, times, torch.ones_like(sol))

``device.select_device`` honours explicit strings (``"cuda:1"``, ``"mps"``) and falls back to CPU otherwise,
while ``supported_backends_message`` prints a user-friendly summary of CUDA/ROCm/MPS availability for diagnostics.

Memory profiling & leak detection
---------------------------------

- Host buffers are tracked in Rust via a lightweight leak detector. Run ``cargo test -p diffsol-pytorch --features diagnostics --test memory_stress`` to ensure repeated solves/free-lists leave zero live allocations.
- The Python suite adds ``pytest`` ``slow`` markers that loop ``solve_dense`` / ``reverse_mode`` calls for thousands of solver steps. Enable them locally with ``pytest bindings/diffsol-pytorch/tests -m slow``.
- Automated leak detection lives in ``scripts/leakcheck.sh``. It downloads LLVM (when ``LLVM_SYS_181_PREFIX`` is unset), builds the ``leak_harness`` binary with the ``mem-profiling`` + ``llvm`` features (activating jemalloc statistics), and runs it under Valgrind. The harness aborts when jemalloc or the Rust guard detects a leak, ensuring Enzyme checkpoint/tape allocations are reclaimed after long adjoint passes.

If you hit something not covered here, please open an issue with your platform, full command log, and (if possible) the DiffSL snippet that reproduces the error.
