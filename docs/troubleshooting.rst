Troubleshooting
===============

This FAQ aggregates the issues we see most often when compiling or running ``diffsol-pytorch``.
Each entry lists the visible symptom, probable cause, and suggested fix.

Autodiff disabled ("module does not support sens autograd")
-----------------------------------------------------------

*Cause:* The build only included the Cranelift backend, or ``LLVM_SYS_181_PREFIX`` points to an
incompatible LLVM version.

*Resolution:* Download LLVM/Enzyme 18.1.x via ``python tools/prepare_toolchain.py fetch ...`` and
export ``LLVM_SYS_181_PREFIX`` before running ``maturin``/``cargo``. Rebuild with
``--features diffsl-llvm`` or set ``DIFFSOL_SKIP_LLVM_VERSION_CHECK=1`` only if you intend to skip
autodiff entirely.

``pyo3_runtime.PanicException`` mentioning gradient dimensions
--------------------------------------------------------------

*Cause:* The gradient buffer passed to ``reverse_mode`` or :func:`device.reverse_mode_tensor` does
not contain exactly ``nout × len(times)`` entries (column-major).

*Resolution:* Reshape your loss gradient to ``(nout, len(times))`` and call ``flatten().tolist()`` or
use the helpers in :mod:`diffsol_pytorch.testing`.

``ModuleNotFoundError: diffsol_pytorch`` when running pytest/scripts
-------------------------------------------------------------------

*Cause:* The extension is not built yet, or ``PYTHONPATH`` lacks the ``bindings/diffsol-pytorch/python`` folder.

*Resolution:* Activate your virtual environment and run
``maturin develop --manifest-path bindings/diffsol-pytorch/Cargo.toml --features python``. The
``scripts/run_validation.py`` helper performs this automatically, but ad-hoc pytest runs need the editable install.

Windows linker errors referencing ``Enzyme-18.lib``
---------------------------------------------------

*Cause:* LLVM/Enzyme artefacts are missing or the ``LLVM_SYS_181_PREFIX`` path is incorrect.

*Resolution:* Fetch the toolchain with ``prepare_toolchain.py`` and ensure the ``lib`` directory
contains ``EnzymeStatic-18.lib`` plus the shim ``Enzyme-18.lib`` (the build script creates it).

PyTorch device mismatch / tensors copied to CPU
-----------------------------------------------

*Cause:* ``diffsol_pytorch`` always runs solvers on CPU; without device helpers, tensors must be
manually moved back and forth.

*Resolution:* Use ``diffsol_pytorch.device.solve_dense_tensor`` and
``diffsol_pytorch.device.reverse_mode_tensor`` so host/device transfers are automatic, and call
``device.select_device`` to pin the target accelerator.

Performance regressions vs ``torchdiffeq``
------------------------------------------

*Cause:* Solver tolerances too strict, choosing BDF for non-stiff problems, or new code introduced
unnecessary host/device copies.

*Resolution:* Review :doc:`performance` for solver selection tips, and run
``pytest -m perf --run-perf`` (requires ``torchdiffeq`` installed) to compare runtimes numerically.

``cargo build`` fails to find Python headers
--------------------------------------------

*Cause:* ``maturin`` is invoked without an active virtual environment or using a Python interpreter
that lacks development headers.

*Resolution:* Create a venv (``python -m venv .venv``) and activate it before calling ``maturin``.
On Linux you may also need ``python3-dev``/``pythonX.Y-dev`` packages.
