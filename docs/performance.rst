Performance Optimisation
========================

``diffsol-pytorch`` exposes several levers to help researchers keep simulation and
autodiff workloads fast and memory friendly. This page summarises the most important
ideas so you can apply them systematically.

Solver selection
----------------

* **BDF** (default) – best for stiff systems and DAEs, but incurs heavier linear solves.
* **SDIRK/ERK** – switch via the Rust builder (``OdeBuilder``) for moderately stiff or purely
  non-stiff systems to avoid unnecessary Jacobian work.
* **Checkpoint spacing** – for adjoint runs, pass a coarser time grid to
  ``solve_dense_with_checkpointing`` so the reverse pass recomputes fewer segments.

Device placement
----------------

Use :mod:`diffsol_pytorch.device` to keep tensors on their preferred accelerator while the
solver runs on CPU:

.. code-block:: python

   from diffsol_pytorch import device
   sol = device.solve_dense_tensor(module, params, times)
   grad = device.reverse_mode_tensor(code, params, times, grad_output)

* ``device.select_device`` automatically prefers CUDA → ROCm → MPS → CPU unless you
  pass ``device="cuda:1"`` explicitly.
* ``supported_backends_message()`` prints a helpful summary for dashboards.

Autodiff cost controls
----------------------

* **Gradient sparsity** – When possible, produce gradient buffers with zeros in unused rows so
  the reverse pass can short-circuit.
* **Finite-difference sanity checks** – Call :func:`diffsol_pytorch.testing.check_gradients`
  sparingly (e.g., once per unit test) to avoid repeated expensive runs.
* **Shape validation** – mismatched ``nout × len(times)`` gradients trigger an early error rather
  than wasted computation.

PyTorch integration
-------------------

* Reuse ``DiffsolModule`` instances across batches to amortise JIT compilation.
* Enable mixed precision only after confirming the solver tolerances (``rtol``/``atol``) make sense.
* Consider wrapping solver calls in ``torch.utils.checkpoint`` when building large neural
  architectures, but keep in mind that diffsol already performs checkpointing internally for adjoints.

Benchmarking & regression detection
-----------------------------------

* ``pytest -m perf --run-perf`` exercises the ``pytest-benchmark`` suite and persists JSON
  results (supply ``--benchmark-json``) so you can track regressions across commits.
* ``scripts/run_validation.py --perf`` combines the standard unit/integration suite with the
  benchmark run to maintain a single entrypoint.
* Use ``pytest --durations=25`` (already enabled in ``pytest.ini``) to keep an eye on the slowest
  tests; unexpected spikes often hint at tolerance misconfiguration or an accidental device copy.

Memory hygiene
--------------

* ``cargo test -p diffsol-pytorch --features diagnostics --test memory_stress`` runs the Rust leak
  harness (jemalloc-based) and should stay part of any release cycle.
* ``pytest -m memory`` uses ``psutil`` to detect resident set drift for long-running CPU workloads.
* For GPU tensors, monitor ``torch.cuda.memory_summary()`` around solver calls if you suspect
  device-side fragmentation – all solver allocations happen on the host, so device spikes usually
  indicate upstream tensor usage.
