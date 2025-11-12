Benchmarks & Comparisons
========================

Diffsol ships with reproducible benchmarking scripts under ``benchmarks/``. They target both ML workloads (neural ODEs, CNFs) and classical stiff problems so you can decide when diffsol outperforms ``torchdiffeq`` or other explicit integrators.

Performance Highlights
----------------------

Recent CPU runs (from ``benchmarks/results.md``):

.. list-table::
   :header-rows: 1

   * - Problem
     - Backend
     - Time (s)
     - Max error / note
   * - Logistic decay (non-stiff, 500 steps)
     - diffsol (BDF)
     - 0.0020
     - ``4.39e-07`` max error vs analytic, adjoint gradient ``-1.51e-01``
   * - Logistic decay (stiff, 1500 steps, k=150)
     - diffsol (BDF)
     - 0.0027
     - ``7.34e-07`` max error; torchdiffeq would require step size control to stay stable
   * - Neural ODE block (2D hidden state)
     - diffsol (BDF)
     - 0.0021
     - Norm of final state ``||z(T)|| = 1.861`` (identical solutions across runs)

When ``torchdiffeq`` is installed, ``benchmarks/perf.py`` adds its timings and accuracy alongside diffsol, making it easy to paste the tables into papers or READMEs.

Accuracy Analysis
-----------------

* **Deterministic problems (logistic, Van der Pol).** Errors stay within single-precision epsilon at tolerances ``rtol=1e-6`` / ``atol=1e-9``. Analytic gradients agree with adjoint output to better than ``1e-5``.
* **Energy-preserving systems (Hamiltonian).** The scientific regression suite tracks energy drift and keeps it below ``10^{-5}`` for double precision.
* **DAEs.** Residual norms decay steadily even when algebraic variables start far from a consistent state; see ``examples/scientific/dae_index1.py`` for the setup mirrored in the tests.

Choosing diffsol vs Alternatives
--------------------------------

- **Use diffsol when…**

  - You need stiff or DAE support (mass matrices, algebraic constraints, piecewise events).
  - You want robust reverse-mode/forward-mode sensitivities without building massive PyTorch graphs.
  - You care about dense output at arbitrary times (e.g., for physics supervision or PINN residuals).

- **Stick with torchdiffeq when…**

  - You only need quick experiments with explicit RHS functions written directly in Python.
  - Your problem is light enough that autograd through the solver is acceptable and you value GPU-native integration (diffsol currently computes on CPU).

Case Studies
------------

1. **Neural ODE classification.** The Fashion-MNIST classifier (see :doc:`scientific-examples`) uses diffsol’s adjoint gradients to train logistic flows without ever differentiating through intermediate solver states. torchdiffeq would need to retain the full trajectory or recompute, incurring a large memory/time penalty.
2. **Continuous Normalizing Flow (CNF).** The CNF example consumes reverse-mode gradients to update cubic dynamics. Finite-difference gradients are numerically unstable here, while autograd-through-ODE requires extremely fine timesteps; diffsol’s adjoint makes the update cost negligible.
3. **Hamiltonian regression.** The regression tests under ``examples/scientific`` verify that diffsol preserves invariants (energy, phase) better than explicit integrators unless those integrators take prohibitively tiny steps.

GPU Validation
--------------

``benchmarks/gpu_validation.py`` measures solver runtime while moving inputs/outputs across devices; it also reports CUDA memory usage when the tensors live on GPU. A typical run:

.. code-block:: bash

   python benchmarks/gpu_validation.py --device cpu
   python benchmarks/gpu_validation.py --device cuda

Even though diffsol currently executes on CPU, this script highlights data-transfer overheads and peak allocator usage so you can balance batching against PCIe latency.

Workflow
--------

.. code-block:: bash

   python benchmarks/perf.py --device cpu
   python benchmarks/perf.py --device cuda   # if torchdiffeq + CUDA are available

Every invocation overwrites ``benchmarks/results.md``. Commit those Markdown tables alongside your experiments to document reproducibility.

Per the release strategy, the latest ``results.md`` snapshot is **kept under version control** and refreshed for each tagged release. If you run additional benchmarks locally, regenerate the table before publishing to ensure the data reflects your environment.
