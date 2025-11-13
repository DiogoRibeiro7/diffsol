Migration from ``torchdiffeq``
===============================

``diffsol-pytorch`` deliberately mirrors the most common ``torchdiffeq`` workflows while adding
stronger numerical guarantees (dense output, reverse-mode via Enzyme, CPU-hosted solvers). This
guide highlights the conceptual differences and walks through a minimal migration.

Quick reference
---------------

.. list-table::
   :header-rows: 1

   * - ``torchdiffeq``
     - ``diffsol-pytorch``
     - Notes
   * - ``torchdiffeq.odeint(func, y0, t, **options)``
     - ``DiffsolModule(code).solve_dense(params, t.tolist())``
     - The RHS lives in DiffSL; tolerances are encoded at compile-time.
   * - ``adjoint`` keyword / ``odeint_adjoint``
     - ``reverse_mode(code, params, times, grad_output)`` or :func:`device.reverse_mode_tensor`
     - Gradients are computed with checkpointing + Enzyme, independent of PyTorch autograd graphs.
   * - Device placement handled by autograd graph
     - Use :mod:`diffsol_pytorch.device` helpers (CPU solver, device-aware transfers)
     - Host buffers hop to CPU before the solver and return to CUDA/ROCm/MPS automatically.
   * - Continuous ODE functions written in Python
     - DiffSL DSL compiled to Rust
     - Rust side enforces bounds/tolerances at compile-time; PyTorch sees only flat buffers.

Migration steps
---------------

1. **Express the RHS in DiffSL.** Scalar/vector expressions map directly to DSL tensors. When the original
   ``func`` receives parameters, add them to the ``in = [...]`` block and refer to them symbolically.
2. **Compile to a ``DiffsolModule``.** Construct once and reuse it across minibatches. If you need multiple
   solvers (e.g., CNF + PINN), instantiate multiple modules.
3. **Prepare time grids/tolerances.** Since ``solve_dense`` interpolates to every requested time, pass the
   same grid you previously fed to ``odeint``. Tolerances can be baked into the DSL or the Rust builder.
4. **Replace autograd hooks.** Call ``reverse_mode`` with a flattened ``grad_output`` buffer. ``testing.check_gradients``
   offers finite-difference validation during development.
5. **Optimise device transfers.** Wrap solver calls with :func:`device.solve_dense_tensor` / :func:`device.reverse_mode_tensor`
   so PyTorch tensors stay on CUDA/ROCm/MPS automatically.

Example: logistic decay
-----------------------

.. code-block:: python

   # torchdiffeq version:
   # sol = odeint(lambda t, u: -k * u, u0, times, rtol=1e-6, atol=1e-9)

   LOGISTIC_CODE = """
   in = [k]
   k { 0.7 }
   u {
       u = 1.0,
   }
   F {
       -k * u,
   }
   """

   module = diffsol_pytorch.DiffsolModule(LOGISTIC_CODE)
   nout, nt, flat = module.solve_dense([0.7], times.tolist())
   sol = np.array(flat, dtype=np.float64).reshape(nout, nt)

   grad_out = np.zeros_like(sol)
   grad_out[0, -1] = 1.0
   grads = diffsol_pytorch.reverse_mode(LOGISTIC_CODE, [0.7], times.tolist(), grad_out.ravel().tolist())

FAQ
---

* **Do I need DiffSL for every RHS?** Yes. DiffSL compile-times are short (milliseconds for small systems),
  and you can keep the DSL source alongside your Python module.
* **Can I reuse PyTorch autograd for parts of the model?** Absolutely. Only the solver core is replaced;
  upstream/downstream PyTorch modules remain differentiable as usual.
* **How do I benchmark fidelity?** Use ``pytest -m perf --run-perf --benchmark-json perf.json`` to compare
  against ``torchdiffeq``; the ``neural_ode_classifier.ipynb`` tutorial contains a practical example.

See also
--------

* :doc:`architecture` – diagrams of the Rust/Python data flow.
* :doc:`performance` – solver selection, checkpoint spacing, and benchmarking advice.
* :doc:`scientific-examples` – CLI scripts plus notebooks for Neural ODEs, PINNs, CNFs, and parameter estimation.
