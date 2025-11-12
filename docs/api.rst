API Reference
=============

High-level Python API
---------------------

DiffsolModule
~~~~~~~~~~~~~

.. autoclass:: diffsol_pytorch.DiffsolModule
    :members:
    :undoc-members:

A module owns the compiled DiffSL problem, including solver tolerances and optional output equations. Construction compiles the RHS once; subsequent calls to ``solve_dense`` reuse the compiled kernel. Output is returned in column-major order (``nout`` rows, ``len(times)`` columns).

reverse_mode
~~~~~~~~~~~~

.. autofunction:: diffsol_pytorch.reverse_mode

Computes gradients with respect to both parameters and the initial state by running the adjoint solver backwards through the checkpoints produced by ``solve_dense``. ``grad_output`` must contain ``nout × len(times)`` entries (column-major). On builds compiled without LLVM/Enzyme the function raises ``RuntimeError``.

forward_mode
~~~~~~~~~~~~

.. automethod:: diffsol_pytorch.DiffsolModule.forward_mode

Returns both the dense trajectory and stacked sensitivities ``∂y/∂p``. Use this when you need the full Jacobian (e.g., Gauss–Newton fits, sensitivity analyses) rather than the adjoint gradient of a scalar loss.

Mathematical Background
-----------------------

DiffSL describes semi-explicit DAEs

.. math::

   M(t, p)\,\dot{u}(t) = F(u(t), t, p),

where the optional mass matrix ``M`` allows algebraic constraints. The PyTorch bindings expose the solutions and their derivatives; solver selection happens on the Rust side:

* **BDF (default)** – variable-order implicit method for stiff ODEs/DAEs.
* **SDIRK (TR-BDF2 / ESDIRK34)** – moderate stiffness, fewer LU factorizations.
* **ERK (Tsitouras 4/5)** – non-stiff neural ODEs / CNFs, minimal overhead.

Performance & Memory Characteristics
------------------------------------

.. list-table::
   :header-rows: 1

   * - Solver
     - When to use
     - Memory footprint
   * - BDF
     - Stiff dynamics, algebraic constraints, Hamiltonian systems with damping
     - Stores ``k`` past states plus LU factorizations (``O(n^2)``). Adjoint checkpointing keeps backward memory constant.
   * - SDIRK
     - Mixed stiffness where implicit solves are still beneficial
     - Slightly smaller history than BDF; same adjoint behaviour.
   * - ERK
     - Smooth neural ODEs, CNFs, PINNs
     - Memory dominated by dense output buffer (``nout × nt``).

Back-of-the-envelope budget: ``8 bytes × nout × len(times)`` for the dense trajectory plus solver state (~``O(n)``). If you only need derived observations, define “outputs” in DiffSL to reduce ``nout``.

Advanced Configuration
----------------------

* **Tolerances** – set ``rtol``/``atol`` in the DiffSL builder (Rust). Lower values increase accuracy but require more steps.
* **Mass matrices** – add an ``M_i`` block to encode differential/algebraic splits. ``dudt_i`` provides starting guesses for algebraic components.
* **Outputs vs states** – use ``out_i`` to emit arbitrary observables; gradients propagate through the output operator automatically.
* **GPU workflows** – The solver runs on CPU; transfer the flatten buffer to CUDA with ``torch.tensor(flat, device='cuda')`` if your downstream model lives on the GPU.
* **Autodiff selection** – Builds without LLVM/Enzyme only expose ``solve_dense``. Install LLVM 18 and rebuild with the ``diffsl-llvm`` feature to enable forward/reverse mode.

These APIs intentionally stay small so they can act as building blocks for larger pipelines. When you need additional knobs (custom solvers, on-the-fly tolerance changes) extend the bindings or call the Rust crate directly.
