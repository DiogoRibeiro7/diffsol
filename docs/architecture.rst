Architecture Overview
=====================

The diffsol stack consists of a Rust core (solver + autodiff) wrapped by a thin PyO3 layer and an optional
PyTorch convenience package. The diagram below highlights how data flows between these layers when you call
``device.solve_dense_tensor`` and ``device.reverse_mode_tensor``.

.. mermaid::

   graph TD
     A[PyTorch tensors (CPU/GPU)] -->|to_cpu()| B[Host buffers (Rust)]
     B --> C[DiffsolModule (Rust solver)]
     C -->|dense solution| B
     B -->|torch.from_numpy| A
     A -.->|grad_output| D[reverse_mode_tensor]
     D -->|CPU gradients| B
     B -->|copy back| A

Key components
--------------

* **Diffsol core (Rust)** – owns the DSL parser, solver implementations (BDF/SDIRK/ERK), checkpointing, and
  Enzyme-based adjoint infrastructure. It is agnostic of PyTorch.
* **PyO3 bindings** – expose ``DiffsolModule`` and ``reverse_mode`` to Python. The bindings focus on error
  translation and host buffer management.
* **Python helpers** – ``diffsol_pytorch.device`` handles device detection, host/GPU transfers, and exposes a
  friendly API for PyTorch workflows. ``diffsol_pytorch.testing`` contains gradient validation routines used in
  unit tests and documentation.

Autodiff data flow
------------------

.. mermaid::

   sequenceDiagram
     participant Torch as PyTorch
     participant Device as device.solve_dense_tensor
     participant Rust as DiffsolModule
     participant Enzyme as Enzyme adjoint

     Torch->>Device: params, times, grad_output
     Device->>Rust: host params/times
     Rust->>Rust: forward solve (BDF/SDIRK)
     Rust->>Torch: dense solution
     Torch->>Device: grad_output.reshape(-1)
     Device->>Enzyme: reverse_mode call
     Enzyme-->>Device: gradients wrt params + initial state
     Device-->>Torch: gradients (same device as inputs)

Runtime invariants
------------------

* All numerical kernels execute on CPU; GPU involvement is limited to copying tensors to/from host memory.
* ``DiffsolModule`` instances are cloneable and inexpensive to reuse across minibatches.
* Error reporting flows through ``TorchDiffsolError`` → ``PyErr`` so Python callers always see actionable messages.

Integrating from other languages (JAX, Julia, C++)
-------------------------------------------------

The PyO3 layer is intentionally thin; alternative front-ends can talk directly to the Rust crate by linking
against ``diffsol`` and using the same ``DiffsolModule`` abstractions. The architectural rules of thumb remain the
same:

* Copy tensors to CPU before invoking the solver.
* Reuse compiled modules and solvers to amortise JIT cost.
* Prefer the adjoint APIs (``bdf_solver_adjoint`` et al.) over reimplementing reverse-mode yourself.
