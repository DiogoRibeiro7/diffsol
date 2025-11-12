Scientific Examples
===================

The repository ships with fully-working examples aimed at both ML practitioners and domain scientists. Each walkthrough pairs the math with an executable script so you can reproduce the figures, benchmark diffsol against ``torchdiffeq``, and plug the outputs into your own training loops.

Neural ODE Classifier
---------------------

*Path:* ``examples/integration/neural_ode/train.py``

**Model.** We treat a logistic decay ODE

.. math::

   \frac{du}{dt} = -k(x) u,\quad u(0) = 1

where ``k(x)`` is produced by a convolutional encoder applied to Fashion‑MNIST images. The final value ``u(1)`` serves as the class probability.

**What diffsol does.** ``DiffsolModule`` integrates the decay for every sample, while ``reverse_mode`` supplies adjoint gradients with respect to the scalar rate ``k``. Torchdiffeq users ordinarily rely on autograd to backpropagate through hundreds of tiny solver steps; diffsol collapses that into a single adjoint solve.

**How to run.**

.. code-block:: bash

   python examples/integration/neural_ode/train.py --samples 512 --epochs 3 --device cpu

Set ``--device cuda`` and install ``torchdiffeq`` to see the comparative timings printed at the end of training.

Physics-Informed Neural Network (PINN)
--------------------------------------

*Path:* ``examples/integration/pinn/train.py``

**Problem.** We approximate the solution to the 1‑D wave equation ``u_tt = c^2 u`` by minimising both the PDE residual and the mismatch with diffsol's dense reference trajectory. The PINN is a small fully-connected network taking space/time coordinates as inputs.

**Workflow.**

1. diffsol produces a high-resolution reference trajectory for ``c = 1``.
2. At every training step we evaluate the PINN residual via PyTorch autograd and add a data term that matches the diffsol trajectory at the same time grid.

**Takeaways.** By letting diffsol handle the PDE integration we focus on the neural ansatz and residuals rather than worrying about numerical stability; more importantly, the example demonstrates how to supervise PINNs with physical trajectories rather than raw data.

Continuous Normalizing Flow
---------------------------

*Path:* ``examples/integration/cnf/mnist_cnf.py``

**Setup.** A one-dimensional CNF evolves latent noise through

.. math::

   \frac{dz}{dt} = a(t) z^3 + b(t) z + c(t)

with parameters stored in a PyTorch ``nn.Parameter`` vector. Diffsol integrates the ODE for a batch of samples and ``reverse_mode`` pushes gradients back to ``a,b,c``.

**Highlights.**

* Uses the same API regardless of whether ``torchdiffeq`` is installed; if it is, the script prints a timing comparison.
* Shows how to call ``DiffsolModule`` repeatedly inside an optimiser loop without paying the compilation cost each step.

**Command.**

.. code-block:: bash

   python examples/integration/cnf/mnist_cnf.py --steps 20

Parameter Estimation (Biological/Economic)
------------------------------------------

*Path:* ``examples/integration/parameter_estimation/estimate.py``

**Model.** Noisy measurements of exponential decay ``x' = -k x`` mimic a drug-clearance or capital depreciation process. We fit ``k`` by minimising a mean-squared loss between diffsol predictions and the synthetic observations.

**Sensitivities.** The example leans on ``forward_mode`` to produce derivatives of the trajectory with respect to ``k``. This yields exact gradients even when the measurements are sparse or the loss landscape is ill-conditioned—unlike finite differences, nothing blows up when we tighten tolerances.

**Usage.**

.. code-block:: bash

   python examples/integration/parameter_estimation/estimate.py --steps 60

The console prints the parameter trajectory and the loss every 10 steps, illustrating convergence to the true rate.

Additional Resources
--------------------

- ``examples/scientific`` covers canonical dynamical systems (Van der Pol, Lorenz, Hamiltonian, DAEs) and doubles as regression tests for the bindings.
- ``examples/integration/README.md`` summarises each integration example, its mathematical context, and command-line arguments.
- ``examples/integration/jax_bridge.py`` illustrates how to embed diffsol inside JAX pipelines via ``host_callback``.

Feel free to copy these scripts into your own projects—they are intentionally self-contained and rely only on PyTorch plus the ``diffsol_pytorch`` module.
