Scientific Examples
===================

The repository ships with fully-working examples aimed at both ML practitioners and domain scientists. Each walkthrough pairs the math with an executable script so you can reproduce the figures, benchmark diffsol against ``torchdiffeq``, and plug the outputs into your own training loops.

Interactive notebooks
---------------------

Every example below also has a Jupyter notebook under ``docs/notebooks`` (e.g.
``docs/notebooks/neural_ode_classifier.ipynb``). Launch them via:

.. code-block:: bash

   pip install -r docs/requirements.txt jupyter
   jupyter notebook docs/notebooks

The notebooks mirror the CLI scripts and include extra visualisations plus widgets for experimenting with parameters. GPU-only sections rely on ``helpers.gpu_section_mode`` to execute on CUDA/MPS when available, fall back to cached metrics under ``docs/notebooks/_cache``, or skip gracefully in CPU-only environments. The nightly ``notebooks`` workflow refreshes those caches so CI stays deterministic.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/hello_diffsol
   notebooks/neural_ode_classifier
   notebooks/pinn_wave_equation
   notebooks/cnf_mnist
   notebooks/parameter_estimation

Notebook gallery
----------------

.. list-table::
   :header-rows: 1

   * - Notebook
     - Highlights
     - Link
   * - Intro & gradients
     - Logistic ODE, gradient validation utilities
     - :doc:`notebooks/hello_diffsol`
   * - Neural ODE classifier
     - Fashion-MNIST workflow, PyTorch interop, reverse-mode gradients
     - :doc:`notebooks/neural_ode_classifier`
   * - PINN wave equation
     - Residual loss fitting and comparison to diffsol trajectory
     - :doc:`notebooks/pinn_wave_equation`
   * - CNF demo
     - Continuous normalizing flow parameters trained with diffsol reverse-mode
     - :doc:`notebooks/cnf_mnist`
   * - Parameter estimation
     - Exponential decay fitting with forward sensitivities
     - :doc:`notebooks/parameter_estimation`

Neural ODE Classifier
---------------------

*Path:* ``examples/integration/neural_ode/train.py`` (notebook: ``docs/notebooks/neural_ode_classifier.ipynb``)

**Model.** We treat a logistic decay ODE

.. math::

   \frac{du}{dt} = -k(x) u,\quad u(0) = 1

where ``k(x)`` is produced by a convolutional encoder applied to Fashionâ€‘MNIST images. The final value ``u(1)`` serves as the class probability.

**What diffsol does.** ``DiffsolModule`` integrates the decay for every sample, while ``reverse_mode`` supplies adjoint gradients with respect to the scalar rate ``k``. Torchdiffeq users ordinarily rely on autograd to backpropagate through hundreds of tiny solver steps; diffsol collapses that into a single adjoint solve.

**How to run.**

.. code-block:: bash

   python examples/integration/neural_ode/train.py --samples 512 --epochs 3 --device cpu

Set ``--device cuda`` and install ``torchdiffeq`` to see the comparative timings printed at the end of training.

Physics-Informed Neural Network (PINN)
--------------------------------------

*Path:* ``examples/integration/pinn/train.py`` (notebook: ``docs/notebooks/pinn_wave_equation.ipynb``)

**Problem.** We approximate the solution to the 1â€‘D wave equation ``u_tt = c^2 u`` by minimising both the PDE residual and the mismatch with diffsol's dense reference trajectory. The PINN is a small fully-connected network taking space/time coordinates as inputs.

**Workflow.**

1. diffsol produces a high-resolution reference trajectory for ``c = 1``.
2. At every training step we evaluate the PINN residual via PyTorch autograd and add a data term that matches the diffsol trajectory at the same time grid.

**Takeaways.** By letting diffsol handle the PDE integration we focus on the neural ansatz and residuals rather than worrying about numerical stability; more importantly, the example demonstrates how to supervise PINNs with physical trajectories rather than raw data.

Continuous Normalizing Flow
---------------------------

*Path:* ``examples/integration/cnf/mnist_cnf.py`` (notebook: ``docs/notebooks/cnf_mnist.ipynb``)

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

*Path:* ``examples/integration/parameter_estimation/estimate.py`` (notebook: ``docs/notebooks/parameter_estimation.ipynb``)

**Model.** Noisy measurements of exponential decay ``x' = -k x`` mimic a drug-clearance or capital depreciation process. We fit ``k`` by minimising a mean-squared loss between diffsol predictions and the synthetic observations.

**Sensitivities.** The example leans on ``forward_mode`` to produce derivatives of the trajectory with respect to ``k``. This yields exact gradients even when the measurements are sparse or the loss landscape is ill-conditionedâ€”unlike finite differences, nothing blows up when we tighten tolerances.

**Usage.**

.. code-block:: bash

   python examples/integration/parameter_estimation/estimate.py --steps 60

The console prints the parameter trajectory and the loss every 10 steps, illustrating convergence to the true rate.

Additional Resources
--------------------

- ``examples/scientific`` covers canonical dynamical systems (Van der Pol, Lorenz, Hamiltonian, DAEs) and doubles as regression tests for the bindings.
- ``examples/integration/README.md`` summarises each integration example, its mathematical context, and command-line arguments.
- ``examples/integration/jax_bridge.py`` illustrates how to embed diffsol inside JAX pipelines via ``host_callback``.

Feel free to copy these scripts into your own projectsâ€”they are intentionally self-contained and rely only on PyTorch plus the ``diffsol_pytorch`` module.
