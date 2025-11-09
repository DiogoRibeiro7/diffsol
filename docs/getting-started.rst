Getting Started
===============

Installation
------------

.. code-block:: bash

    pip install diffsol-pytorch

Example
-------

.. code-block:: python

    import torch
    import diffsol_pytorch as dsp

    code = """
    state x
    param k
    der(x) = -k * x
    """

    module = dsp.DiffsolModule(code)
    params = torch.tensor([0.1], dtype=torch.float64)
    times = torch.linspace(0, 1, 5, dtype=torch.float64)
    solution = module.solve_dense(params, times)
    print(solution)

Reverse-mode gradients integrate seamlessly with PyTorch tensors:

.. code-block:: python

    grad_out = torch.zeros((1, times.numel()), dtype=torch.float64)
    grad_out[0, -1] = 1.0
    grad_params, grad_init = dsp.reverse_mode(
        code,
        params.tolist(),
        times.tolist(),
        grad_out.flatten().tolist(),
    )

Scientific validation and integration demos live under ``examples/``:

- ``examples/scientific/``: Lorenz, Van der Pol, Hamiltonian, DAE, and sensitivity studies.
- ``examples/integration/``: neural ODE classifier, continuous normalizing flow, physics-informed neural network, and parameter estimation pipelines.

Each script prints diagnostic information so you can verify accuracy or reproduce the performance comparisons described in this guide.
