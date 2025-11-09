Benchmarks
==========

The ``benchmarks/perf.py`` script now compares diffsol against ``torchdiffeq`` for
non-stiff (logistic decay) and stiff (Van der Pol) problems and can also run a
neural ODE benchmark on MNIST. ``benchmarks/gpu_validation.py`` checks GPU execution:

.. code-block:: bash

    python benchmarks/perf.py --device cpu --mnist
    python benchmarks/gpu_validation.py --device cuda

The performance script writes Markdown tables to ``benchmarks/results.md`` summarizing
wall-clock times, gradient accuracy, energy spans, and neural ODE accuracy so these
numbers can be embedded into the documentation before publishing new results.
