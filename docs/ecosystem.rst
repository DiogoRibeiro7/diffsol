Ecosystem Integration
=====================

Diffsol’s PyTorch bindings are only the first piece of a larger interoperability story. This page outlines current compatibility with the broader scientific Python stack, plus recommendations for cloud and container deployments.

Scientific Python Stack
-----------------------

- **NumPy / SciPy** – Dense outputs are returned as Python lists, which can be reshaped into ``numpy.ndarray`` for downstream SciPy analyses. Jacobians and sensitivities coming from ``forward_mode`` can be fed into optimizers such as ``scipy.optimize.least_squares`` without conversion.
- **JAX / TensorFlow** – ``examples/integration/jax_bridge.py`` shows how to wrap diffsol via ``jax.experimental.host_callback``. A TensorFlow ``tf.py_function`` sample is planned next.
- **Pandas / xarray** – Solutions can be packaged into DataFrames for time-series pipelines. Use column-major reshape to map states into tidy data tables.

Machine-Learning Frameworks
---------------------------

- **PyTorch (official)** – Autograd-friendly adjoint and sensitivity APIs are exposed.
- **PyTorch Lightning / HuggingFace Accelerate** – treat diffsol calls as regular PyTorch ops. Since the solver runs on CPU, keep them outside ``torch.compile`` regions.
- **Future targets (roadmap)** – JAX custom call (via XLA FFI) and TensorFlow custom ops are in design. See GitHub Discussions ``#jax-integration`` for updates.

Cloud & Distributed Deployments
-------------------------------

- **Docker images** – the ``docker/`` directory (see ``docker/README.md``) contains base CPU/GPU images that install system LLVM + PyTorch bindings. Example build:

  .. code-block:: bash

     docker build -f docker/Dockerfile.cpu -t diffsol-pytorch:cpu .

- **Kubernetes / Batch** – bake the container above and mount your DiffSL scripts via ConfigMaps. Use environment variables to tune tolerances without rebuilding the image.
- **Serverless** – for short-running inference jobs, keep modules serialised (e.g., cache compiled DiffSL to disk) so you can warm-start lambdas/functions without invoking ``maturin``.
- **Distributed training** – wrap diffsol calls inside ``torch.distributed`` barriers or use RPC to offload ODE solves to dedicated CPU pools while GPUs handle the rest of the model.

Performance Optimisations
-------------------------

* Pin solver threads via ``RAYON_NUM_THREADS`` when running multi-tenant workloads to avoid noisy neighbours.
* Use batch parameter sweeps by reusing a single ``DiffsolModule`` inside a multiprocessing pool (each worker creates its own module to avoid PyO3 sharing issues).
* For cloud costs, profile with ``benchmarks/perf.py --device cpu`` on your instance type (e.g., AWS c7i) to pick the cheapest hardware that still meets latency targets.

Interoperability Plans
----------------------

1. **SciPy integrator drop-in** – expose ``diffsol.integrate.solve_ivp`` for scripts that currently call SciPy.
2. **ONNX / TorchScript export** – research storing solver outputs in ONNX nodes for deployment.
3. **DataFrames** – helper utilities to export trajectories into Parquet/Arrow for analytics pipelines.

If you rely on a specific framework or cloud provider not listed here, open an issue so we can prioritise it.
