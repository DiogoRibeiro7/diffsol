<div align="center">
<p></p><img src="https://raw.githubusercontent.com/martinjrobins/diffsol/refs/heads/main/assets/diffsol_rectangle.svg" alt="Diffsol logo" width="300"/></p>
<a href="https://docs.rs/diffsol">
    <img src="https://img.shields.io/crates/v/diffsol.svg?label=docs&color=blue&logo=rust" alt="docs.rs badge">
</a>
<a href="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml">
    <img src="https://github.com/martinjrobins/diffsol/actions/workflows/rust.yml/badge.svg" alt="CI build status badge">
</a>
<a href="https://codecov.io/gh/martinjrobins/diffsol">
    <img src="https://codecov.io/gh/martinjrobins/diffsol/branch/main/graph/badge.svg" alt="code coverage">
</a>
</div>

---

Diffsol is a library for solving ordinary differential equations (ODEs) or semi-explicit differential algebraic equations (DAEs) in Rust. It can solve equations in the following form:

```math
M \frac{dy}{dt} = f(t, y, p)
```

where $M$ is a (possibly singular and optional) mass matrix, $y$ is the state vector, $t$ is the time and $p$ is a vector of parameters.

The equations can be given by either rust code or the [DiffSL](https://martinjrobins.github.io/diffsl/) Domain Specific Language (DSL). The DSL uses automatic differentiation using [Enzyme](https://enzyme.mit.edu/) to calculate the necessary jacobians, and JIT compilation (using either [LLVM](https://llvm.org/) or [Cranelift](https://cranelift.dev/)) to generate efficient native code at runtime. The DSL is ideal for using Diffsol from a higher-level language like Python or R while still maintaining similar performance to pure rust.

## Installation and Usage

See installation instructions on the [crates.io page](https://crates.io/crates/diffsol).

The [Diffsol book](https://martinjrobins.github.io/diffsol/) describes how to use Diffsol using examples taken from several application areas (e.g. population dynamics, electrical circuits and pharmacological modelling), as well as more detailed information on the various APIs used to specify the ODE equations. For a more complete description of the API, please see the [docs.rs API documentation](https://docs.rs/diffsol).

## System Requirements

- 64-bit Linux (glibc 2.31+), macOS 13+, or Windows 10/11 with the MSVC toolchain.
- Rust 1.80+ (via `rustup`) and Python 3.9+ with a working PyTorch installation.
- Command-line build tools: `cmake` 3.24+, Ninja (Windows only), and a recent `clang`/Visual Studio build chain.
- At least 15â€¯GB of free disk space for the LLVM/Enzyme toolchain plus build artifacts.

## LLVM/Enzyme Toolchain

DiffSLâ€™s LLVM backend requires a matching set of LLVM libraries and Enzyme static archives. The repository ships a helper that downloads the official LLVM binaries and validates your environment before compiling the Rust crates:

```bash
python tools/prepare_toolchain.py fetch --version 18.1.8 --install-dir .deps --export-prefix
setx LLVM_SYS_181_PREFIX C:\path\to\diffsol\.deps\clang+llvm-18.1.8-x86_64-pc-windows-msvc  # Windows
export LLVM_SYS_181_PREFIX=$PWD/.deps/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-22.04     # Linux
```

The tool also exposes a `check` sub-command that verifies the LLVM version and the availability of `cmake`/`ninja`:

```bash
python tools/prepare_toolchain.py check --prefix %LLVM_SYS_181_PREFIX%
```

Build scripts now enforce the same checks. Set the following environment variables to tweak the behaviour:

- `DIFFSOL_SKIP_TOOLCHAIN_CHECK=1` â€“ skip the `cmake`/`ninja` detection performed by `bindings/diffsol-pytorch/build.rs`.
- `DIFFSOL_ALLOW_UNSUPPORTED_LLVM=1` â€“ opt into building against LLVM majors other than the validated list (currently 18).
- `DIFFSOL_SKIP_LLVM_VERSION_CHECK=1` â€“ bypass the LLVM version gate inside the vendored `diffsl` Enzyme build helper.

On Windows the Enzyme build automatically produces both `EnzymeStatic-<version>.lib` and a compatibility `Enzyme-<version>.lib` so MSVC linkers can satisfy either naming scheme.

## Memory Diagnostics

The bindings now ship an opt-in diagnostics feature set to track host buffers and Enzyme tape allocations:

- `cargo test -p diffsol-pytorch --features diagnostics --test memory_stress` exercises the Rust leak guards.
- `pytest bindings/diffsol-pytorch/tests -m slow` runs the long-duration PyTorch stress suite.
- `scripts/leakcheck.sh` downloads LLVM (if needed), builds the `leak_harness` binary with jemalloc instrumentation, and runs it under Valgrind (`llvm` + `mem-profiling` features). The harness fails the build if either the host-buffer tracker or jemalloc reports a leak.

On macOS you can pair the diagnostics feature with `leaks` or Instruments, while Windows users can launch the `leak_harness` binary under Dr. Memory after setting `LLVM_SYS_181_PREFIX` to their local toolchain.

## Device Management

`diffsol_pytorch.device` centralises accelerator detection and the tensor copies needed to keep PyTorch
workflows ergonomic. The solver itself still executes on CPU, but the helpers automatically move tensors
to host memory before the FFI call and materialise outputs back on the requested device (CUDA, ROCm, Metal,
or CPU).

```python
import torch
from diffsol_pytorch import DiffsolModule, device

module = DiffsolModule(LOGISTIC_CODE)
params = torch.tensor([0.7], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
times = torch.linspace(0.0, 1.0, 128, dtype=torch.float64, device=params.device)

sol = device.solve_dense_tensor(module, params, times)  # returned on the same device
grad = device.reverse_mode_tensor(LOGISTIC_CODE, params, times, torch.ones_like(sol))

print(device.supported_backends_message())
```

Key capabilities:

- Automatic backend detection (`CUDA`, `ROCm`, `MPS`) with graceful CPU fallback
- Device-aware buffer management for `solve_dense` and `reverse_mode`
- Helper text (``supported_backends_message``) to surface availability in notebooks/UIs

## Documentation

- `docs/install.rst` -- platform-specific setup, dependencies, and validation steps
- `docs/performance.rst` -- solver/device tuning strategies and benchmarking workflows
- `docs/troubleshooting.rst` -- common build/runtime issues with actionable fixes
- `docs/architecture.rst` -- Rust/Python data-flow diagrams for integrators
- `docs/migration.rst` -- guidance for moving workloads from `torchdiffeq` to diffsol
- `docs/scientific-examples.rst` -- CLI walkthroughs plus matching Jupyter notebooks under `docs/notebooks/`

## Tutorials & Notebooks

Interactive notebooks live in `docs/notebooks/` and render automatically in the Sphinx docs:

- `hello_diffsol.ipynb` -- introductory logistic ODE + gradient check
- `neural_ode_classifier.ipynb` -- Fashion-MNIST neural ODE demo with PyTorch integration
- `pinn_wave_equation.ipynb` -- physics-informed neural network for a 1-D wave equation
- `cnf_mnist.ipynb` -- continuous normalizing flow training loop with diffsol reverse-mode
- `parameter_estimation.ipynb` -- exponential-decay fitting using forward sensitivities

Each notebook imports shared utilities from `docs/notebooks/helpers.py`, which centralise seeding, device selection, and cached datasets/results. GPU-only sections call `gpu_section_mode`, so CPU-only runs (including CI) automatically reuse artifacts under `docs/notebooks/_cache` or skip gracefully; set `NB_FORCE_GPU=1` if you want those sections to fail when no accelerator is present.

The nightly `.github/workflows/notebooks.yml` job executes the full suite through `nbmake`, caches notebook data (`_cache`, `_data`), and surfaces the duration of each run. Trigger it manually from GitHub's UI and pass `fail_on_warning=false` to inspect warnings without failing the workflow.

## Development Workflow

We rely on [`pre-commit`](https://pre-commit.com) for linting and formatting (Ruff, Mypy, trailing whitespace, end-of-file fixes). Install the hooks once per clone:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## CI/CD Pipeline

- `ci.yml` runs on every PR/push with cached LLVM/Enzyme toolchains, multi-platform Rust/Python builds, pytest coverage/JUnit artifacts, docs builds, and a valgrind/jemalloc-based leak harness.
- `wheels.yml` builds release wheels (Linux, macOS, Windows) with cached toolchains and publishes them to both PyPI and GitHub Releases whenever a `v*` tag is pushed.
- `notebooks.yml` executes the full tutorial suite via nbmake (nightly + on demand) while reusing cached notebook data under `docs/notebooks/_cache`.
- `perf.yml` executes `scripts/run_validation.py --perf`, archives the `pytest-benchmark` output, and pushes historical trends to the dedicated `benchmarks` branch for regression tracking.
- `security.yml` runs `cargo audit`, `pip-audit`, and Trivy weekly (or on demand) and fails the workflow on newly discovered CVEs.
- Dependabot (`.github/dependabot.yml`) keeps Cargo, pip, and workflow dependencies fresh with weekly PRs so releases stay reproducible across platforms.

## Testing & Validation

The repository ships a multi-tier test suite covering unit, integration, gradient, performance, and memory-leak scenarios.

- `python scripts/run_validation.py` â€“ orchestrates unit/integration/gradient tests, memory leak checks, and the Rust diagnostics harness (runs on CI).
- `pytest -m "unit or integration or gradient"` â€“ quick developer loop (coverage enabled by default via `pytest.ini`).
- `pytest -m memory` â€“ psutil-backed leak checks plus long-running stress tests.
- `pytest -m perf --run-perf --benchmark-json perf.json` â€“ opt-in benchmarks using `pytest-benchmark` and `torchdiffeq` (skipped unless `--run-perf` is set).
- `cargo test -p diffsol-pytorch --features diagnostics --test memory_stress` â€“ low-level Rust leak harness (also invoked by `run_validation.py`).

The gradient suite relies on `diffsol_pytorch.testing` (finite differences + forward/reverse AD), while performance tests warn when `diffsol` falls behind `torchdiffeq`. Coverage reports are emitted via `pytest-cov`, and CI publishes the summary in the test logs.

## Error Handling

`diffsol-pytorch` surfaces every failure as a structured `TorchDiffsolError` belonging to one of three categories:

| Category  | When it triggers                                   | Typical fix                                                                 |
|-----------|-----------------------------------------------------|------------------------------------------------------------------------------|
| **Build** | DiffSL parsing/compilation, solver initialisation   | Fix syntax errors or rebuild with `--features diffsl-llvm` for autodiff     |
| **Autodiff** | Forward/reverse mode setup and adjoint checkpoints | Ensure LLVM/Enzyme is installed, provide consistent gradients/shapes         |
| **Runtime** | Dense solves, interpolation, Python bindings      | Adjust tolerances, pass non-empty time grids, ensure gradients match output |

Every variant carries an optional suggestion so users know the next action. Python bindings automatically convert these errors to `ValueError` (shape issues) or `RuntimeError` while preserving the help text:

```python
>>> module = diffsol_pytorch.DiffsolModule("bad code")
RuntimeError: build failure during DiffSL parsing/compilation: expected `}` near line 3

Help: Run `diffsl bad.dsl` locally or rebuild with `--features diffsl-llvm`
```

From Rust you can match on the variants to provide custom recovery:

```rust
match module.solve_dense(&params, &times) {
    Ok((nout, nt, flat)) => { /* ... */ }
    Err(err @ TorchDiffsolError::Autodiff { .. }) => {
        eprintln!("{err}");
        if let Some(help) = err.suggestion() {
            eprintln!("Suggestion: {help}");
        }
    }
    Err(err) => return Err(err),
}
```

## Features

The following solvers are available in Diffsol

1. A variable order Backwards Difference Formulae (BDF) solver, suitable for stiff problems and singular mass matrices. The basic algorithm is derived in [(Byrne & Hindmarsh, 1975)](#1), however this particular implementation follows that implemented in the Matlab routine ode15s [(Shampine & Reichelt, 1997)](#4) and the SciPy implementation [(Virtanen et al., 2020)](#5), which features the NDF formulas for improved stability
2. A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver, suitable for moderately stiff problems and singular mass matrices. Two different butcher tableau are provided, TR-BDF2 [(Hosea & Shampine, 1996)](#2) and ESDIRK34 [(JÃ¸rgensen et al., 2018)](#3), or users can supply their own.
3. A variable order Explict Runge-Kutta (ERK) solver, suitable for non-stiff problems. One butcher tableau is provided, the 4th order TSIT45 [(Tsitouras, 2011)](#5), or users can supply their own.

All solvers feature:

- Linear algebra containers and linear solvers from the nalgebra or faer crates, including both dense and sparse matrix support.
- Adaptive step-size control to given relative and absolute tolerances. Tolerances can be set separately for the main equations, quadrature of the output function, and sensitivity analysis.
- Dense output, interpolating to times provided by the user.
- Event handling, stopping when a given condition $g_e(t, y , p)$ is met or at a specific time.
- Numerical quadrature of an optional output $g_o(t, y, p)$ function over time.
- Forward sensitivity analysis, calculating the gradient of an output function or the solver states $y$ with respect to the parameters $p$.
- Adjoint sensitivity analysis, calculating the gradient of cost function $G(p)$ with respect to the parameters $p$. The cost function can be the integral of a continuous output function $g(t, y, p)$ or a sum of a set of discrete functions $h_i(t_i, y_i, p)$ at time points $t_i$.

## Documentation & Community

- [`docs/user-guide.rst`](docs/user-guide.rst) â€“ installation, migration from ``torchdiffeq``, logging/troubleshooting tips.
- [`docs/scientific-examples.rst`](docs/scientific-examples.rst) â€“ neural ODEs, CNFs, PINNs, parameter estimation walkthroughs.
- [`docs/release-strategy.rst`](docs/release-strategy.rst) â€“ semantic versioning, compatibility matrix, deprecation policy.
- [`docs/community.rst`](docs/community.rst) â€“ discussions, chat, maintainer guidelines.
- [`docs/ecosystem.rst`](docs/ecosystem.rst) â€“ integration plans for JAX/TensorFlow, cloud deployments, container guidance.

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and open issues using the templates under `.github/ISSUE_TEMPLATE`. Join the discussion at <https://github.com/martinjrobins/diffsol/discussions>.

## Validation & Benchmarks

- `bindings/diffsol-pytorch/tests/test_gradients.py` now exercises finite differences, analytic checks, and forward/reverse-mode autodiff (skipped automatically when the build lacks LLVM/Enzyme support).
- `bindings/diffsol-pytorch/tests/test_scientific_validation.py` runs quick regression tests on Lorenz, Van der Pol, Hamiltonian mechanics, and DAE examples to ensure numerics stay within expected envelopes.
- `benchmarks/perf.py` produces Markdown summaries comparing diffsol against `torchdiffeq` (when installed) for nonâ€‘stiff vs stiff dynamics and a neural ODE block, including memory usage samples and gradient diagnostics.
- `benchmarks/gpu_validation.py` reports CPU vs CUDA timing/memory for a micro-problem so regressions are easy to spot.

Example usage:

```bash
.\.venv\Scripts\python.exe -m pytest bindings/diffsol-pytorch/tests
.\.venv\Scripts\python.exe benchmarks\perf.py
```

## Wanted - Developers for higher-level language wrappers

Diffsol is designed to be easy to use from higher-level languages like Python or R. I'd prefer not to split my focus away from the core library, so I'm looking for developers who would like to lead the development of these wrappers. If you're interested, please get in touch.

- [x] Python (e.g. using [PyO3](https://pyo3.rs/v0.24.0/)). <https://github.com/alexallmont/pydiffsol>.
- [ ] Python ML frameworks (e.g. [JAX](https://docs.jax.dev/en/latest/ffi.html), [PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html))
- [ ] R (e.g. using [extendr](https://extendr.github.io/)).
- [ ] Julia
- [ ] Matlab
- [ ] Javascript in backend (e.g using [Neon](https://neon-rs.dev/))
- [ ] Javascript in browser (e.g. using [wasm-pack](https://rustwasm.github.io/wasm-pack/))
- [ ] Others, feel free to suggest your favourite language.

## References

- <a id="1"></a> Byrne, G. D., & Hindmarsh, A. C. (1975). A polyalgorithm for the numerical solution of ordinary differential equations. ACM Transactions on Mathematical Software (TOMS), 1(1), 71â€“96.81
- <a id="2"></a> Hosea, M., & Shampine, L. (1996). Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20(1-2), 21â€“37.
- <a id="3"></a> JÃ¸rgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv Preprint arXiv:1803.01613.
- <a id="4"></a> Shampine, L. F., & Reichelt, M. W. (1997). The matlab ode suite. SIAM Journal on Scientific Computing, 18(1), 1â€“22.
- <a id="5"></a> Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., & others. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods, 17(3), 261â€“272.
 - <a id="5"></a> Tsitouras, C. (2011). Rungeâ€“Kutta pairs of order 5 (4) satisfying only the first column simplifying assumption. Computers & Mathematics with Applications, 62(2), 770-775.
