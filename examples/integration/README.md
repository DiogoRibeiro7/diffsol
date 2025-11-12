# Integration Examples

These self‑contained scripts highlight where diffsol’s differentiable solvers give PyTorch users an advantage over black‑box ODE integrators. Every example runs on CPU by default and only depends on the PyTorch bindings (`pip install -e bindings/diffsol-pytorch` after building the crate).

> **Tip:** All commands below assume you are inside the repository root and have the virtualenv activated as `.venv`.

---

## Neural ODE Image Classifier

- **Path:** `examples/integration/neural_ode/train.py`
- **What it shows:** Diffsol reverse‑mode sensitivities drive a logistic decay classifier over Fashion‑MNIST embeddings, while the optional torchdiffeq baseline integrates the same flow.
- **Why diffsol:** Reverse‑mode gradients come directly from the solver, so we avoid building an autograd graph over hundreds of solver steps—handy on long grids or stiff dynamics.
- **Run:**

```bash
python examples/integration/neural_ode/train.py --samples 256 --epochs 2
```

When `torchdiffeq` is installed, the script prints timings and accuracies for both backends.

---

## Continuous Normalizing Flow (CNF)

- **Path:** `examples/integration/cnf/mnist_cnf.py`
- **What it shows:** Training loop for a 1‑D CNF driven by diffsol reverse‑mode gradients. Falls back gracefully if torchvision/torchdiffeq aren’t available.
- **Why diffsol:** CNFs differentiate through the flow parameters at every step; adjoint checkpointing keeps memory flat while still providing exact parameter gradients.
- **Run:**

```bash
python examples/integration/cnf/mnist_cnf.py --steps 10
```

The script reports diffsol vs torchdiffeq wall‑clock times when both backends are present.

---

## Physics‑Informed Neural Network (PINN)

- **Path:** `examples/integration/pinn/train.py`
- **What it shows:** A small PINN regresses residuals of a 1‑D wave equation while matching diffsol’s dense reference solution.
- **Why diffsol:** Having fast, differentiable solvers lets you penalise PDE residuals against high‑accuracy references without writing custom autograd code.
- **Run:**

```bash
python examples/integration/pinn/train.py
```

Modify the network depth/width or time grid to explore stability vs accuracy trade‑offs.

---

## Parameter Estimation with Sensitivities

- **Path:** `examples/integration/parameter_estimation/estimate.py`
- **What it shows:** Inverse problem for an exponential decay model where diffsol forward sensitivities drive a simple Adam optimizer.
- **Why diffsol:** Forward sensitivities yield exact Jacobian‑vector products, so gradient steps remain stable even for noisy observations without finite differences.
- **Run:**

```bash
python examples/integration/parameter_estimation/estimate.py --steps 60
```

The script prints the parameter trajectory and loss, demonstrating convergence toward the ground truth without ever building large computational graphs.

---

## Notes

- All examples default to double precision to match diffsol’s solver tolerances. Feel free to change to `float32` if you want faster baselines; just remember to adjust tolerances accordingly.
- If you install `torchdiffeq`, the logistic classifier and CNF automatically include it in their comparisons. Otherwise only the diffsol path runs.
- Large datasets (MNIST) are downloaded on demand into `examples/integration/data/`.
