"""
Parameter estimation for a dynamical system using diffsol forward sensitivities.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import torch

from diffsol_pytorch import DiffsolModule, testing

DECAY_CODE = """
in = [k]
k { 0.4 }
u {
    x = 1.0,
}
F {
    -k * x,
}
"""


def synthetic_data(k_true=0.4, noise=0.01) -> Tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 2.0, 80)
    clean = np.exp(-k_true * times)
    noisy = clean + noise * np.random.randn(times.size)
    return times, noisy


def estimate(times: np.ndarray, obs: np.ndarray, steps: int = 50):
    module = DiffsolModule(DECAY_CODE)
    params = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=5e-2)
    for step in range(steps):
        forward = testing.forward_mode(module, params.detach().tolist(), times.tolist())
        pred = forward.solution[0]
        residual = pred - obs
        loss = 0.5 * float(np.mean(residual**2))
        grad_sol = residual.reshape(1, -1) / residual.size
        grads = np.einsum("pij,ij->p", forward.sensitivities, grad_sol)
        optimizer.zero_grad()
        params.grad = torch.tensor(grads, dtype=params.dtype)
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:03d} k={params.item():.4f} loss={loss:.6f}")
    return params.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    times, obs = synthetic_data()
    k_hat = estimate(times, obs, steps=args.steps)
    print(f"estimated decay constant: {k_hat:.4f}")


if __name__ == "__main__":
    main()
