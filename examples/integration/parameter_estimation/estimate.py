"""
Parameter estimation for a dynamical system using diffsol sensitivities.
"""

import numpy as np
import torch

import diffsol_pytorch as dsp

DECAY_CODE = """
state x
param k
der(x) = -k * x
"""


def synthetic_data(k_true=0.5, noise=0.01):
    times = np.linspace(0.0, 2.0, 40)
    signal = np.exp(-k_true * times)
    noisy = signal + noise * np.random.randn(*signal.shape)
    return times.tolist(), noisy


def loss_for_params(k, times, obs):
    module = dsp.DiffsolModule(DECAY_CODE)
    nout, nt, flat = module.solve_dense([k], times)
    pred = np.array(flat, dtype=float)
    return 0.5 * np.mean((pred - obs) ** 2)


def estimate():
    times, obs = synthetic_data()
    k = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([k], lr=1e-1)
    for _ in range(50):
        loss = loss_for_params(k.item(), times, obs)
        optimizer.zero_grad()
        loss_tensor = torch.tensor(loss, dtype=k.dtype, requires_grad=True)
        grad_out = [1.0] * len(times)
        grads = dsp.reverse_mode(DECAY_CODE, [k.item()], times, grad_out)
        k.grad = torch.tensor([grads[0]], dtype=k.dtype)
        optimizer.step()
        print(f"k={k.item():.4f}, loss={loss:.6f}")


if __name__ == "__main__":
    estimate()
