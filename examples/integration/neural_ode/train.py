"""
Toy neural ODE classifier comparing diffsol against torchdiffeq.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    torch_odeint = None

import diffsol_pytorch as dsp


class ODEFunc(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t, x):
        return self.net(x)


def make_spiral(batch=128):
    theta = torch.sqrt(torch.rand(batch)) * 2 * torch.pi
    radius = 2 * theta + torch.pi
    data = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    data += 0.1 * torch.randn_like(data)
    labels = (theta > torch.pi).long()
    return data, labels


def train_diffsol(steps=3):
    code = """
state x
state y
param w0
param w1
param b0
der(x) = w0 * x + w1 * y + b0
der(y) = -w1 * x + w0 * y
"""
    module = dsp.DiffsolModule(code)
    optimizer = optim.Adam([torch.randn(3, requires_grad=True)], lr=1e-2)
    times = torch.linspace(0.0, 1.0, 21).tolist()
    for step in range(steps):
        x, y = make_spiral()
        params = optimizer.param_groups[0]["params"][0]
        nout, nt, flat = module.solve_dense(params.tolist(), times)
        traj = torch.tensor(flat[-2:], dtype=torch.float64).reshape(2, nt)
        loss = traj.pow(2).mean()
        grad_out = torch.autograd.grad(loss, traj, retain_graph=True)[0].reshape(-1).tolist()
        grads = dsp.reverse_mode(code, params.tolist(), times, grad_out)
        optimizer.zero_grad()
        params.grad = torch.tensor(grads, dtype=params.dtype)
        optimizer.step()
    return traj


def train_torchdiffeq(steps=3):
    if torch_odeint is None:
        return None
    func = ODEFunc()
    optimizer = optim.Adam(func.parameters(), lr=1e-2)
    times = torch.linspace(0.0, 1.0, 21)
    for step in range(steps):
        x, y = make_spiral()
        traj = torch_odeint(func, x, times)
        loss = traj.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    start = time.perf_counter()
    train_diffsol(args.steps)
    diffsol_time = time.perf_counter() - start
    print(f"diffsol training finished in {diffsol_time:.2f}s")
    if torch_odeint is not None:
        start = time.perf_counter()
        train_torchdiffeq(args.steps)
        print(f"torchdiffeq training finished in {time.perf_counter() - start:.2f}s")
    else:
        print("torchdiffeq not available")


if __name__ == "__main__":
    main()
