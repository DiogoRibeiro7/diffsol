"""
Continuous normalizing flow on MNIST using diffsol reverse-mode gradients.
Falls back gracefully if torchvision/torchdiffeq are unavailable.
"""

import argparse
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torchvision
    from torchvision import transforms
except ImportError:
    torchvision = None

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    torch_odeint = None

import diffsol_pytorch as dsp

CNF_CODE = """
in = [a, b, c]
a { 0.1 }
b { 0.1 }
c { 0.0 }
u {
    z = 0.0,
}
F {
    a * z * z * z + b * z + c,
}
"""


def load_mnist(batch_size: int = 256) -> Optional[torch.utils.data.DataLoader]:
    if torchvision is None:
        return None
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = torchvision.datasets.MNIST("./data", download=True, transform=transform)
    return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)


class DiffsolCNF(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.randn(3, dtype=torch.float64) * 0.01)
        self.module = dsp.DiffsolModule(CNF_CODE)
        self.times = torch.linspace(0.0, 1.0, 51).tolist()

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        nout, nt, flat = self.module.solve_dense(
            self.params.detach().tolist(), self.times
        )
        traj = torch.tensor(flat, dtype=torch.float64).reshape(nout, nt)
        return traj[-1]

    def log_prob(self, z0: torch.Tensor) -> torch.Tensor:
        zT = self.forward(z0)
        return -0.5 * (zT**2).mean()

    def backward(self, grad_scalar: float):
        grad_out = [grad_scalar] * len(self.times)
        grads = dsp.reverse_mode(
            CNF_CODE,
            self.params.detach().tolist(),
            self.times,
            grad_out,
        )
        self.params.grad = torch.tensor(grads[:3], dtype=self.params.dtype)


def train_diffsol(loader, steps: int, device: torch.device):
    model = DiffsolCNF().to(device)
    optimizer = optim.Adam([model.params], lr=1e-2)
    data_iter = iter(loader)
    start = time.perf_counter()
    for step in range(steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, _ = next(data_iter)
        x = x.to(device, dtype=torch.float64).view(x.size(0), -1)
        noise = torch.randn_like(x[:, :1])
        loss = -model.log_prob(noise)
        optimizer.zero_grad()
        model.backward(float(loss.item()))
        optimizer.step()
    return time.perf_counter() - start


def train_torchdiffeq(loader, steps: int, device: torch.device):
    if torch_odeint is None:
        return float("nan")
    func = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)).to(
        device, dtype=torch.float64
    )
    optimizer = optim.Adam(func.parameters(), lr=1e-2)
    times = torch.linspace(0.0, 1.0, 51, device=device, dtype=torch.float64)
    data_iter = iter(loader)
    start = time.perf_counter()
    for step in range(steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, _ = next(data_iter)
        z0 = torch.randn(x.size(0), 1, device=device, dtype=torch.float64)
        traj = torch_odeint(lambda t, z: func(z), z0, times)
        loss = traj[-1].pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    loader = load_mnist()
    if loader is None:
        raise SystemExit("torchvision is required for the MNIST CNF example.")
    device = torch.device(args.device)
    diffsol_time = train_diffsol(loader, args.steps, device)
    print(f"diffsol CNF training time: {diffsol_time:.2f}s")
    torch_time = train_torchdiffeq(loader, args.steps, device)
    if not math.isnan(torch_time):
        print(f"torchdiffeq CNF training time: {torch_time:.2f}s")
    else:
        print("torchdiffeq not installed; skipping comparison.")


if __name__ == "__main__":
    main()
