"""
Neural ODE image classification with diffsol reverse-mode gradients.

Each image is mapped to a scalar rate parameter that drives a one-dimensional
logistic ODE. The final value of the trajectory acts as the class probability.
For comparison we optionally run the same classifier using torchdiffeq.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

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

LOGISTIC_CODE = """
in = [k]
k { 0.5 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"""

TIMES = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
TIMES_LIST = TIMES.tolist()
GRAD_OUT_FINAL = [0.0] * (len(TIMES_LIST) - 1) + [1.0]
MODULE = dsp.DiffsolModule(LOGISTIC_CODE)


class DiffsolLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rates: torch.Tensor) -> torch.Tensor:
        outputs = []
        for rate in rates.detach().cpu().tolist():
            _, _, flat = MODULE.solve_dense([rate], TIMES_LIST)
            outputs.append(flat[-1])
        u_final = torch.tensor(outputs, dtype=rates.dtype, device=rates.device)
        probabilities = 1.0 - u_final
        ctx.save_for_backward(rates.detach())
        return probabilities.clamp(1e-6, 1 - 1e-6)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (rates,) = ctx.saved_tensors
        grads = []
        for rate, upstream in zip(rates.tolist(), grad_output.detach().cpu().tolist()):
            g = dsp.reverse_mode(LOGISTIC_CODE, [rate], TIMES_LIST, GRAD_OUT_FINAL)[0]
            grads.append(-g * upstream)
        grad_tensor = torch.tensor(
            grads, dtype=grad_output.dtype, device=grad_output.device
        )
        return grad_tensor


class DiffsolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(16 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        rates = torch.nn.functional.softplus(self.fc(z.flatten(1))) + 1e-3
        probs = DiffsolLogistic.apply(rates.squeeze(1))
        return probs


class TorchdiffeqClassifier(DiffsolClassifier):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch_odeint is None:
            raise RuntimeError("torchdiffeq not installed.")
        z = self.encoder(x)
        rates = torch.nn.functional.softplus(self.fc(z.flatten(1))) + 1e-3
        rates = rates.squeeze(1).to(dtype=torch.float64)

        def func(t, u):
            return -rates * u

        sol = torch_odeint(func, torch.ones_like(rates), TIMES.to(x.device))
        return (1.0 - sol[-1]).to(dtype=x.dtype)


def load_dataset(
    samples: int, batch_size: int
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if torchvision is None:
        data = torch.randn(samples, 1, 28, 28)
        labels = torch.randint(0, 2, (samples,))
        dataset = torch.utils.data.TensorDataset(data, labels)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        base = torchvision.datasets.FashionMNIST(
            "./data", train=True, download=True, transform=transform
        )
        idx = torch.randperm(len(base))[:samples]
        dataset = torch.utils.data.Subset(base, idx.tolist())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_diffsol(loader, epochs: int, device: torch.device):
    model = DiffsolClassifier().to(device)
    optimiz = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    start = time.perf_counter()
    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            targets = (labels % 2 == 0).float().to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimiz.zero_grad()
            loss.backward()
            optimiz.step()
    elapsed = time.perf_counter() - start
    acc = evaluate(model, loader, device)
    return elapsed, acc


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            targets = (labels % 2 == 0).to(device)
            preds = model(images)
            predicted = (preds > 0.5).long()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
    model.train()
    return correct / max(1, total)


def torchdiffeq_forward(rate: float, device: torch.device) -> float:
    if torch_odeint is None:
        raise RuntimeError("torchdiffeq not installed.")

    def func(t, x):
        return -rate * x

    sol = torch_odeint(
        func, torch.ones(1, dtype=torch.float64, device=device), TIMES.to(device)
    )
    return float(1.0 - sol[-1].cpu())


def train_torchdiffeq(loader, epochs: int, device: torch.device):
    if torch_odeint is None:
        return math.nan, math.nan
    model = TorchdiffeqClassifier().to(device)
    optimiz = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    start = time.perf_counter()
    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            targets = (labels % 2 == 0).float().to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimiz.zero_grad()
            loss.backward()
            optimiz.step()
    elapsed = time.perf_counter() - start
    acc = evaluate(model, loader, device)
    return elapsed, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    loader = load_dataset(args.samples, args.batch_size)
    diffsol_time, diffsol_acc = train_diffsol(loader, args.epochs, device)
    print(f"diffsol: time={diffsol_time:.2f}s, accuracy={diffsol_acc:.3f}")
    if torch_odeint is not None:
        torch_time, torch_acc = train_torchdiffeq(loader, args.epochs, device)
        if not math.isnan(torch_time):
            print(f"torchdiffeq: time={torch_time:.2f}s, accuracy={torch_acc:.3f}")
    else:
        print("torchdiffeq not available; skipping baseline.")


if __name__ == "__main__":
    main()
