import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    torch_odeint = None

try:
    import torchvision
    from torchvision import transforms
except ImportError:
    torchvision = None

import diffsol_pytorch as dsp


DECAY_CODE = """
state x
param k
der(x) = -k * x
"""

VDP_CODE = """
state x
state y
param mu
der(x) = y
der(y) = mu * (1 - x^2) * y - x
"""

RESULTS_PATH = Path("benchmarks/results.md")


def run_diffsol(code: str, params: List[float], times: torch.Tensor) -> Tuple[float, torch.Tensor]:
    module = dsp.DiffsolModule(code)
    start = time.perf_counter()
    nout, nt, flat = module.solve_dense(params, times.tolist())
    elapsed = time.perf_counter() - start
    sol = torch.tensor(flat, dtype=torch.float64).reshape(nout, nt)
    return elapsed, sol


def run_torchdiffeq(func, y0: torch.Tensor, times: torch.Tensor) -> Tuple[float, torch.Tensor]:
    if torch_odeint is None:
        return float("nan"), torch.empty(0)
    start = time.perf_counter()
    sol = torch_odeint(func, y0, times)
    elapsed = time.perf_counter() - start
    return elapsed, sol.T


def logistic_benchmark(device: torch.device) -> Dict[str, float]:
    params = torch.tensor([0.7], dtype=torch.float64)
    times = torch.linspace(0.0, 5.0, 500, dtype=torch.float64, device=device)
    diffsol_time, diffsol_sol = run_diffsol(DECAY_CODE, params.tolist(), times.cpu())
    reference_time = float("nan")
    if torch_odeint is not None:
        func = lambda t, x: -params[0] * x
        reference_time, torch_sol = run_torchdiffeq(func, torch.tensor([1.0], dtype=torch.float64), times)
        torch_sol = torch_sol.cpu()
    else:
        torch_sol = torch.exp(-params[0] * times.cpu()).unsqueeze(0)
    diffsol_grad = dsp.reverse_mode(
        DECAY_CODE,
        params.tolist(),
        times.tolist(),
        [0.0] * (len(times) - 1) + [1.0],
    )[0]
    analytic_grad = (-times[-1] * torch.exp(-params[0] * times[-1])).item()
    return {
        "diffsol_time": diffsol_time,
        "torch_time": reference_time,
        "grad_diffsol": diffsol_grad,
        "grad_analytic": analytic_grad,
        "max_error": float(torch.max(torch.abs(diffsol_sol[0] - torch_sol[0].cpu()))),
        "steps": len(times),
        "rtol": 1e-6,
    }


def van_der_pol_benchmark(mu: float, device: torch.device) -> Dict[str, float]:
    params = [mu]
    times = torch.linspace(0.0, 20.0, 2000, dtype=torch.float64, device=device)
    diffsol_time, diffsol_sol = run_diffsol(VDP_CODE, params, times.cpu())
    ref_time = float("nan")
    if torch_odeint is not None:
        def func(t, state):
            x, y = state
            return torch.stack([y, mu * (1 - x ** 2) * y - x])

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)
        ref_time, torch_sol = run_torchdiffeq(func, y0, times)
        torch_sol = torch_sol.cpu()
    else:
        torch_sol = diffsol_sol
    energy = 0.5 * (diffsol_sol[0] ** 2 + diffsol_sol[1] ** 2)
    return {
        "diffsol_time": diffsol_time,
        "torch_time": ref_time,
        "energy_span": float(energy.max() - energy.min()),
        "solution_norm": float(torch.linalg.vector_norm(diffsol_sol)),
        "steps": len(times),
        "rtol": 1e-6,
    }


def neural_ode_mnist(device: torch.device, samples: int = 256) -> Dict[str, float]:
    if torchvision is None:
        return {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=samples, shuffle=False)
    images, labels = next(iter(loader))
    images = images.to(device, dtype=torch.float64).view(images.size(0), -1)
    labels = labels.to(device)
    classifier = torch.nn.Linear(images.size(1), 1, dtype=torch.float64).to(device)
    logits = classifier(images).squeeze(1)
    params = logits.detach().tolist()
    times = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
    start = time.perf_counter()
    preds = []
    for k in params:
        module = dsp.DiffsolModule(DECAY_CODE)
        _, _, flat = module.solve_dense([k], times.tolist())
        preds.append(flat[-1])
    diffsol_time = time.perf_counter() - start
    diffsol_preds = torch.tensor(preds, dtype=torch.float64, device=device)
    diffsol_acc = ((diffsol_preds > 0).long() == labels).float().mean().item()
    torch_time = float("nan")
    torch_acc = float("nan")
    if torch_odeint is not None:
        start = time.perf_counter()
        z0 = torch.ones_like(logits)
        sol = torch_odeint(lambda t, z: -logits * z, z0, times.to(device))
        torch_time = time.perf_counter() - start
        torch_acc = ((sol[-1] > 0).long() == labels).float().mean().item()
    return {
        "diffsol_time": diffsol_time,
        "diffsol_acc": diffsol_acc,
        "torch_time": torch_time,
        "torch_acc": torch_acc,
    }


def write_markdown(logistic_stats, vdp_stats, mnist_stats, device):
    lines = [
        f"# Benchmark Results (device: {device})",
        "",
        "## Logistic Decay",
        "",
        "| Backend | Time (s) | Max Error | Steps |",
        "|---------|----------|-----------|-------|",
        f"| diffsol | {logistic_stats['diffsol_time']:.4f} | {logistic_stats['max_error']:.2e} | {logistic_stats['steps']} |",
    ]
    if not math.isnan(logistic_stats["torch_time"]):
        lines.append(f"| torchdiffeq | {logistic_stats['torch_time']:.4f} | - | {logistic_stats['steps']} |")
    lines += [
        "",
        "## Van der Pol",
        "",
        "| Backend | Time (s) | Energy Span | Steps |",
        "|---------|----------|-------------|-------|",
        f"| diffsol | {vdp_stats['diffsol_time']:.4f} | {vdp_stats['energy_span']:.3f} | {vdp_stats['steps']} |",
    ]
    if not math.isnan(vdp_stats["torch_time"]):
        lines.append(f"| torchdiffeq | {vdp_stats['torch_time']:.4f} | - | {vdp_stats['steps']} |")
    if mnist_stats:
        lines += [
            "",
            "## Neural ODE (MNIST subset)",
            "",
            "| Backend | Time (s) | Accuracy |",
            "|---------|----------|----------|",
            f"| diffsol | {mnist_stats['diffsol_time']:.2f} | {mnist_stats['diffsol_acc']:.3f} |",
        ]
        if not math.isnan(mnist_stats["torch_time"]):
            lines.append(f"| torchdiffeq | {mnist_stats['torch_time']:.2f} | {mnist_stats['torch_acc']:.3f} |")
    RESULTS_PATH.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mnist", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    logistic_stats = logistic_benchmark(device)
    vdp_stats = van_der_pol_benchmark(5.0, device)
    mnist_stats = neural_ode_mnist(device) if args.mnist else {}
    write_markdown(logistic_stats, vdp_stats, mnist_stats, device)
    print(RESULTS_PATH.read_text())


if __name__ == "__main__":
    main()
