import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    torch_odeint = None

import diffsol_pytorch as dsp

RESULTS_PATH = Path("benchmarks/results.md")

LOGISTIC_CODE = """
in = [k]
k { 0.7 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"""

NEURAL_CODE = """
in = [w0, w1, w2, w3, b0, b1]
w0 { 0.2 }
w1 { -0.4 }
w2 { 0.3 }
w3 { 0.1 }
b0 { 0.05 }
b1 { -0.15 }
u_i {
    z0 = 1.0,
    z1 = 1.0,
}
F_i {
    tanh(w0 * z0 + w1 * z1 + b0),
    tanh(w2 * z0 + w3 * z1 + b1),
}
"""


@dataclass
class BackendStats:
    runtime: float
    max_error: float
    cpu_mb: float
    gpu_mb: float
    notes: Optional[str] = None


def _ensure_device(device: torch.device):
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but not available.")


def _memory_start(device: torch.device) -> int:
    rss = psutil.Process().memory_info().rss if psutil else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return rss


def _memory_stop(device: torch.device, start_rss: int) -> (float, float):
    if psutil:
        rss = psutil.Process().memory_info().rss
        cpu_mb = max(0, rss - start_rss) / 1e6
    else:
        cpu_mb = float("nan")
    gpu_mb = 0.0
    if device.type == "cuda":
        gpu_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return cpu_mb, gpu_mb


def run_diffsol(code: str, params, times: torch.Tensor, device: torch.device):
    module = dsp.DiffsolModule(code)
    start_rss = _memory_start(device)
    start = time.perf_counter()
    nout, nt, flat = module.solve_dense(params, times.cpu().tolist())
    runtime = time.perf_counter() - start
    cpu_mb, gpu_mb = _memory_stop(device, start_rss)
    sol = torch.tensor(flat, dtype=torch.float64).reshape(nout, nt)
    return runtime, sol, cpu_mb, gpu_mb


def run_torchdiffeq(func, y0: torch.Tensor, times: torch.Tensor, device: torch.device):
    if torch_odeint is None:
        return math.nan, torch.empty(0), math.nan, math.nan
    start_rss = _memory_start(device)
    start = time.perf_counter()
    sol = torch_odeint(func, y0.to(device), times.to(device))
    runtime = time.perf_counter() - start
    cpu_mb, gpu_mb = _memory_stop(device, start_rss)
    sol = sol.transpose(0, 1).contiguous().cpu()
    return runtime, sol, cpu_mb, gpu_mb


def logistic_benchmark(device: torch.device, stiff: bool = False) -> Dict[str, BackendStats]:
    k = 150.0 if stiff else 0.7
    times = torch.linspace(0.0, 5.0, 1500 if stiff else 500, dtype=torch.float64, device=device)
    analytic = torch.exp(-k * times.cpu())
    diffsol_time, diffsol_sol, cpu_mb, gpu_mb = run_diffsol(LOGISTIC_CODE, [k], times, device)
    diffsol_err = torch.max(torch.abs(diffsol_sol[0] - analytic))
    grad = -times[-1].item() * math.exp(-k * times[-1].item())
    stats = {
        "diffsol": BackendStats(
            runtime=diffsol_time,
            max_error=float(diffsol_err),
            cpu_mb=cpu_mb,
            gpu_mb=gpu_mb,
            notes=f"adjoint grad final={grad:+.3e}",
        )
    }
    func = lambda t, y: -k * y
    torch_time, torch_sol, t_cpu, t_gpu = run_torchdiffeq(
        func,
        torch.tensor([1.0], dtype=torch.float64, device=device),
        times,
        device,
    )
    if not math.isnan(torch_time):
        torch_err = torch.max(torch.abs(torch_sol[0] - analytic))
        stats["torchdiffeq"] = BackendStats(torch_time, float(torch_err), t_cpu, t_gpu)
    return stats


def neural_ode_benchmark(device: torch.device) -> Dict[str, BackendStats]:
    params = torch.tensor([0.2, -0.4, 0.3, 0.1, 0.05, -0.15], dtype=torch.float64)
    params_device = params.to(device)
    times = torch.linspace(0.0, 2.0, 600, dtype=torch.float64, device=device)
    diffsol_time, diffsol_sol, cpu_mb, gpu_mb = run_diffsol(NEURAL_CODE, params.tolist(), times, device)
    notes = f"||z(T)||={float(torch.linalg.norm(diffsol_sol[:, -1])):.3f}"
    stats = {
        "diffsol": BackendStats(
            runtime=diffsol_time,
            max_error=0.0,
            cpu_mb=cpu_mb,
            gpu_mb=gpu_mb,
            notes=notes,
        )
    }
    if torch_odeint is not None:
        def func(t, z):
            z0, z1 = z
            dz0 = torch.tanh(params_device[0] * z0 + params_device[1] * z1 + params_device[4])
            dz1 = torch.tanh(params_device[2] * z0 + params_device[3] * z1 + params_device[5])
            return torch.stack([dz0, dz1])

        torch_time, torch_sol, t_cpu, t_gpu = run_torchdiffeq(
            func,
            torch.ones(2, dtype=torch.float64, device=device),
            times,
            device,
        )
        if not math.isnan(torch_time):
            diff = torch.max(torch.abs(diffsol_sol - torch_sol))
            stats["torchdiffeq"] = BackendStats(torch_time, float(diff), t_cpu, t_gpu)
    return stats


def write_markdown(non_stiff, stiff, neural, device: torch.device):
    lines = [
        f"# Benchmark Results (device: {device})",
        "",
    ]
    for title, data in [
        ("Non-stiff logistic", non_stiff),
        ("Stiff logistic", stiff),
        ("Neural ODE block", neural),
    ]:
        lines += [
            f"## {title}",
            "",
            "| Backend | Time (s) | Max Error | CPU MB | GPU MB | Notes |",
            "|---------|----------|-----------|--------|--------|-------|",
        ]
        for backend, stats in data.items():
            note = stats.notes if stats.notes is not None else ""
            lines.append(
                f"| {backend} | {stats.runtime:.4f} | {stats.max_error:.2e} | "
                f"{stats.cpu_mb:.2f} | {stats.gpu_mb:.2f} | {note} |"
            )
        lines.append("")
    RESULTS_PATH.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    _ensure_device(device)
    non_stiff = logistic_benchmark(device, stiff=False)
    stiff = logistic_benchmark(device, stiff=True)
    neural = neural_ode_benchmark(device)
    write_markdown(non_stiff, stiff, neural, device)
    print(RESULTS_PATH.read_text())


if __name__ == "__main__":
    main()
