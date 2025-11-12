import argparse
import math
import time

import torch

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    torch_odeint = None

import diffsol_pytorch as dsp

CODE = """
in = [k]
k { 1.5 }
u {
    x = 1.0,
}
F {
    -k * x,
}
"""


def analytic(k: float, times: torch.Tensor) -> torch.Tensor:
    return torch.exp(-k * times)


def validate(device: torch.device):
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but not available.")
    module = dsp.DiffsolModule(CODE)
    params = [1.5]
    times = torch.linspace(0.0, 5.0, 2000, dtype=torch.float64)
    start = time.perf_counter()
    nout, nt, flat = module.solve_dense(params, times.tolist())
    diffsol_time = time.perf_counter() - start
    sol = torch.tensor(flat, dtype=torch.float64).reshape(nout, nt)
    grad_out = torch.zeros_like(sol)
    grad_out[:, -1] = 1.0
    grads = dsp.reverse_mode(CODE, params, times.tolist(), grad_out.reshape(-1).tolist())
    analytic_grad = -times[-1].item() * math.exp(-params[0] * times[-1].item())
    print(f"diffsol: runtime={diffsol_time:.3f}s, final_grad={grads[0]:+.3e}, analytic={analytic_grad:+.3e}")

    if torch_odeint is None:
        print("torchdiffeq not installed; skipping GPU comparison.")
        return

    gpu_times = times.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    sol_gpu = torch_odeint(lambda t, x: -params[0] * x, torch.ones(1, device=device, dtype=torch.float64), gpu_times)
    torch_time = time.perf_counter() - start
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        peak = 0.0
    sol_gpu = sol_gpu.squeeze(-1).cpu()
    diff = torch.max(torch.abs(sol_gpu - sol[0]))
    print(f"torchdiffeq ({device.type}): runtime={torch_time:.3f}s, max_diff={diff:.2e}, peak_mem={peak:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    validate(torch.device(args.device))


if __name__ == "__main__":
    main()
