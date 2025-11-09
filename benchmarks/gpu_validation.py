import argparse
import time

import torch

import diffsol_pytorch as dsp

CODE = """
state x
param k
der(x) = -k * x
"""


def run(device: torch.device):
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but not available.")
    module = dsp.DiffsolModule(CODE)
    params = [0.5]
    times = torch.linspace(0.0, 2.0, 200, device=torch.device("cpu")).tolist()
    start = time.perf_counter()
    module.solve_dense(params, times)
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device=device)
    else:
        mem = 0
    print(f"{device}: time={elapsed:.4f}s, peak_mem={mem / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(torch.device(args.device))


if __name__ == "__main__":
    main()
