import numpy as np
import pytest
import torch

from diffsol_pytorch import DiffsolModule, device as device_utils

LOGISTIC_CODE = """
in = [k]
k { 1.0 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"""


@pytest.mark.perf
def test_against_torchdiffeq(run_perf, benchmark):
    if not run_perf:
        pytest.skip("performance benchmarks disabled (use --run-perf)")
    torchdiffeq = pytest.importorskip("torchdiffeq")
    module = DiffsolModule(LOGISTIC_CODE)

    params = torch.tensor([1.0], dtype=torch.float64)
    times = torch.linspace(0.0, 4.0, 256, dtype=torch.float64)

    def diffsol_run():
        device_utils.solve_dense_tensor(module, params, times)

    diffsol_time = benchmark(diffsol_run)

    def torchdiffeq_run():
        torchdiffeq.odeint(lambda t, y: -params[0] * y, torch.ones_like(params), times)

    ode_time = benchmark(torchdiffeq_run)

    assert diffsol_time <= ode_time * 2.0
