import math

import pytest
import torch

from diffsol_pytorch import DiffsolModule, device

from conftest import AD_SKIP_REASON, HAS_AUTODIFF

pytestmark = pytest.mark.unit

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


def _expected_solution(module, params, times):
    nout, nt, flat = module.solve_dense(params.tolist(), times.tolist())
    return torch.tensor(flat, dtype=torch.float64).reshape(nout, nt)


def test_select_device_defaults_to_cpu():
    info = device.select_device()
    assert info.device.type in {"cpu", "cuda", "mps"}


def test_solve_dense_tensor_matches_cpu_result():
    module = DiffsolModule(LOGISTIC_CODE)
    params = torch.tensor([0.7], dtype=torch.float64)
    times = torch.linspace(0.0, 2.0, 16, dtype=torch.float64)
    tensor_result = device.solve_dense_tensor(module, params, times)
    expected = _expected_solution(module, params, times)
    assert torch.allclose(tensor_result, expected)


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_reverse_mode_tensor_matches_host_value():
    module = DiffsolModule(LOGISTIC_CODE)
    params = torch.tensor([0.7], dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    grads = torch.zeros((1, times.numel()), dtype=torch.float64)
    grads[0, -1] = 1.0
    tensor_grad = device.reverse_mode_tensor(LOGISTIC_CODE, params, times, grads)
    # Expect derivative of exp(-k t) wrt k at final time.
    t_final = times[-1].item()
    analytic = -t_final * math.exp(-params.item() * t_final)
    assert math.isclose(tensor_grad[0].item(), analytic, rel_tol=1e-5)
