import math

import pytest
import torch

import diffsol_pytorch as dsp

from conftest import AD_SKIP_REASON, HAS_AUTODIFF

pytestmark = pytest.mark.unit

CODE = """
in = [k]
k { 0.4 }
u {
    x = 1.0,
}
F {
    -k * x,
}
"""


def analytic_solution(k: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    return torch.exp(-k * times)


def test_solve_dense_matches_analytic():
    module = dsp.DiffsolModule(CODE)
    params = torch.tensor([0.4], dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    nout, ntimes, flat = module.solve_dense(params.tolist(), times.tolist())
    result = torch.tensor(flat, dtype=torch.float64).reshape(nout, ntimes)
    expected = analytic_solution(params[0], times).unsqueeze(0)
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.gradient
@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_reverse_mode_matches_closed_form_gradient():
    params = torch.tensor([0.7], dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    grad_output = torch.zeros((1, times.numel()), dtype=torch.float64)
    grad_output[0, -1] = 1.0  # sensitivity with respect to final value

    out = dsp.reverse_mode(
        CODE, params.tolist(), times.tolist(), grad_output.flatten().tolist()
    )
    grad_params = torch.tensor(out[:-1], dtype=torch.float64)
    grad_init = torch.tensor(out[-1:], dtype=torch.float64)
    t_final = times[-1]
    expected_state = math.exp(-params.item() * t_final.item())
    expected_grad_param = -t_final.item() * expected_state
    assert torch.allclose(grad_params[0, 0], torch.tensor(expected_grad_param))
    assert torch.allclose(grad_init[0], torch.tensor(expected_state))
