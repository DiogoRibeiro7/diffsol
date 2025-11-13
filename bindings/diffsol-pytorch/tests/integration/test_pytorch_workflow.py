import math

import pytest
import torch

from diffsol_pytorch import device as device_utils

from conftest import AD_SKIP_REASON, HAS_AUTODIFF


@pytest.mark.integration
@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_forward_backward_on_selected_device(
    logistic_module, logistic_code, time_grid, default_device
):
    params = torch.tensor([0.7], dtype=torch.float64, device=default_device)
    times = torch.tensor(time_grid, dtype=torch.float64, device=params.device)
    sol = device_utils.solve_dense_tensor(logistic_module, params, times)
    assert sol.device == params.device
    grad_out = torch.zeros_like(sol)
    grad_out[:, -1] = 1.0
    grads = device_utils.reverse_mode_tensor(logistic_code, params, times, grad_out)
    assert grads.device == params.device
    t_final = time_grid[-1]
    expected = -t_final * math.exp(-params.item() * t_final)
    assert math.isclose(grads[0].item(), expected, rel_tol=1e-5)
