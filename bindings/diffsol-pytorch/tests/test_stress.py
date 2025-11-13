import numpy as np
import pytest

from diffsol_pytorch import DiffsolModule, reverse_mode

from conftest import AD_SKIP_REASON, HAS_AUTODIFF

pytestmark = pytest.mark.memory


@pytest.mark.slow
def test_dense_solver_handles_long_runs(logistic_code):
    module = DiffsolModule(logistic_code)
    params = [0.6]
    times = np.linspace(0.0, 6.0, 256).tolist()
    for _ in range(64):
        nout, nt, flat = module.solve_dense(params, times)
        assert nout * nt == len(flat)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_reverse_mode_reuses_checkpoints(logistic_code):
    module = DiffsolModule(logistic_code)
    params = [0.4]
    times = np.linspace(0.0, 4.0, 128).tolist()
    grad = np.zeros((1, len(times)), dtype=float)
    grad[0, -1] = 1.0
    grad_flat = grad.reshape(-1).tolist()
    for _ in range(16):
        grads = reverse_mode(logistic_code, params, times, grad_flat)
        assert len(grads) >= len(params)
