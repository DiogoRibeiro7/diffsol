import pytest
import torch

from diffsol_pytorch import device as device_utils

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@pytest.mark.memory
def test_repeated_solves_hold_memory(logistic_module, time_grid):
    if psutil is None:
        pytest.skip("psutil not available")
    process = psutil.Process()
    baseline = process.memory_info().rss
    params = torch.tensor([0.7], dtype=torch.float64)
    for _ in range(64):
        device_utils.solve_dense_tensor(logistic_module, params, time_grid)
    after = process.memory_info().rss
    assert after - baseline < 20 * 1024 * 1024  # 20 MB budget
