from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pytest
import torch

import diffsol_pytorch as dsp
from diffsol_pytorch import device as device_utils
from diffsol_pytorch import testing as testing_utils

_AUTODIFF_CODE = """
in = [k]
k { 0.5 }
u {
    x = 1.0,
}
F {
    -k * x,
}
"""

_LOGISTIC_CODE = """
in = [k]
k { 0.7 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-perf",
        action="store_true",
        help="Enable performance benchmarks (requires torchdiffeq and pytest-benchmark).",
    )


def _check_autodiff() -> bool:
    times = np.linspace(0.0, 1.0, 2).tolist()
    try:
        dsp.reverse_mode(_AUTODIFF_CODE, [0.5], times, [0.0, 1.0])
        return True
    except BaseException as exc:
        if "module does not support sens autograd" in str(exc):
            return False
        raise


HAS_AUTODIFF = _check_autodiff()
AD_SKIP_REASON = "diffsol build lacks LLVM/Enzyme autodiff support"


@pytest.fixture(scope="session")
def autodiff_available() -> bool:
    return HAS_AUTODIFF


@pytest.fixture(scope="session")
def run_perf(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--run-perf"))


@pytest.fixture(scope="session")
def logistic_code() -> str:
    return _LOGISTIC_CODE


@pytest.fixture(scope="session")
def logistic_module(logistic_code: str) -> dsp.DiffsolModule:
    return dsp.DiffsolModule(logistic_code)


@pytest.fixture(scope="session")
def time_grid() -> Sequence[float]:
    return np.linspace(0.0, 1.0, 32).tolist()


@pytest.fixture(scope="session")
def default_device() -> torch.device:
    info = device_utils.select_device()
    return info.device


@pytest.fixture(scope="session")
def gradient_checker() -> Callable[[str, Sequence[float], Sequence[float]], Dict[str, np.ndarray]]:
    def _checker(code: str, params: Sequence[float], times: Sequence[float]) -> Dict[str, np.ndarray]:
        def _loss(sol: np.ndarray) -> tuple[float, np.ndarray]:
            grad = np.zeros_like(sol)
            grad[0, -1] = 1.0
            return float(sol[0, -1]), grad

        return testing_utils.check_gradients(code, params, times, _loss)

    return _checker
