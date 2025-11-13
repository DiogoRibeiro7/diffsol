import math

import numpy as np
import pytest

from diffsol_pytorch import DiffsolModule


@pytest.mark.unit
def test_solve_dense_shape(logistic_module, time_grid):
    nout, nt, data = logistic_module.solve_dense([0.7], time_grid)
    assert nout == 1
    assert nt == len(time_grid)
    assert len(data) == nout * nt


@pytest.mark.unit
def test_solve_dense_matches_closed_form(logistic_module):
    params = [0.3]
    times = np.linspace(0.0, 1.0, 6).tolist()
    _, _, data = logistic_module.solve_dense(params, times)
    solution = np.array(data).reshape(1, len(times))
    analytic = np.exp(-params[0] * np.array(times))
    assert np.allclose(solution[0], analytic)


@pytest.mark.unit
def test_empty_time_grid_raises():
    module = DiffsolModule(
        """
        in = [k]
        k { 0.5 }
        u { x = 1.0, }
        F { -k * x, }
        """
    )
    with pytest.raises((RuntimeError, ValueError)):
        module.solve_dense([0.5], [])


@pytest.mark.unit
def test_enable_logging_idempotent():
    from diffsol_pytorch import enable_logging

    enable_logging("info")
    enable_logging("debug")
