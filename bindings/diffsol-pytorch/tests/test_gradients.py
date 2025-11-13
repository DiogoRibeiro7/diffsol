import math

import numpy as np
import pytest

from diffsol_pytorch import DiffsolModule, reverse_mode, testing

from conftest import AD_SKIP_REASON, HAS_AUTODIFF

pytestmark = pytest.mark.gradient


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

HARMONIC_CODE = """
in = [omega]
omega { 1.0 }
u {
    q = 1.0,
    p = 0.0,
}
F {
    p,
    -omega * omega * q,
}
"""

SCHRODINGER_CODE = """
in = [V]
V { 1.0 }
u {
    psi_real = 1.0,
    psi_imag = 0.0,
}
F {
    -psi_imag * V,
    psi_real * V,
}
"""

COMPLEX_STEP_CODE = """
in = [kr, ki]
kr { 0.7 }
ki { 1e-8 }
u {
    xr = 1.0,
    xi = 0.0,
}
F {
    -kr * xr + ki * xi,
    -kr * xi - ki * xr,
}
"""


def final_value_loss(component: int = 0):
    def _loss(sol: np.ndarray):
        grad = np.zeros_like(sol)
        grad[component, -1] = 1.0
        return float(sol[component, -1]), grad

    return _loss


def weighted_loss(weights: np.ndarray):
    weights = np.asarray(weights, dtype=float)

    def _loss(sol: np.ndarray):
        grad = np.zeros_like(sol)
        grad[0] = weights
        value = float(np.dot(weights, sol[0]))
        return value, grad

    return _loss


def analytic_sensitivity(k: float, times: np.ndarray) -> np.ndarray:
    t = times.reshape(1, -1)
    sol = np.exp(-k * t)
    return -t * sol


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_reverse_and_forward_match_analytic():
    params = [0.7]
    times = np.linspace(0.0, 2.0, 80)
    results = testing.check_gradients(
        LOGISTIC_CODE,
        params,
        times.tolist(),
        final_value_loss(),
    )
    analytic = analytic_sensitivity(params[0], times)[0, -1]
    assert math.isclose(results["reverse_mode"][0], analytic, rel_tol=1e-5)
    assert math.isclose(results["forward_mode"][0], analytic, rel_tol=1e-5)


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_finite_difference_matches_reverse_mode():
    params = [0.5]
    times = np.linspace(0.0, 1.0, 40)
    loss = weighted_loss(np.linspace(0.0, 1.0, times.size))
    fd, _ = testing.finite_difference_gradients(
        LOGISTIC_CODE, params, times.tolist(), loss
    )
    module = DiffsolModule(LOGISTIC_CODE)
    nout, nt, flat = module.solve_dense(params, times.tolist())
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    grad_out = loss(sol)[1]
    reverse = testing.reverse_mode_gradients(
        LOGISTIC_CODE,
        params,
        times.tolist(),
        grad_out,
    )
    rel_err = np.abs(fd - reverse) / (np.abs(fd) + 1e-12)
    assert rel_err.max() < 1e-4


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_forward_mode_returns_full_sensitivity_matrix():
    params = [0.3]
    times = np.linspace(0.0, 0.5, 5).tolist()
    module = DiffsolModule(LOGISTIC_CODE)
    sens = testing.forward_mode(module, params, times)
    expected = analytic_sensitivity(params[0], np.array(times))
    assert sens.sensitivities.shape == (len(params), sens.solution.shape[0], len(times))
    assert np.allclose(sens.sensitivities[0], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_complex_system_gradients_are_consistent():
    params = [0.8]
    times = np.linspace(0.0, 2.0, 50).tolist()
    results = testing.check_gradients(
        SCHRODINGER_CODE, params, times, final_value_loss(0)
    )
    fd = results["finite_difference"]
    rev = results["reverse_mode"]
    fwd = results["forward_mode"]
    assert np.allclose(fd, rev, rtol=1e-3)
    assert np.allclose(fd, fwd, rtol=1e-3)


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_complex_step_reference_matches_reverse_mode():
    params = [0.7, 1e-8]
    times = np.linspace(0.0, 1.0, 60)
    module = DiffsolModule(COMPLEX_STEP_CODE)
    loss = final_value_loss(1)
    _, grad_sol = loss(testing._solve(module, params, times.tolist()))
    grads = reverse_mode(
        COMPLEX_STEP_CODE, params, times.tolist(), grad_sol.reshape(-1).tolist()
    )
    _, _, flat = module.solve_dense(params, times.tolist())
    sol = np.array(flat, dtype=float).reshape(2, times.size)
    complex_derivative = sol[1, -1] / params[1]
    assert math.isclose(grads[0], complex_derivative, rel_tol=1e-3)


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_time_weighted_loss_exercises_checkpointing():
    params = [0.6]
    times = np.linspace(0.0, 5.0, 120)
    weights = np.exp(-times)
    loss = weighted_loss(weights)
    results = testing.check_gradients(LOGISTIC_CODE, params, times.tolist(), loss)
    diff = np.abs(results["reverse_mode"] - results["finite_difference"])
    assert diff.max() < 1e-4
