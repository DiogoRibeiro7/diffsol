import math
from typing import Callable, Tuple

import numpy as np
import pytest

import diffsol_pytorch as dsp


LOGISTIC_CODE = """
in = [k]
k { 1.0 }
u { 1.0 }
F { -k * u }
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
kr { 1.0 }
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


def run_module(code: str, params, times) -> Tuple[int, int, np.ndarray]:
    module = dsp.DiffsolModule(code)
    nout, nt, flat = module.solve_dense(params, times)
    return nout, nt, np.array(flat, dtype=float).reshape(nout, nt)


def finite_difference(
    code: str,
    params,
    times,
    observable: Callable[[np.ndarray], float],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    grads = np.zeros_like(params, dtype=float)
    base = observable(run_module(code, params, times)[2])
    for i in range(len(params)):
        perturbed = np.array(params, dtype=float)
        perturbed[i] += eps
        plus = observable(run_module(code, perturbed.tolist(), times)[2])
        perturbed[i] -= 2 * eps
        minus = observable(run_module(code, perturbed.tolist(), times)[2])
        grads[i] = (plus - minus) / (2 * eps)
    return grads, base


def reverse_mode_grad(
    code: str,
    params,
    times,
    grad_out: np.ndarray,
) -> np.ndarray:
    grads = dsp.reverse_mode(code, params, times, grad_out.flatten().tolist())
    return np.array(grads[: len(params)], dtype=float)


def build_grad_out(observable: Callable[[np.ndarray], float], sol: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(sol)
    idx = np.unravel_index(np.argmax(sol), sol.shape)
    grad[idx] = 1.0
    return grad


def relative_error(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.abs(expected - actual) / (np.abs(expected) + 1e-12)


def analytic_logistic_grad(k: float, t: float) -> float:
    return -t * math.exp(-k * t)


def test_logistic_reverse_mode_matches_analytic():
    params = [0.8]
    times = np.linspace(0.0, 2.0, 25).tolist()
    grad_out = np.zeros((1, len(times)))
    grad_out[0, -1] = 1.0
    grads = reverse_mode_grad(LOGISTIC_CODE, params, times, grad_out)
    expected = analytic_logistic_grad(params[0], times[-1])
    assert pytest.approx(expected, rel=1e-5) == grads[0]


def test_logistic_reverse_mode_matches_finitediff():
    params = [0.5]
    times = np.linspace(0.0, 1.0, 15).tolist()
    fd, _ = finite_difference(LOGISTIC_CODE, params, times, lambda sol: sol[0, -1])
    grad_out = np.zeros((1, len(times)))
    grad_out[0, -1] = 1.0
    grads = reverse_mode_grad(LOGISTIC_CODE, params, times, grad_out)
    rel_err = relative_error(fd, grads)
    assert rel_err.max() < 1e-3


def test_forward_solution_respects_decay():
    params = [0.2]
    times = np.linspace(0.0, 3.0, 40).tolist()
    _, _, sol = run_module(LOGISTIC_CODE, params, times)
    expected = np.exp(-params[0] * np.array(times))
    assert np.allclose(sol[0], expected, atol=1e-5)


def test_harmonic_gradients():
    omega = [2.0]
    times = np.linspace(0.0, 2 * np.pi, 60).tolist()
    grad_out = np.zeros((2, len(times)))
    grad_out[:, -1] = 1.0
    grads = reverse_mode_grad(HARMONIC_CODE, omega, times, grad_out)
    fd, _ = finite_difference(HARMONIC_CODE, omega, times, lambda sol: sol[0, -1])
    assert relative_error(fd, grads).max() < 1e-3


def test_complex_schrodinger():
    V = [1.0]
    times = np.linspace(0.0, 2.0, 40).tolist()
    grad_out = np.zeros((2, len(times)))
    grad_out[:, -1] = 1.0
    grads = reverse_mode_grad(SCHRODINGER_CODE, V, times, grad_out)
    fd, _ = finite_difference(SCHRODINGER_CODE, V, times, lambda sol: sol[0, -1])
    assert relative_error(fd, grads).max() < 1e-3


def test_complex_step_matches_reverse_mode():
    k = [0.7, 1e-8]
    times = np.linspace(0.0, 2.0, 80).tolist()
    grad_out = np.zeros((2, len(times)))
    grad_out[:, -1] = 1.0
    grads = reverse_mode_grad(COMPLEX_STEP_CODE, k, times, grad_out)
    nout, nt, flat = run_module(COMPLEX_STEP_CODE, k, times)
    sol = flat.reshape(nout, nt)
    complex_derivative = sol[1, -1] / k[1]
    assert pytest.approx(complex_derivative, rel=1e-3) == grads[0]


def test_time_dependent_loss_exercises_checkpointing():
    params = [0.6]
    times = np.linspace(0.0, 5.0, 100).tolist()
    weights = np.linspace(0.0, 1.0, len(times))
    grad_out = weights.reshape(1, -1)
    grads = reverse_mode_grad(LOGISTIC_CODE, params, times, grad_out)
    fd, _ = finite_difference(
        LOGISTIC_CODE,
        params,
        times,
        lambda sol_: np.dot(weights, sol_[0]),
    )
    assert relative_error(fd, grads).max() < 1e-4
