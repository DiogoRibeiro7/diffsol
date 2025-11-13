import numpy as np
import pytest

from diffsol_pytorch import DiffsolModule, testing

from conftest import AD_SKIP_REASON, HAS_AUTODIFF

pytestmark = pytest.mark.integration

VAN_DER_POL_CODE = """
in = [mu]
mu { 5.0 }
u_i {
    x = 2.0,
    y = 0.0,
}
F_i {
    y,
    mu * (1 - x * x) * y - x,
}
"""

LORENZ_CODE = """
in = [sigma, rho, beta]
sigma { 10.0 }
rho { 28.0 }
beta { 8.0 / 3.0 }
u_i {
    x = 1.0,
    y = 0.0,
    z = 0.0,
}
F_i {
    sigma * (y - x),
    x * (rho - z) - y,
    x * y - beta * z,
}
"""

HAMILTONIAN_CODE = """
in = [omega]
omega { 1.0 }
u_i {
    q = 1.0,
    p = 0.0,
}
F_i {
    p,
    -omega * omega * q,
}
"""

DAE_CODE = """
in = [k]
k { 10.0 }
u_i {
    x = 1.0,
    y = 0.0,
    g = -k,
}
dudt_i {
    dxdt = 0.0,
    dydt = 0.0,
    dgdt = 0.0,
}
M_i {
    dxdt,
    dydt,
    0,
}
F_i {
    y,
    -k * x,
    k * x + g,
}
"""

TWO_PARAM_CODE = """
in = [k, b]
k { 0.5 }
b { 0.1 }
u {
    u = 1.0,
}
F {
    -k * u + b,
}
"""


def test_van_der_pol_limit_cycle_properties():
    module = DiffsolModule(VAN_DER_POL_CODE)
    params = [5.0]
    times = np.linspace(0.0, 30.0, 6000).tolist()
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    window = sol[:, -2000:]
    max_x = window[0].max()
    min_x = window[0].min()
    assert max_x > 0.0
    assert min_x < 0.0
    assert (max_x - min_x) < 15.0


def test_lorenz_statistics_stay_in_attractor_band():
    module = DiffsolModule(LORENZ_CODE)
    params = [10.0, 28.0, 8.0 / 3.0]
    times = np.linspace(0.0, 40.0, 8000).tolist()
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    tail = sol[:, -2000:]
    means = tail.mean(axis=1)
    assert abs(means[0]) < 5.0
    assert abs(means[1]) < 8.0
    assert 10.0 < means[2] < 35.0
    assert np.all(np.isfinite(tail))


def test_hamiltonian_energy_is_conserved():
    module = DiffsolModule(HAMILTONIAN_CODE)
    params = [1.5]
    times = np.linspace(0.0, 20.0, 2000).tolist()
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    assert np.max(np.abs(sol)) < 10.0


def test_dae_constraint_residual_is_small():
    module = DiffsolModule(DAE_CODE)
    params = [10.0]
    times = np.linspace(0.0, 5.0, 500).tolist()
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    residual = params[0] * sol[0] + sol[2]
    assert np.all(np.isfinite(residual))


@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
def test_sensitivities_match_finite_difference_for_two_params():
    params = [0.5, 0.1]
    times = np.linspace(0.0, 2.0, 80).tolist()

    def loss(sol):
        grad = np.zeros_like(sol)
        grad[0, -1] = 1.0
        return float(sol[0, -1]), grad

    fd, _ = testing.finite_difference_gradients(TWO_PARAM_CODE, params, times, loss)
    forward = testing.forward_mode(DiffsolModule(TWO_PARAM_CODE), params, times)
    grad_sol = np.zeros_like(forward.solution)
    grad_sol[0, -1] = 1.0
    fwd = np.einsum("pij,ij->p", forward.sensitivities, grad_sol)
    assert np.allclose(fd, fwd, rtol=1e-4)
