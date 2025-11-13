import numpy as np
import pytest

from diffsol_pytorch import testing as testing_utils

from conftest import AD_SKIP_REASON, HAS_AUTODIFF

LOGISTIC_CODE = """
in = [k]
k { 0.4 }
u {
    u = 1.0,
}
F {
    -k * u,
}
"""


@pytest.mark.gradient
@pytest.mark.skipif(not HAS_AUTODIFF, reason=AD_SKIP_REASON)
@pytest.mark.parametrize("param", [0.2, 0.7])
def test_gradients_match_finite_difference(param):
    times = np.linspace(0.0, 1.25, 30).tolist()

    def loss_fn(sol):
        grad = np.zeros_like(sol)
        grad[0, -1] = 1.0
        return float(sol[0, -1]), grad

    results = testing_utils.check_gradients(LOGISTIC_CODE, [param], times, loss_fn)
    fd = results["finite_difference"]
    rev = results["reverse_mode"]
    fwd = results["forward_mode"]
    assert np.allclose(fd, rev, rtol=1e-4)
    assert np.allclose(fd, fwd, rtol=1e-4)
