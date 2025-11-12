import numpy as np
import pytest

import diffsol_pytorch as dsp

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
def autodiff_available():
    return HAS_AUTODIFF
