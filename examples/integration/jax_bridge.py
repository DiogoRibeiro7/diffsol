"""
Use diffsol inside JAX computations via host callbacks.

This script integrates a scalar decay ODE using diffsol and compares it
against JAX's native solver. It demonstrates how to keep most of your model
in JAX while outsourcing stiff or feature-rich solves to diffsol.
"""

from __future__ import annotations

import numpy as np

import diffsol_pytorch as dsp

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "JAX is required for this example. pip install jax jaxlib."
    ) from exc


DECAY_CODE = """
in = [k]
k { 0.7 }
u_i {
    x = 1.0,
}
F_i {
    -k * x,
}
"""


class DiffsolDecay:
    def __init__(self):
        self.module = dsp.DiffsolModule(DECAY_CODE)
        self.times = np.linspace(0.0, 1.0, 64).tolist()

    def _solve_numpy(self, k: float) -> np.ndarray:
        nout, nt, flat = self.module.solve_dense([k], self.times)
        sol = np.array(flat, dtype=float).reshape(nout, nt)
        return sol[-1]  # final value

    def __call__(self, k: jnp.ndarray) -> jnp.ndarray:
        def _callback(rate):
            return self._solve_numpy(float(rate))

        result = jax.experimental.host_callback.call(
            _callback,
            k,
            result_shape=jax.ShapeDtypeStruct((), k.dtype),
        )
        return result


def main():
    dsp.enable_logging("warn")
    integrator = DiffsolDecay()

    @jax.jit
    def batch_eval(rates: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(integrator)(rates)

    k_values = jnp.linspace(0.1, 2.0, 5)
    outputs = batch_eval(k_values)
    print("Rates:", np.array(k_values))
    print("Final states via diffsol:", np.array(outputs))


if __name__ == "__main__":
    main()
