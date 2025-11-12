import numpy as np

import diffsol_pytorch as dsp

CODE = """
in = [mu]
mu { 5.0 }
u {
    x = 2.0,
    y = 0.0,
}
F {
    y,
    mu * (1 - x * x) * y - x,
}
"""


def run(mu: float = 5.0):
    module = dsp.DiffsolModule(CODE)
    times = np.linspace(0.0, 20.0, 2000).tolist()
    params = [mu]
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    energy = 0.5 * (sol[0] ** 2 + sol[1] ** 2)
    print(f"Energy range: {energy.min():.4f} to {energy.max():.4f}")
    return sol


if __name__ == "__main__":
    run()
