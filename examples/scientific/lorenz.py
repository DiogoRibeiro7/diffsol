import numpy as np

import diffsol_pytorch as dsp

CODE = """
state x
state y
state z
param sigma
param rho
param beta
der(x) = sigma * (y - x)
der(y) = x * (rho - z) - y
der(z) = x * y - beta * z
"""


def run(params=None):
    if params is None:
        params = [10.0, 28.0, 8.0 / 3.0]
    module = dsp.DiffsolModule(CODE)
    times = np.linspace(0.0, 30.0, 6000).tolist()
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    print("Lorenz attractor sample:", sol[:, -5:])
    return sol


if __name__ == "__main__":
    run()
