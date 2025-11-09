import numpy as np

import diffsol_pytorch as dsp

CODE = """
state x
state y
algebraic g
param k
der(x) = y
0 = k * x + g
"""


def run(k: float = 10.0):
    module = dsp.DiffsolModule(CODE)
    times = np.linspace(0.0, 5.0, 200).tolist()
    params = [k]
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    print("DAE solution head:", sol[:, :5])
    return sol


if __name__ == "__main__":
    run()
