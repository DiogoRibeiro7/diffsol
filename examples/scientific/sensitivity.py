import numpy as np

import diffsol_pytorch as dsp

CODE = """
state x
param k
der(x) = -k * x
"""


def run():
    module = dsp.DiffsolModule(CODE)
    times = np.linspace(0.0, 1.0, 50).tolist()
    params = [0.4]
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    grad_out = [0.0] * (nt - 1) + [1.0]
    grads = dsp.reverse_mode(CODE, params, times, grad_out)
    print(f"Sensitivity d(x(T))/d k = {grads[0]:.6f}")
    return sol


if __name__ == "__main__":
    run()
