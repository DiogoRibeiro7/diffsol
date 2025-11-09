import numpy as np

import diffsol_pytorch as dsp

CODE = """
state q
state p
der(q) = p
der(p) = -q
"""


def run():
    module = dsp.DiffsolModule(CODE)
    times = np.linspace(0.0, 50.0, 5000).tolist()
    nout, nt, flat = module.solve_dense([], times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    energy = 0.5 * (sol[0] ** 2 + sol[1] ** 2)
    print(f"Energy deviation: {np.max(np.abs(energy - energy[0])):.2e}")
    return sol


if __name__ == "__main__":
    run()
