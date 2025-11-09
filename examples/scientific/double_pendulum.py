import numpy as np

import diffsol_pytorch as dsp

CODE = """
state theta1
state theta2
state omega1
state omega2
param m1
param m2
param l1
param l2
param g
der(theta1) = omega1
der(theta2) = omega2
der(omega1) = (-g*(2*m1+m2)*sin(theta1) - m2*g*sin(theta1-2*theta2) - 2*sin(theta1-theta2)*m2*(omega2^2*l2+omega1^2*l1*cos(theta1-theta2)))/(l1*(2*m1+m2-m2*cos(2*theta1-2*theta2)))
der(omega2) = (2*sin(theta1-theta2)*(omega1^2*l1*(m1+m2)+g*(m1+m2)*cos(theta1)+omega2^2*l2*m2*cos(theta1-theta2)))/(l2*(2*m1+m2-m2*cos(2*theta1-2*theta2)))
"""


def run():
    params = [1.0, 1.0, 1.0, 1.0, 9.81]
    module = dsp.DiffsolModule(CODE)
    times = np.linspace(0.0, 20.0, 2000).tolist()
    nout, nt, flat = module.solve_dense(params, times)
    sol = np.array(flat, dtype=float).reshape(nout, nt)
    # approximate energy
    m1, m2, l1, l2, g = params
    theta1, theta2, omega1, omega2 = sol
    y1 = -l1 * np.cos(theta1)
    y2 = y1 - l2 * np.cos(theta2)
    v1 = l1 * omega1
    v2 = np.sqrt(v1**2 + (l2 * omega2)**2)
    potential = (m1 + m2) * g * y1 + m2 * g * y2
    kinetic = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    energy = kinetic + potential
    print(f"Double pendulum energy excursion: {(energy.max()-energy.min()):.4f}")
    return sol


if __name__ == "__main__":
    run()
