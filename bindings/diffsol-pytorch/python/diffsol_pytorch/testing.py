"""Utilities for validating diffsol gradients against references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from . import DiffsolModule as _NativeModule
from . import reverse_mode as _reverse_mode

LossFn = Callable[[np.ndarray], Tuple[float, np.ndarray]]


def _ensure_module(code_or_module) -> _NativeModule:
    if isinstance(code_or_module, _NativeModule):
        return code_or_module
    return _NativeModule(str(code_or_module))


def _solve(
    module: _NativeModule, params: Sequence[float], times: Sequence[float]
) -> np.ndarray:
    nout, nt, flat = module.solve_dense(list(params), list(times))
    return np.array(flat, dtype=float).reshape(nout, nt)


@dataclass
class ForwardSensitivities:
    solution: np.ndarray
    sensitivities: np.ndarray  # shape: (nparams, nout, nt)


def forward_mode(module: _NativeModule, params, times) -> ForwardSensitivities:
    nout, nt, sol_flat, nsens, _, _, sens_flat = module.forward_mode(params, times)
    solution = np.array(sol_flat, dtype=float).reshape(nout, nt)
    if nsens == 0:
        sens = np.zeros((0, nout, nt), dtype=float)
    else:
        sens = np.array(sens_flat, dtype=float).reshape(nsens, nout, nt)
    return ForwardSensitivities(solution=solution, sensitivities=sens)


def finite_difference_gradients(
    code_or_module,
    params: Sequence[float],
    times: Sequence[float],
    loss_fn: LossFn,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    module = _ensure_module(code_or_module)
    base = _solve(module, params, times)
    loss0, grad0 = loss_fn(base)
    if grad0.shape != base.shape:
        raise ValueError("loss_fn gradient must match solution shape")
    params_np = np.array(params, dtype=float)
    gradients = np.zeros_like(params_np)
    for i in range(len(params_np)):
        params_np[i] += eps
        loss_plus, _ = loss_fn(_solve(module, params_np, times))
        params_np[i] -= 2 * eps
        loss_minus, _ = loss_fn(_solve(module, params_np, times))
        gradients[i] = (loss_plus - loss_minus) / (2 * eps)
        params_np[i] += eps  # restore
    return gradients, loss0


def reverse_mode_gradients(
    code: str,
    params: Sequence[float],
    times: Sequence[float],
    grad_output: np.ndarray,
) -> np.ndarray:
    grads = _reverse_mode(
        code, list(params), list(times), grad_output.reshape(-1).tolist()
    )
    return np.array(grads[: len(params)], dtype=float)


def check_gradients(
    code: str,
    params: Sequence[float],
    times: Sequence[float],
    loss_fn: LossFn,
) -> dict:
    module = _ensure_module(code)
    sol = _solve(module, params, times)
    _, grad_sol = loss_fn(sol)
    grad_out = grad_sol
    fd, _ = finite_difference_gradients(module, params, times, loss_fn)
    rev = reverse_mode_gradients(code, params, times, grad_out)
    fwd = forward_mode(module, params, times)
    sens = fwd.sensitivities
    if sens.shape[0] != len(params):
        forward = np.zeros_like(fd)
    else:
        forward = np.einsum("pij,ij->p", sens, grad_sol)
    return {
        "finite_difference": fd,
        "reverse_mode": rev,
        "forward_mode": forward,
        "solution": sol,
    }


__all__ = [
    "ForwardSensitivities",
    "finite_difference_gradients",
    "reverse_mode_gradients",
    "forward_mode",
    "check_gradients",
]
