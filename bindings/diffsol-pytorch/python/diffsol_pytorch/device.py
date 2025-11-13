"""
Device management helpers for diffsol-pytorch.

This module centralises Torch device detection, backend capability checks,
and the conversions required to shuttle buffers between CPU-hosted solvers
and accelerated tensors (CUDA/ROCm/Metal).  The solver itself always runs on
CPU – we copy inputs to host memory, execute the Rust kernel, and copy the
results back to the desired device on completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, TYPE_CHECKING, Union, Protocol

import numpy as np
import torch

from . import DiffsolModule as _NativeModule
from . import reverse_mode as _reverse_mode

if TYPE_CHECKING:
    class _ModuleProtocol(Protocol):
        def solve_dense(self, params: Sequence[float], times: Sequence[float]) -> Tuple[int, int, Sequence[float]]:
            ...

    NativeModuleLike = _ModuleProtocol
else:
    NativeModuleLike = Any


@dataclass(frozen=True)
class BackendStatus:
    cuda: bool
    rocm: bool
    mps: bool

    @property
    def any_gpu(self) -> bool:
        return self.cuda or self.rocm or self.mps


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    backend: str

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"


def detect_backends() -> BackendStatus:
    """Return availability flags for CUDA, ROCm, and Metal (MPS) backends."""
    cuda_available = torch.cuda.is_available()
    rocm_available = bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    return BackendStatus(cuda=cuda_available, rocm=rocm_available, mps=mps_available)


def select_device(explicit: Optional[str] = None, *, sample: Optional[torch.Tensor] = None) -> DeviceInfo:
    """
    Choose an execution device based on user preference, sample tensors, and availability.

    Parameters
    ----------
    explicit:
        Optional device string ("cuda", "cuda:1", "mps", "cpu", ...).
    sample:
        Optional tensor whose device should be honoured (useful when gradients live on GPU).
    """
    if sample is not None and sample.is_cuda:
        return DeviceInfo(device=sample.device, backend="cuda")
    if sample is not None and sample.device.type == "mps":
        return DeviceInfo(device=sample.device, backend="mps")

    if explicit is not None:
        dev = torch.device(explicit)
        backend = dev.type
        return DeviceInfo(device=dev, backend=backend)

    status = detect_backends()
    if status.cuda:
        return DeviceInfo(device=torch.device("cuda"), backend="cuda")
    if status.rocm:
        return DeviceInfo(device=torch.device("cuda"), backend="rocm")
    if status.mps:
        return DeviceInfo(device=torch.device("mps"), backend="mps")
    return DeviceInfo(device=torch.device("cpu"), backend="cpu")


def _tensor_to_cpu_array(values: Union[Sequence[float], torch.Tensor]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return (
            values.detach()
            .to("cpu", dtype=torch.float64, non_blocking=True)
            .contiguous()
            .view(-1)
            .numpy()
        )
    return np.asarray(list(values), dtype=np.float64)


def _times_to_list(times: Union[Sequence[float], torch.Tensor]) -> Sequence[float]:
    arr = _tensor_to_cpu_array(times)
    if arr.ndim != 1:
        raise ValueError("time grid must be 1-D")
    return arr.tolist()


def solve_dense_tensor(
    module: NativeModuleLike,
    params: Union[Sequence[float], torch.Tensor],
    times: Union[Sequence[float], torch.Tensor],
    *,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Execute ``DiffsolModule.solve_dense`` with automatic device placement.

    Parameters
    ----------
    module:
        Compiled ``DiffsolModule`` instance.
    params, times:
        PyTorch tensors or Python sequences convertible to float64.
    device:
        Optional target device (``"cuda"``, ``"mps"``, ``"cpu"``...).  When omitted,
        we attempt to reuse the device of ``params`` (if tensor) or pick the first
        available accelerator.
    """
    sample_tensor = params if torch.is_tensor(params) else None
    info = select_device(device, sample=sample_tensor)

    params_vec = _tensor_to_cpu_array(params).tolist()
    times_vec = _times_to_list(times)
    nout, ntimes, flat = module.solve_dense(params_vec, times_vec)
    result = torch.from_numpy(np.array(flat, dtype=np.float64).reshape(nout, ntimes))
    return result.to(info.device, non_blocking=True)


def reverse_mode_tensor(
    code: str,
    params: Union[Sequence[float], torch.Tensor],
    times: Union[Sequence[float], torch.Tensor],
    grad_output: torch.Tensor,
    *,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Run ``reverse_mode`` with device-aware buffer copies and return gradients on ``device``.

    Parameters match :func:`solve_dense_tensor`.
    """
    info = select_device(device, sample=grad_output)
    params_vec = _tensor_to_cpu_array(params).tolist()
    times_vec = _times_to_list(times)
    grad_vec = grad_output.detach().to("cpu", dtype=torch.float64).contiguous().view(-1).tolist()
    grads = _reverse_mode(code, params_vec, times_vec, grad_vec)
    grad_tensor = torch.from_numpy(np.asarray(grads, dtype=np.float64))
    return grad_tensor.to(info.device, non_blocking=True)


def supported_backends_message() -> str:
    status = detect_backends()
    entries = []
    entries.append(f"CUDA: {'yes' if status.cuda else 'no'}")
    entries.append(f"ROCm: {'yes' if status.rocm else 'no'}")
    entries.append(f"MPS: {'yes' if status.mps else 'no'}")
    entries.append("CPU: always available")
    return ", ".join(entries)
