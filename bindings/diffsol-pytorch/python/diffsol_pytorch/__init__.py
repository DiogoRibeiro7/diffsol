"""
Convenience Python wrappers around the ``diffsol_pytorch`` extension module.

The Rust extension is exposed as ``diffsol_pytorch.diffsol_pytorch`` by maturin.
To make ``import diffsol_pytorch as dsp`` ergonomic in tests and examples we
mirror the public surface here and re-export the extension symbols.
"""

from __future__ import annotations

from importlib import import_module

_ffi = import_module("diffsol_pytorch.diffsol_pytorch")

# Re-export the low-level classes/functions so existing code keeps working.
DiffsolModule = _ffi.DiffsolModule
reverse_mode = _ffi.reverse_mode
_init_logging = _ffi.init_logging


def enable_logging(level: str = "info") -> None:
    """
    Initialize Rust-side logging once for the current process.

    Parameters
    ----------
    level:
        Logging filter passed directly to env_logger (e.g. "info", "debug").
    """
    _init_logging(level)


from . import device  # noqa: E402
from . import testing  # noqa: E402  (depends on _ffi being imported first)

__all__ = [
    "DiffsolModule",
    "reverse_mode",
    "enable_logging",
    "device",
    "testing",
]
