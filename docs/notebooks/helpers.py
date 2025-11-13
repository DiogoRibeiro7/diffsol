"""
Shared helpers for the diffsol tutorial notebooks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

try:
    import torchvision
    from torchvision import transforms
except Exception:  # pragma: no cover - optional dependency
    torchvision = None

NOTEBOOK_ROOT = Path(__file__).resolve().parent
DATA_DIR = NOTEBOOK_ROOT / "_data"
CACHE_DIR = NOTEBOOK_ROOT / "_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def dataset_path(name: str) -> Path:
    """
    Return a stable path for downloading small helper datasets.
    """
    return DATA_DIR / name


def _cache_file(name: str) -> Path:
    return CACHE_DIR / name


def save_cached_json(name: str, payload: dict[str, Any]) -> Path:
    """
    Persist a JSON payload under docs/notebooks/_cache.
    """
    path = _cache_file(name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_cached_json(name: str) -> Optional[dict[str, Any]]:
    """
    Load a cached JSON payload if it exists.
    """
    if not name:
        return None
    path = _cache_file(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def accelerator_available() -> bool:
    """
    Return True if CUDA or MPS is available.
    """
    if torch.cuda.is_available():
        return True
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def gpu_section_mode(
    section_name: str, cache_key: Optional[str] = None
) -> Tuple[str, Optional[dict[str, Any]]]:
    """
    Decide how to handle a GPU-only section.

    Returns a tuple of (mode, cached_payload) where mode is one of
    ``run`` (accelerator present), ``cache`` (no accelerator but cached data
    exists), or ``skip`` (no accelerator/cached data). When the ``NB_FORCE_GPU``
    env var is truthy we raise instead of skipping.
    """
    cached_payload = load_cached_json(cache_key) if cache_key else None
    force_gpu = os.getenv("NB_FORCE_GPU", "0").lower() in {"1", "true", "yes"}
    if accelerator_available():
        return "run", cached_payload
    if cached_payload is not None:
        print(f"[cached] {section_name}: accelerator unavailable; reusing {cache_key}.")
        return "cache", cached_payload
    if force_gpu:
        raise RuntimeError(f"{section_name} requires CUDA/MPS but no accelerator is available.")
    print(f"[skip] {section_name}: accelerator unavailable; skipping GPU-only logic.")
    return "skip", None


def load_fashion_mnist_subset(
    samples: int = 128, batch_size: int = 32
) -> torch.utils.data.DataLoader:
    """
    Return a small Fashion-MNIST subset (or random noise if torchvision is unavailable).
    """
    if torchvision is None:
        data = torch.randn(samples, 1, 28, 28)
        labels = torch.randint(0, 2, (samples,))
        dataset = torch.utils.data.TensorDataset(data, labels)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        base = torchvision.datasets.FashionMNIST(
            dataset_path("fashion-mnist"),
            train=True,
            download=True,
            transform=transform,
        )
        idx = torch.randperm(len(base))[:samples]
        dataset = torch.utils.data.Subset(base, idx.tolist())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_decay_data(
    k_true: float = 0.4,
    noise: float = 0.02,
    size: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic exponential-decay measurements with additive Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 2.0, size)
    clean = np.exp(-k_true * times)
    noisy = clean + noise * rng.standard_normal(times.shape)
    return times, noisy


def preferred_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return a torch.device, preferring CUDA/MPS when available.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """
    Human-readable description of the selected device.
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        return f"CUDA ({name})"
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"
