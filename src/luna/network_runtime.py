"""Device selection and low-level tensor helpers for Luna networks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from loguru import logger
from torch._inductor import config as torch_inductor_config


def configure_dynamic_cudagraphs() -> None:
    torch_inductor_config.triton.cudagraph_skip_dynamic_graphs = True


def has_non_finite_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
    )


def pinned_h2d_float32(array: np.ndarray, device: torch.device) -> torch.Tensor:
    if device.type != "cuda" or not array.flags.c_contiguous:
        return torch.as_tensor(array, dtype=torch.float32, device=device)
    tensor = torch.from_numpy(array)
    pinned = torch.empty(array.shape, dtype=torch.float32, pin_memory=True)
    pinned.copy_(tensor)
    return pinned.to(device, non_blocking=True)


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Preserve the forward value while multiplying its backward gradient."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


def get_device(device_type: str = "cuda", cuda_device_index: int | None = None) -> torch.device:
    """Resolve an available compute device or raise with setup guidance."""
    requested = device_type.lower()
    if requested == "cpu":
        logger.info("Using CPU backend")
        return torch.device("cpu")
    if requested == "mps":
        return _get_mps_device()
    if requested == "cuda":
        return _get_cuda_device(cuda_device_index)
    raise ValueError(f"Unknown device type '{requested}'. Valid options are 'cuda', 'mps', and 'cpu'.")


def _get_mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS backend requested but unavailable. Verify the host and PyTorch build, or use --learner.device cpu."
        )
    if not torch.backends.mps.is_built():
        raise RuntimeError(
            "This PyTorch installation has no MPS support. Use a compatible build or --learner.device cpu."
        )
    logger.info("Using MPS backend")
    return torch.device("mps")


def _get_cuda_device(cuda_device_index: int | None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA backend requested but not available. "
            "Verify the driver and PyTorch build, or use --learner.device cpu."
        )
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("CUDA available but no devices found.")
    indices = [cuda_device_index] if cuda_device_index is not None else list(range(device_count))
    for index in indices:
        if index is not None and 0 <= index < device_count and _cuda_device_is_compatible(index):
            logger.info("Using CUDA device {}", index)
            return torch.device(f"cuda:{index}")
    available = ", ".join(str(index) for index in range(device_count))
    if cuda_device_index is not None:
        raise RuntimeError(
            f"CUDA device {cuda_device_index} unavailable or incompatible. "
            f"Detected device indices: {available}. "
            "Try another --learner.cuda-device index or use --learner.device cpu."
        )
    raise RuntimeError(
        f"No compatible CUDA device found among indices {available}. "
        "Use a compatible PyTorch build or --learner.device cpu."
    )


def _cuda_device_is_compatible(index: int) -> bool:
    try:
        with torch.cuda.device(index):
            probe = torch.zeros(1, device=f"cuda:{index}")
            _ = probe + 1
        return True
    except RuntimeError:
        return False
