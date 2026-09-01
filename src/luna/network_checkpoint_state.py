"""Validation and CPU cloning for serialized optimizer and scaler state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import torch

GRAD_SCALER_FIELDS = frozenset({"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"})


def clone_state_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_to_cpu(item) for item in value)
    return deepcopy(value)


def first_non_finite_path(value: object, path: str) -> str | None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            return path
        return None
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            return path
        return None
    if isinstance(value, Mapping):
        return _first_invalid_mapping_value(value, path)
    if isinstance(value, list | tuple):
        return _first_invalid_sequence_value(value, path)
    if isinstance(value, float | np.floating) and not math.isfinite(float(value)):
        return path
    return None


def _first_invalid_mapping_value(value: Mapping[object, object], path: str) -> str | None:
    for key, item in value.items():
        invalid = first_non_finite_path(item, f"{path}.{key}")
        if invalid is not None:
            return invalid
    return None


def _first_invalid_sequence_value(value: list[object] | tuple[object, ...], path: str) -> str | None:
    for index, item in enumerate(value):
        invalid = first_non_finite_path(item, f"{path}[{index}]")
        if invalid is not None:
            return invalid
    return None


def validate_finite_state(value: object, label: str) -> None:
    invalid = first_non_finite_path(value, label)
    if invalid is not None:
        raise ValueError(f"Checkpoint contains a non-finite value at {invalid}")


def validate_grad_scaler_state(state: Mapping[str, object]) -> None:
    if not state:
        return
    missing = sorted(name for name in GRAD_SCALER_FIELDS if name not in state)
    unexpected = sorted(str(name) for name in state if name not in GRAD_SCALER_FIELDS)
    if missing or unexpected:
        raise ValueError(f"Checkpoint scaler fields are invalid (missing={missing}, unexpected={unexpected})")
    scale = _scaler_float32(state, "scale")
    if scale < torch.finfo(torch.float32).tiny:
        raise ValueError("Checkpoint scaler scale must be a positive normal float32")
    if _scaler_float32(state, "growth_factor") <= 1:
        raise ValueError("Checkpoint scaler growth_factor must be greater than 1")
    backoff_factor = _scaler_float32(state, "backoff_factor")
    if not 0 < backoff_factor < 1:
        raise ValueError("Checkpoint scaler backoff_factor must be between 0 and 1")
    if float(np.float32(scale * backoff_factor)) >= scale:
        raise ValueError("Checkpoint scaler backoff_factor must reduce the float32 scale")
    growth_interval = _scaler_integer(state, "growth_interval")
    if growth_interval <= 0:
        raise ValueError("Checkpoint scaler growth_interval must be positive")
    if growth_interval > torch.iinfo(torch.int32).max:
        raise ValueError("Checkpoint scaler growth_interval must fit int32")
    growth_tracker = _scaler_integer(state, "_growth_tracker")
    if growth_tracker < 0:
        raise ValueError("Checkpoint scaler _growth_tracker must be non-negative")
    if growth_tracker >= growth_interval:
        raise ValueError("Checkpoint scaler _growth_tracker must be less than growth_interval")


def _scaler_float32(state: Mapping[str, object], name: str) -> float:
    value = _scaler_number(state, name)
    if abs(value) > torch.finfo(torch.float32).max:
        raise ValueError(f"Checkpoint scaler field {name} must be representable as float32")
    return float(np.float32(value))


def _scaler_number(state: Mapping[str, object], name: str) -> float:
    value = state[name]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Checkpoint scaler field {name} must be finite and numeric")
    return float(value)


def _scaler_integer(state: Mapping[str, object], name: str) -> int:
    value = state[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Checkpoint scaler field {name} must be an integer")
    return value
