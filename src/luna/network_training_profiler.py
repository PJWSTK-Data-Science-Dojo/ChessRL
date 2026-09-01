"""Optional PyTorch profiler lifecycle for learner iterations."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from luna.network_types import NetworkRuntime


@dataclass(frozen=True, slots=True)
class TrainingProfilerConfig:
    active_steps: int
    output_directory: str | None
    trainer_iteration: int
    export_chrome: bool
    tensorboard_directory: str | None
    with_stack: bool


def start_profiler(network: NetworkRuntime, config: TrainingProfilerConfig) -> profile | None:
    chrome_enabled = bool(config.export_chrome and config.output_directory)
    tensorboard_enabled = bool(config.tensorboard_directory)
    if config.active_steps <= 0 or not (chrome_enabled or tensorboard_enabled):
        return None
    trace_path = _trace_path(config) if chrome_enabled else None
    tensorboard_callback = _tensorboard_callback(config) if tensorboard_enabled else None
    callback = _trace_callback(trace_path, config.tensorboard_directory, tensorboard_callback)
    activities = [ProfilerActivity.CPU]
    if network.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profiler = profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=config.active_steps, repeat=1),
        on_trace_ready=callback,
        record_shapes=True,
        profile_memory=True,
        with_stack=config.with_stack,
    )
    profiler.start()
    return profiler


def _trace_path(config: TrainingProfilerConfig) -> str:
    if config.output_directory is None:
        raise ValueError("A profile directory is required for Chrome trace export")
    os.makedirs(config.output_directory, exist_ok=True)
    return os.path.join(config.output_directory, f"train_trace_iter{config.trainer_iteration}.json")


def _tensorboard_callback(config: TrainingProfilerConfig) -> Callable[[profile], None]:
    if config.tensorboard_directory is None:
        raise ValueError("A TensorBoard directory is required for profiler export")
    os.makedirs(config.tensorboard_directory, exist_ok=True)
    callback = tensorboard_trace_handler(config.tensorboard_directory)
    return cast(Callable[[profile], None], callback)


def _trace_callback(
    trace_path: str | None,
    tensorboard_directory: str | None,
    tensorboard_callback: Callable[[profile], None] | None,
) -> Callable[[profile], None]:
    def on_trace_ready(profiler: profile) -> None:
        if tensorboard_callback is not None:
            # Kineto permits a single trace export per cycle; TensorBoard performs it internally.
            tensorboard_callback(profiler)
            logger.info("TensorBoard / Kineto trace written under {}", tensorboard_directory)
        elif trace_path is not None:
            profiler.export_chrome_trace(trace_path)
            logger.info("PyTorch Chrome trace saved to {}", trace_path)

    return on_trace_ready
