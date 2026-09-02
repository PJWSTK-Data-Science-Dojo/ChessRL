from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import wandb

from luna.game.chess_game import ChessGame
from luna.lc0_batch_stream import iter_lc0_corpus_batches
from luna.lc0_corpus import dataset_fingerprint
from luna.lc0_dataset import Lc0Batch, iter_lc0_batches
from luna.lc0_pretraining_config import (
    LC0_CHECKPOINT_METADATA_KEY,
    LC0_CHECKPOINT_PREFIX,
    Lc0PretrainingConfig,
    Lc0TrainScope,
    lc0_dataset_metadata,
    lc0_resume_contract,
    lc0_resume_seed,
    seed_lc0_pretraining,
    validate_lc0_pretraining_config,
)
from luna.lc0_pretraining_validation import (
    Lc0TrainingMetrics,
    Lc0ValidationMetrics,
    Lc0ValidationPlan,
    evaluate_lc0_validation,
    initialize_lc0_wandb,
    log_lc0_training,
    log_lc0_validation,
    validate_lc0_batch,
)
from luna.network import LunaNetwork
from luna.network_losses import soft_ce_with_support
from luna.pgn_pretraining_checkpoints import (
    CheckpointPublication,
    publish_pretraining_checkpoints,
    resolve_pretraining_resume,
    validate_resume_contract,
)

_HEAD_PREFIXES = ("prediction.policy_head.", "prediction.value_head.")
_BEST_CHECKPOINT_NAME = "best.pth.tar"


@dataclass(frozen=True, slots=True)
class Lc0PretrainingResult:
    global_step: int
    validation: Lc0ValidationMetrics
    latest_checkpoint: Path


@dataclass(frozen=True, slots=True)
class _Context:
    config: Lc0PretrainingConfig
    game: ChessGame
    fingerprint: str
    frozen_digest: str


def run_lc0_pretraining(config: Lc0PretrainingConfig) -> Lc0PretrainingResult:
    validate_lc0_pretraining_config(config)
    seed_lc0_pretraining(config.seed)
    game = ChessGame()
    fingerprint = dataset_fingerprint(config.dataset_path.expanduser().resolve())
    network = LunaNetwork(game, config.learner)
    resume = _restore_network(network, config, fingerprint)
    frozen_digest = _freeze_for_root_supervision(network, config.train_scope)
    context = _Context(config, game, fingerprint, frozen_digest)
    if resume is not None:
        validate_resume_contract(
            resume,
            lc0_resume_contract(config, fingerprint, frozen_digest),
            metadata_key=LC0_CHECKPOINT_METADATA_KEY,
        )
        seed_lc0_pretraining(lc0_resume_seed(config.seed, network.global_step))
    config.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    if resume is None:
        _publish_checkpoints(network, context, None, None)
    metadata = lc0_dataset_metadata(config, fingerprint, frozen_digest)
    wandb_started = initialize_lc0_wandb(network, config, metadata)
    try:
        return _train_chunks(network, context)
    finally:
        if wandb_started:
            wandb.finish()


def _restore_network(
    network: LunaNetwork,
    config: Lc0PretrainingConfig,
    fingerprint: str,
) -> Path | None:
    if config.resume_checkpoint is not None:
        checkpoint = resolve_pretraining_resume(
            config.resume_checkpoint,
            config.output_dir,
            checkpoint_prefix=LC0_CHECKPOINT_PREFIX,
        )
        validate_resume_contract(
            checkpoint,
            lc0_resume_contract(config, fingerprint),
            metadata_key=LC0_CHECKPOINT_METADATA_KEY,
        )
        network.load_checkpoint(str(checkpoint.parent), checkpoint.name, load_optimizer=True)
        if network.global_step > config.total_steps:
            raise ValueError("resume checkpoint is beyond total_steps")
        return checkpoint
    source = config.source_checkpoint
    if source is None:
        raise RuntimeError("Validated source checkpoint is unexpectedly missing")
    resolved = source.expanduser().resolve()
    network.initialize_training_phase(str(resolved.parent), resolved.name)
    return None


def _freeze_for_root_supervision(network: LunaNetwork, scope: Lc0TrainScope = "prediction_heads") -> str:
    trainable_prefixes = _trainable_prefixes(scope)
    trainable_names: list[str] = []
    for name, parameter in network.nnet.named_parameters():
        trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names or not all(
        any(name.startswith(prefix) for name in trainable_names) for prefix in _HEAD_PREFIXES
    ):
        raise RuntimeError("Configured model does not expose both LC0 policy and value heads")
    network.optimizer.zero_grad(set_to_none=True)
    _set_training_mode(network, scope)
    return _frozen_parameter_digest(network, trainable_prefixes)


def _trainable_prefixes(scope: Lc0TrainScope) -> tuple[str, ...]:
    if scope == "representation_and_heads":
        return ("representation.", *_HEAD_PREFIXES)
    return _HEAD_PREFIXES


def _set_training_mode(network: LunaNetwork, scope: Lc0TrainScope) -> None:
    network.nnet.eval()
    if scope == "representation_and_heads":
        network.nnet.representation.train()
    network.nnet.prediction.policy_head.train()
    network.nnet.prediction.value_head.train()


def _frozen_parameter_digest(
    network: LunaNetwork,
    trainable_prefixes: tuple[str, ...] = _HEAD_PREFIXES,
) -> str:
    digest = hashlib.sha256()
    for name, parameter in network.nnet.named_parameters():
        if name.startswith(trainable_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        raw = parameter.detach().cpu().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def _train_chunks(network: LunaNetwork, context: _Context) -> Lc0PretrainingResult:
    config = context.config
    network._resolve_lr_schedule_total(config.total_steps, config.total_steps)
    batches = _training_batches(context, network.global_step)
    validation = _evaluate_validation(network, context)
    best_validation = _load_best_validation(config.output_dir)
    log_lc0_validation(network.global_step, validation)
    best_validation = _publish_checkpoints(network, context, validation, best_validation)
    try:
        while network.global_step < config.total_steps:
            boundary = (network.global_step // config.chunk_steps + 1) * config.chunk_steps
            steps = min(boundary, config.total_steps) - network.global_step
            train_metrics = _train_steps(network, batches, steps, config.total_steps)
            _assert_frozen_parameters(network, context.frozen_digest, config.train_scope)
            log_lc0_training(network.global_step, train_metrics)
            validation = _evaluate_validation(network, context)
            log_lc0_validation(network.global_step, validation)
            best_validation = _publish_checkpoints(network, context, validation, best_validation)
    except KeyboardInterrupt:
        _assert_frozen_parameters(network, context.frozen_digest, config.train_scope)
        _publish_checkpoints(network, context, None, best_validation)
        raise
    return Lc0PretrainingResult(network.global_step, validation, config.output_dir / "latest.pth.tar")


def _training_batches(context: _Context, skip_batches: int) -> Iterator[Lc0Batch]:
    if context.config.dataset_path.expanduser().resolve().is_dir():
        yield from _corpus_training_batches(context, skip_batches)
        return
    yield from _archive_training_batches(context, skip_batches)


def _corpus_training_batches(context: _Context, starting_step: int) -> Iterator[Lc0Batch]:
    config = context.config
    chunk_index, offset = divmod(starting_step, config.chunk_steps)
    window_count = math.ceil(config.total_steps / config.chunk_steps)
    while True:
        epoch = config.dataset.epoch + chunk_index
        dataset = replace(
            config.dataset,
            split="train",
            epoch=epoch,
            batch_size=config.learner.batch_size,
            max_samples=config.chunk_steps * config.learner.batch_size,
        )
        batches = iter_lc0_corpus_batches(
            config.dataset_path,
            dataset,
            context.game,
            archive_offset=epoch,
            member_window_index=epoch % window_count,
            member_window_count=window_count,
        )
        yield from _window_batches(batches, config.chunk_steps, offset, config.learner.batch_size)
        chunk_index += 1
        offset = 0


def _window_batches(
    batches: Iterator[Lc0Batch],
    steps: int,
    offset: int,
    batch_size: int,
) -> Iterator[Lc0Batch]:
    for index in range(steps):
        try:
            batch = next(batches)
        except StopIteration:
            raise ValueError("LC0 training corpus cannot fill a deterministic chunk") from None
        if len(batch.observations) != batch_size:
            raise ValueError("LC0 training corpus produced a partial deterministic batch")
        if index >= offset:
            yield batch


def _archive_training_batches(context: _Context, skip_batches: int) -> Iterator[Lc0Batch]:
    epoch = context.config.dataset.epoch
    remaining_skip = skip_batches
    while True:
        dataset = replace(
            context.config.dataset,
            split="train",
            epoch=epoch,
            batch_size=context.config.learner.batch_size,
        )
        yielded = False
        for batch in iter_lc0_batches(context.config.dataset_path, dataset, context.game):
            yielded = True
            if remaining_skip:
                remaining_skip -= 1
                continue
            yield batch
        if not yielded:
            raise ValueError("LC0 training split contains no accepted samples")
        epoch += 1


def _train_steps(
    network: LunaNetwork,
    batches: Iterator[Lc0Batch],
    steps: int,
    total_steps: int,
) -> Lc0TrainingMetrics:
    totals = np.zeros(4, dtype=np.float64)
    for _ in range(steps):
        metrics = _train_batch(network, next(batches), total_steps)
        totals += (
            np.asarray(
                [metrics.policy_loss, metrics.value_loss, metrics.total_loss, 1.0],
                dtype=np.float64,
            )
            * metrics.positions
        )
    positions = int(totals[3])
    return Lc0TrainingMetrics(totals[0] / positions, totals[1] / positions, totals[2] / positions, positions)


def _train_batch(network: LunaNetwork, batch: Lc0Batch, total_steps: int) -> Lc0TrainingMetrics:
    size = validate_lc0_batch(batch, network)
    learning_rate = network._lr_schedule(network.global_step + 1, total_steps)
    for group in network.optimizer.param_groups:
        group["lr"] = learning_rate
    network.optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(3, device=network.device, dtype=torch.float32)
    for indices in _microbatch_indices(size, network._learner.grad_accum_steps):
        losses = _microbatch_losses(network, batch, indices)
        weight = len(indices) / size
        torch.autograd.backward(network.scaler.scale(losses[2] * weight))
        sums += torch.stack([loss.detach().float() for loss in losses]) * len(indices)
    network.scaler.unscale_(network.optimizer)
    trainable = [parameter for parameter in network.nnet.parameters() if parameter.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, network._learner.grad_clip_norm)
    if not bool(torch.isfinite(gradient_norm)):
        network.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("LC0 pretraining produced non-finite gradients")
    network.scaler.step(network.optimizer)
    network.scaler.update()
    network.optimizer.zero_grad(set_to_none=True)
    network._global_step += 1
    averages = (sums / size).cpu().numpy()
    return Lc0TrainingMetrics(float(averages[0]), float(averages[1]), float(averages[2]), size)


def _microbatch_losses(
    network: LunaNetwork,
    batch: Lc0Batch,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(batch.observations[indices], device=network.device, dtype=torch.float32)
    valid_moves = torch.as_tensor(batch.valid_moves[indices], device=network.device, dtype=torch.float32)
    policies = torch.as_tensor(batch.policies[indices], device=network.device, dtype=torch.float32)
    values = torch.as_tensor(batch.value_targets[indices], device=network.device, dtype=torch.float32)
    with torch.autocast(
        "cuda",
        enabled=network._learner.mixed_precision and network.device.type == "cuda",
        dtype=network._amp_dtype,
    ):
        _latent, log_policy, value_logits = network._training_initial_inference(observations, valid_moves)
        policy_loss = -(policies * log_policy).sum(dim=1).mean()
        value_loss = soft_ce_with_support(value_logits, values).mean()
        total = network._learner.policy_loss_weight * policy_loss
        total = total + network._learner.value_loss_weight * value_loss
    if not bool(torch.isfinite(total)):
        raise RuntimeError("LC0 pretraining produced a non-finite loss")
    return policy_loss, value_loss, total


def _microbatch_indices(size: int, parts: int) -> list[np.ndarray]:
    return [indices for indices in np.array_split(np.arange(size), min(parts, size)) if len(indices)]


def _evaluate_validation(network: LunaNetwork, context: _Context) -> Lc0ValidationMetrics:
    plan = Lc0ValidationPlan(
        context.config.dataset_path,
        context.config.dataset,
        context.game,
        context.config.validation_batch_size,
        context.config.validation_positions,
    )
    return evaluate_lc0_validation(network, plan)


def _assert_frozen_parameters(network: LunaNetwork, expected: str, scope: Lc0TrainScope) -> None:
    if _frozen_parameter_digest(network, _trainable_prefixes(scope)) != expected:
        raise RuntimeError("LC0 root-only pretraining modified a frozen model parameter")


def _publish_checkpoints(
    network: LunaNetwork,
    context: _Context,
    validation: Lc0ValidationMetrics | None,
    best_validation: float | None,
) -> float | None:
    config = context.config
    metadata = lc0_dataset_metadata(config, context.fingerprint, context.frozen_digest)
    objective = _validation_objective(validation, config) if validation is not None else None
    if validation is not None and objective is not None:
        metadata["validation"] = asdict(validation)
        metadata["validation_objective"] = objective
    publication = CheckpointPublication(config.output_dir, config.checkpoint_top_k, metadata)
    publish_pretraining_checkpoints(
        network,
        publication,
        metadata_key=LC0_CHECKPOINT_METADATA_KEY,
        checkpoint_prefix=LC0_CHECKPOINT_PREFIX,
    )
    if objective is None or (best_validation is not None and objective >= best_validation):
        return best_validation
    network.save_checkpoint(
        str(config.output_dir.expanduser().resolve()),
        _BEST_CHECKPOINT_NAME,
        extra_state={LC0_CHECKPOINT_METADATA_KEY: metadata},
    )
    return objective


def _validation_objective(validation: Lc0ValidationMetrics, config: Lc0PretrainingConfig) -> float:
    learner = config.learner
    policy = learner.policy_loss_weight * validation.policy_cross_entropy
    value = learner.value_loss_weight * validation.value_cross_entropy
    return policy + value


def _load_best_validation(output_dir: Path) -> float | None:
    checkpoint = output_dir.expanduser().resolve() / _BEST_CHECKPOINT_NAME
    if not checkpoint.is_file():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get(LC0_CHECKPOINT_METADATA_KEY) if isinstance(payload, dict) else None
    objective = metadata.get("validation_objective") if isinstance(metadata, dict) else None
    if isinstance(objective, bool) or not isinstance(objective, int | float) or not math.isfinite(objective):
        raise RuntimeError(f"LC0 best checkpoint has invalid validation metadata: {checkpoint}")
    return float(objective)
