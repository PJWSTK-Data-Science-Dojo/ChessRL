from __future__ import annotations

from pathlib import Path

import pytest
import torch

from luna.pgn_pretraining_checkpoints import (
    CHECKPOINT_METADATA_KEY,
    CheckpointPublication,
    publish_pretraining_checkpoints,
    resolve_pretraining_resume,
    validate_resume_contract,
)


def _write_checkpoint(path: Path, step: int, metadata: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {"format_version": 2, "global_step": step}
    if metadata is not None:
        payload[CHECKPOINT_METADATA_KEY] = metadata
    torch.save(payload, path)


def test_resume_recovers_newest_immutable_checkpoint(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pth.tar"
    newest = tmp_path / "pretrain_step_00000003.pth.tar"
    _write_checkpoint(latest, 2)
    _write_checkpoint(newest, 3)

    assert resolve_pretraining_resume(latest, tmp_path) == newest


def test_resume_ignores_corrupt_latest_when_numbered_checkpoint_is_healthy(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pth.tar"
    healthy = tmp_path / "pretrain_step_00000003.pth.tar"
    latest.write_bytes(b"interrupted checkpoint write")
    _write_checkpoint(healthy, 3)

    assert resolve_pretraining_resume(latest, tmp_path) == healthy


def test_resume_fails_when_every_checkpoint_is_corrupt(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pth.tar"
    numbered = tmp_path / "pretrain_step_00000003.pth.tar"
    latest.write_bytes(b"broken latest")
    numbered.write_bytes(b"broken numbered")

    with pytest.raises(RuntimeError, match="No valid pretraining checkpoint"):
        resolve_pretraining_resume(latest, tmp_path)


@pytest.mark.parametrize(
    ("changed_key", "changed_value"),
    [
        ("dataset_sha256", "b" * 64),
        ("dataset_config", {"min_player_elo": 1500}),
        ("planned_steps", 20_000),
        ("seed", 7),
        ("wandb_run_id", "different-run"),
    ],
)
def test_resume_rejects_changed_dataset_contract(
    tmp_path: Path,
    changed_key: str,
    changed_value: object,
) -> None:
    expected: dict[str, object] = {
        "dataset_sha256": "a" * 64,
        "dataset_config": {"min_player_elo": 2000},
        "planned_steps": 10_000,
        "seed": 0,
        "wandb_run_id": "pretrain-v1",
    }
    checkpoint = tmp_path / "latest.pth.tar"
    stored = {**expected, changed_key: changed_value}
    _write_checkpoint(checkpoint, 1, stored)

    with pytest.raises(RuntimeError, match="does not match"):
        validate_resume_contract(checkpoint, expected)


class _WritingCheckpointNetwork:
    def __init__(self, global_step: int) -> None:
        self.global_step = global_step

    def save_checkpoint(
        self,
        folder: str,
        filename: str,
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        path = Path(folder) / filename
        payload: dict[str, object] = {"format_version": 2, "global_step": self.global_step}
        if extra_state is not None:
            payload.update(extra_state)
        torch.save(payload, path)


def test_checkpoint_publication_retains_only_configured_numbered_snapshots(tmp_path: Path) -> None:
    for step in range(1, 5):
        _write_checkpoint(tmp_path / f"pretrain_step_{step:08d}.pth.tar", step)
    network = _WritingCheckpointNetwork(global_step=5)
    publication = CheckpointPublication(tmp_path, keep=3, metadata={"dataset_sha256": "a" * 64})

    publish_pretraining_checkpoints(network, publication)

    numbered = sorted(path.name for path in tmp_path.glob("pretrain_step_*.pth.tar"))
    assert numbered == [
        "pretrain_step_00000003.pth.tar",
        "pretrain_step_00000004.pth.tar",
        "pretrain_step_00000005.pth.tar",
    ]
    assert (tmp_path / "latest.pth.tar").is_file()
