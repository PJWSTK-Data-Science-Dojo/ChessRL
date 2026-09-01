"""Dry-run contract tests for the search-contempt training canary."""

import subprocess
from pathlib import Path

SOURCE_FILE = "luna-lc0-warmstart-iter30.pth.tar"
TEST_SOURCE_SHA256 = "527bcbaf2964073969f8fc7597ff48eb028d97ba420c56922776925efba9f123"


def _dry_run_canary(repository: Path, source: Path, target: Path) -> str:
    result = subprocess.run(
        [
            "make",
            "-n",
            "train-search-contempt-canary",
            f"SEARCH_CONTEMPT_CANARY_SOURCE_DIR={source}",
            f"SEARCH_CONTEMPT_CANARY_SOURCE_SHA256={TEST_SOURCE_SHA256}",
            f"SEARCH_CONTEMPT_CANARY_CHECKPOINT_DIR={target}",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _active_training_command(output: str) -> str:
    return output.rsplit("uv run --frozen python src/main.py", maxsplit=1)[-1]


def _assert_canary_contract(command: str) -> None:
    assert '--wandb-run-id "luna-balanced-ezv2-search-contempt-l8-canary-v1"' in command
    assert '--wandb-run-name "Luna Balanced EZ-V2 · Search-contempt L8 Canary v1"' in command
    assert "--run.num-iters 40" in command
    assert "--run.checkpoint-top-k 12" in command
    assert "--run.search-contempt-visit-limit 8" in command
    assert "--run.temp-threshold 40" in command
    assert "--run.stockfish-eval-every 10" in command
    assert "--run.ladder-start-elo 600" in command
    assert "--run.ladder-eval-every 5" in command


def test_canary_pins_the_validated_lc0_warmstart_source() -> None:
    repository = Path(__file__).resolve().parents[1]
    makefile = (repository / "Makefile").read_text(encoding="utf-8")

    assert "SEARCH_CONTEMPT_CANARY_SOURCE_DIR ?= ./runs/sources" in makefile
    assert f"SEARCH_CONTEMPT_CANARY_SOURCE_FILE ?= {SOURCE_FILE}" in makefile
    assert (
        "SEARCH_CONTEMPT_CANARY_SOURCE_SHA256 ?= edb1aee2b5c560eb7b38ba3209c52baa9bc4d982cefa1ce7eed0f8f9448cba4a"
    ) in makefile


def test_canary_first_start_migrates_complete_state(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    (source / SOURCE_FILE).write_bytes(b"source checkpoint")
    target = tmp_path / "canary"
    target.mkdir()
    (target / "ladder_eval_state.json").write_text("{}", encoding="utf-8")

    output = _dry_run_canary(repository, source, target)
    command = _active_training_command(output)

    _assert_canary_contract(command)
    assert "sha256sum --check --status" in output
    assert f'--run.checkpoint "{target}"' in command
    assert "--load-model" in command
    assert "--initialize-evaluation-state" in command
    assert f'--load-checkpoint-dir "{source}"' in command
    assert f'--load-checkpoint-file "{SOURCE_FILE}"' in command
    assert "--wandb-resume allow" in command
    assert "--new-training-phase" not in command


def test_canary_resume_keeps_identity_and_overrides(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    target = tmp_path / "canary"
    target.mkdir()
    (target / "latest.pth.tar").write_bytes(b"canary checkpoint")

    output = _dry_run_canary(repository, tmp_path / "unused-source", target)
    command = _active_training_command(output)

    _assert_canary_contract(command)
    assert f'--run.checkpoint "{target}"' in command
    assert f'--load-checkpoint-dir "{target}"' in command
    assert "--wandb-resume must" in command
    assert "--initialize-evaluation-state" not in command
