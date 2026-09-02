"""Training Makefile target contract tests."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("target", "resume_mode", "run_id", "run_name"),
    [
        (
            "train-phase",
            "never",
            "luna-balanced-ezv2-anti-collapse-v2",
            "Luna Balanced EZ-V2 · Anti-Collapse v2",
        ),
        (
            "resume-phase",
            "must",
            "luna-balanced-ezv2-anti-collapse-v2",
            "Luna Balanced EZ-V2 · Anti-Collapse v2",
        ),
        (
            "migrate-ladder-phase",
            "never",
            "luna-fairy-ladder-v1",
            "Luna Fairy Ladder 500+ · Benchmark 1500 v1",
        ),
        (
            "resume-migrated-phase",
            "must",
            "luna-fairy-ladder-v1",
            "Luna Fairy Ladder 500+ · Benchmark 1500 v1",
        ),
    ],
)
def test_phase_make_target_sets_explicit_wandb_resume_policy(
    target: str,
    resume_mode: str,
    run_id: str,
    run_name: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", target],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f'--wandb-run-id "{run_id}"' in result.stdout
    assert f'--wandb-run-name "{run_name}"' in result.stdout
    assert f"--wandb-resume {resume_mode}" in result.stdout
    assert "--run.self-play-workers 4" in result.stdout
    assert "--run.stockfish-elo 1500" in result.stdout
    assert "--run.ladder-start-elo 500" in result.stdout
    assert "--run.ladder-step-elo 100" in result.stdout
    assert "--learner.model-name balanced" in result.stdout
    assert "--learner.repr-blocks 10" in result.stdout
    assert "--learner.dyn-blocks 1" in result.stdout
    assert "--learner.unroll-steps 5" in result.stdout
    assert "--learner.td-steps 5" in result.stdout
    assert "--run.self-play-repetition-guard" in result.stdout
    assert "--run.target-replay-ratio 2.0" in result.stdout
    assert "--run.lr-schedule-total-steps 60000" in result.stdout
    assert "--run.replay-warmup-positions 50000" in result.stdout
    assert "--learner.reanalyze-mcts-sims 8" in result.stdout
    assert "--learner.reanalyze-prob 0.02" in result.stdout
    assert "--learner.no-reanalyze-policy" in result.stdout
    assert "--learner.reanalyze-start-step 10000" in result.stdout
    if target == "migrate-ladder-phase":
        assert "--initialize-evaluation-state" in result.stdout


def test_train_phase_make_target_keeps_pinned_source_preflight() -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", "train-phase"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "luna-balanced-precollapse-iter40.pth.tar" in result.stdout
    assert "dd07d8ddf2aa652719b405b4e3b6f7381bb652873a34d139fe37b95327ba99dd" in result.stdout


def test_maintained_train_target_uses_bootstrapped_state_anchored_contract() -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", "train"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_flags = (
        "--run.search-mode gumbel",
        "--run.gumbel-max-considered-actions 8",
        "--run.num-mcts-sims 32",
        "--run.self-play-workers 4",
        "--run.stockfish-eval-every 25",
        "--run.stockfish-elo 1500",
        "--run.ladder-eval-every 10",
        "--run.ladder-start-elo 500",
        "--learner.model-name balanced_reconstruction",
        "--learner.batch-size 256",
        "--learner.repr-blocks 10",
        "--learner.dyn-blocks 1",
        "--learner.unroll-steps 5",
        "--learner.td-steps 32",
        "--learner.lr 1e-4",
        "--learner.value-loss-weight 1.0",
        "--learner.reward-loss-weight 0.1",
        "--learner.consistency-loss-weight 0.0",
        "--learner.reconstruction-loss-weight 0.5",
        "--learner.compile-training",
        "--run.self-play-repetition-guard",
        "--run.target-replay-ratio 2.0",
        "--run.lr-schedule-total-steps 72000",
        "--run.replay-warmup-positions 50000",
        "--learner.reanalyze-mcts-sims 8",
        "--learner.reanalyze-prob 0.02",
        "--learner.reanalyze-policy",
        "--learner.reanalyze-start-step 5000",
    )
    for flag in expected_flags:
        assert flag in result.stdout
    assert "--learner.no-reanalyze-policy" not in result.stdout
    assert "--load-model" not in result.stdout
    assert "--new-training-phase" not in result.stdout
