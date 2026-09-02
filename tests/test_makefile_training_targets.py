"""Training Makefile target contract tests."""

import hashlib
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


def test_lc0_10m_pretrain_target_pins_objective_identity_and_resume_modes() -> None:
    repository = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        ["make", "-n", "pretrain-lc0-exact-10m", "ARGS=--learner.lr 9e-5"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_flags = (
        '--wandb-run-id "luna-lc0-exact-10m-recovery-v1"',
        '--wandb-run-name "Luna LC0 Exact · 10M Joint Recovery v1"',
        "--train-scope representation_and_heads",
        "--dataset.value-source root",
        "--total-steps 20000",
        "--chunk-steps 1000",
        "--validation-positions 50000",
        "--learner.batch-size 512",
        "--learner.no-compile-training",
    )
    for flag in expected_flags:
        assert flag in result.stdout
    assert "wandb_resume=never" in result.stdout
    assert "wandb_resume=allow" in result.stdout
    assert '--wandb-resume "$wandb_resume"' in result.stdout
    assert result.stdout.rfind("--learner.lr 9e-5") > result.stdout.rfind("--learner.lr 1e-4")


def test_lc0_pcr_anchor_target_expands_resume_and_fresh_contracts(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shard = corpus / "audit-shard.tar"
    shard.write_bytes(b"contract-only shard")
    shard_digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    checkpoint_dir = tmp_path / "online"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "latest.pth.tar").write_bytes(b"dry-run checkpoint")

    result = subprocess.run(
        [
            "make",
            "-n",
            "train-lc0-exact-pcr-anchor",
            f"LC0_10M_DATA_PATH={corpus}",
            "LC0_10M_SHARD_COUNT=1",
            f"LC0_10M_SHARDS={shard.name}:{shard_digest}",
            f"LC0_PCR_CHECKPOINT_DIR={checkpoint_dir}",
            "ARGS=--learner.lr 3e-5",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_flags = (
        '--wandb-run-id "luna-lc0-exact-pcr128x16-p25-anchor25-v1"',
        '--wandb-run-name "Luna LC0 Exact · PCR 128/16 p25 + Expert Anchor 25% v1"',
        "--wandb-resume allow",
        "--wandb-resume never",
        "--load-model",
        "--new-training-phase",
        "--run.tree-state-mode exact",
        "--run.playout-cap-full-sims 128",
        "--run.playout-cap-fast-sims 16",
        "--run.playout-cap-full-probability 0.25",
        "--run.evaluation-num-mcts-sims 32",
        "--run.stockfish-elo 1500",
        "--run.ladder-eval-every 5",
        "--run.ladder-start-elo 700",
        "--run.replay-capacity 500000",
        "--run.replay-warmup-positions 100000",
        "--learner.batch-size 512",
        "--learner.expert-anchor-fraction 0.25",
        "--learner.expert-anchor-loss-weight 0.25",
        "--learner.no-compile-inference",
        "--learner.no-compile-training",
        "--learner.no-reanalyze-policy",
    )
    for flag in expected_flags:
        assert flag in result.stdout
    assert str(corpus) in result.stdout
    assert str(checkpoint_dir) in result.stdout
    assert result.stdout.rfind("--wandb-resume allow") > result.stdout.rfind("--wandb-resume never")
    assert result.stdout.rfind("--learner.no-compile-training") > result.stdout.rfind("--learner.compile-training")
    assert result.stdout.rfind("--learner.no-reanalyze-policy") > result.stdout.rfind("--learner.reanalyze-policy")
    assert result.stdout.rfind("--learner.lr 3e-5") > result.stdout.rfind("--learner.lr 2e-5")
