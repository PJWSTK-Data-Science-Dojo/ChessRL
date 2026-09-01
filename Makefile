ARGS ?=
TRAIN_ARGS ?=
CHECKPOINT_DIR ?= ./runs/luna-main
CHECKPOINT_PATH = $(CHECKPOINT_DIR)/latest.pth.tar
ARENA_CHECKPOINT_A ?=
ARENA_CHECKPOINT_B ?=
SEARCH_CONTEMPT_CHECKPOINT ?=
SEARCH_CONTEMPT_REPORT ?= ./runs/search-contempt-ablation/report.json
NEW_PHASE_SOURCE_DIR ?= ./runs/sources
NEW_PHASE_SOURCE_FILE ?= luna-balanced-precollapse-iter40.pth.tar
NEW_PHASE_SOURCE_SHA256 ?= dd07d8ddf2aa652719b405b4e3b6f7381bb652873a34d139fe37b95327ba99dd
MIGRATION_SOURCE_DIR ?= ./runs/luna-strength-1500-v1
NEW_PHASE_CHECKPOINT_DIR ?= ./runs/luna-balanced-ezv2-anti-collapse-v2
MIGRATION_CHECKPOINT_DIR ?= ./runs/luna-fairy-ladder-v1
FAIRY_STOCKFISH_PATH ?= ./vendor/stockfish/fairy-stockfish-14
TRAIN_ENV_FILE ?= .env
WANDB_PROJECT ?= ChessRL
PGN_DATA_DIR ?= ./data
PGN_DATA_FILE ?= lichess_db_broadcast_2026-07.pgn.zst
PGN_DATA_PATH ?= $(PGN_DATA_DIR)/$(PGN_DATA_FILE)
PGN_DATA_URL ?= https://database.lichess.org/broadcast/$(PGN_DATA_FILE)
PGN_DATA_SHA256 ?= 714d0eb99f99fca8d791142038b6c59b5ca6a51b3339bd3891a92f4bdffcbf0c
PGN_SOURCE_CHECKPOINT ?= ./runs/luna-balanced-ezv2-state-anchor-v3/best.pth.tar
PGN_PRETRAIN_CHECKPOINT_DIR ?= ./runs/luna-balanced-pgn-pretrain-v1
PGN_PRETRAIN_WANDB_RUN_ID ?= luna-balanced-pgn-pretrain-v1
PGN_PRETRAIN_WANDB_RUN_NAME ?= Luna Balanced · Expert PGN Pretrain v1
PGN_EVAL_CHECKPOINT ?=
PGN_SELECTED_CHECKPOINT ?=
PGN_RL_CHECKPOINT_DIR ?= ./runs/luna-balanced-ezv2-pgn-warmstart-v1
PGN_RL_WANDB_RUN_ID ?= luna-balanced-ezv2-pgn-warmstart-v1
PGN_RL_WANDB_RUN_NAME ?= Luna Balanced EZ-V2 · PGN Warm Start v1
LC0_DATA_DIR ?= ./data/lc0
LC0_DATA_FILE ?= training-run2-test91-20260901-1317.tar
LC0_DATA_PATH ?= $(LC0_DATA_DIR)/$(LC0_DATA_FILE)
LC0_DATA_URL ?= https://storage.lczero.org/files/training_data/test91/$(LC0_DATA_FILE)
LC0_DATA_SHA256 ?= d6fe77a11c71d758dfbff0d07e80958f04440d26fa1f925e0e3683e1a3ad7409
LC0_SOURCE_CHECKPOINT ?= ./runs/sources/luna-balanced-pgn-warmstart-iter25.pth.tar
LC0_SOURCE_SHA256 ?= 79376fa55a6f276f59af30479dc12f6bc939c87d91d342947578157782d4f7c6
LC0_PRETRAIN_CHECKPOINT_DIR ?= ./runs/luna-balanced-lc0-heads-pretrain-v1
LC0_PRETRAIN_WANDB_RUN_ID ?= luna-balanced-lc0-heads-pretrain-v1
LC0_PRETRAIN_WANDB_RUN_NAME ?= Luna Balanced · LC0 Policy+Value Heads v1
LC0_EVAL_CHECKPOINT ?=
LC0_SELECTED_CHECKPOINT ?=
LC0_RL_CHECKPOINT_DIR ?= ./runs/luna-balanced-ezv2-lc0-warmstart-v1
LC0_RL_WANDB_RUN_ID ?= luna-balanced-ezv2-lc0-warmstart-v1
LC0_RL_WANDB_RUN_NAME ?= Luna Balanced EZ-V2 · LC0 Warm Start v1
NEW_PHASE_WANDB_RUN_ID ?= luna-balanced-ezv2-anti-collapse-v2
NEW_PHASE_WANDB_RUN_NAME ?= Luna Balanced EZ-V2 · Anti-Collapse v2
MIGRATION_WANDB_RUN_ID ?= luna-fairy-ladder-v1
MIGRATION_WANDB_RUN_NAME ?= Luna Fairy Ladder 500+ · Benchmark 1500 v1
PUBLIC_ENV ?= .env.public
RELEASE_DIR ?= ./release
RELEASE_ID ?=
RELEASE_SOURCE ?= $(CHECKPOINT_DIR)/best.pth.tar

NEW_PHASE_IDENTITY_ARGS = \
	--wandb-project "$(WANDB_PROJECT)" \
	--wandb-run-id "$(NEW_PHASE_WANDB_RUN_ID)" \
	--wandb-run-name "$(NEW_PHASE_WANDB_RUN_NAME)" \
	--run.checkpoint "$(NEW_PHASE_CHECKPOINT_DIR)"

MIGRATION_PHASE_IDENTITY_ARGS = \
	--wandb-project "$(WANDB_PROJECT)" \
	--wandb-run-id "$(MIGRATION_WANDB_RUN_ID)" \
	--wandb-run-name "$(MIGRATION_WANDB_RUN_NAME)" \
	--run.checkpoint "$(MIGRATION_CHECKPOINT_DIR)"

PHASE_TRAIN_ARGS = \
	--run.search-mode gumbel \
	--run.gumbel-max-considered-actions 16 \
	--run.gumbel-scale 1.0 \
	--run.gumbel-value-scale 0.1 \
	--run.num-iters 400 \
	--run.num-episodes 128 \
	--run.self-play-workers 4 \
	--run.parallel-games 32 \
	--run.num-mcts-sims 32 \
	--run.recurrent-policy-topk 256 \
	--run.temp-threshold 257 \
	--run.self-play-repetition-guard \
	--run.max-ply 256 \
	--run.train-steps-per-iter 200 \
	--run.target-replay-ratio 2.0 \
	--run.lr-schedule-total-steps 60000 \
	--run.replay-capacity 300000 \
	--run.replay-warmup-positions 50000 \
	--run.stockfish-eval-every 25 \
	--run.stockfish-eval-games 20 \
	--run.stockfish-elo 1500 \
	--run.ladder-eval-every 5 \
	--run.ladder-eval-games 20 \
	--run.ladder-start-elo 500 \
	--run.ladder-step-elo 100 \
	--run.ladder-max-elo 2800 \
	--run.ladder-required-passes 2 \
	--run.ladder-depth 10 \
	--run.ladder-path "$(FAIRY_STOCKFISH_PATH)" \
	--run.checkpoint-top-k 3 \
	--learner.device cuda \
	--learner.cuda-device 0 \
	--learner.model-name balanced \
	--learner.batch-size 256 \
	--learner.num-channels 128 \
	--learner.repr-blocks 10 \
	--learner.dyn-blocks 1 \
	--learner.proj-dim 256 \
	--learner.lr 2e-4 \
	--learner.lr-min 1e-5 \
	--learner.lr-warmup-steps 1000 \
	--learner.weight-decay 1e-4 \
	--learner.amp-dtype bfloat16 \
	--learner.unroll-steps 5 \
	--learner.td-steps 5 \
	--learner.policy-loss-weight 1.0 \
	--learner.value-loss-weight 0.25 \
	--learner.reward-loss-weight 1.0 \
	--learner.consistency-loss-weight 2.0 \
	--learner.grad-accum-steps 2 \
	--learner.grad-clip-norm 5 \
	--learner.recurrent-gradient-scale 0.5 \
	--learner.dataloader-workers 4 \
	--learner.compile-inference \
	--learner.compile-training \
	--learner.reanalyze-mcts-sims 8 \
	--learner.reanalyze-prob 0.02 \
	--learner.no-reanalyze-policy \
	--learner.reanalyze-start-step 10000

WEB_COMPOSE = docker compose --env-file .env -f docker-compose.yml
PUBLIC_COMPOSE = docker compose --env-file $(PUBLIC_ENV) -f docker-compose.yml -f docker-compose.public.yml

fmt:
	uv run --frozen --extra dev ruff format .
	uv run --frozen --extra dev ruff check --fix .

format-check:
	uv run --frozen --extra dev ruff format --check .

lint:
	uv run --frozen --extra dev ruff check .

types:
	uv run --frozen --extra dev mypy src

lock-check:
	uv lock --check

check: lock-check format-check lint types test

test:
	env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen --extra dev pytest tests/ -v

audit:
	@audit_requirements=$$(mktemp); \
	trap 'rm -f "$$audit_requirements"' EXIT; \
	uv export --frozen --no-dev --extra perf --extra web --no-hashes \
		--no-emit-project --output-file "$$audit_requirements" >/dev/null; \
	uvx --from pip-audit==2.10.1 pip-audit --requirement "$$audit_requirements"

bench:
	uv run --frozen python tests/bench_throughput.py $(ARGS)

# Fresh, state-anchored single-accelerator recipe. It uses bootstrapped n-step
# targets and no self-predictive consistency during the cold start.
# ARGS is appended last so every setting can be overridden without editing this file.
train:
	uv run --frozen python src/main.py \
		--run.search-mode gumbel \
		--run.gumbel-max-considered-actions 8 \
		--run.gumbel-scale 1.0 \
		--run.gumbel-value-scale 0.1 \
		--run.num-iters 400 \
		--run.num-episodes 128 \
		--run.self-play-workers 4 \
		--run.parallel-games 32 \
		--run.num-mcts-sims 32 \
		--run.recurrent-policy-topk 256 \
		--run.temp-threshold 257 \
		--run.self-play-repetition-guard \
		--run.max-ply 256 \
		--run.train-steps-per-iter 200 \
		--run.target-replay-ratio 2.0 \
		--run.lr-schedule-total-steps 72000 \
		--run.replay-capacity 300000 \
		--run.replay-warmup-positions 50000 \
		--run.stockfish-eval-every 25 \
		--run.stockfish-eval-games 20 \
		--run.stockfish-elo 1500 \
		--run.ladder-eval-every 10 \
		--run.ladder-eval-games 20 \
		--run.ladder-start-elo 500 \
		--run.ladder-step-elo 100 \
		--run.ladder-max-elo 2800 \
		--run.ladder-required-passes 2 \
		--run.ladder-depth 10 \
		--run.ladder-path "$(FAIRY_STOCKFISH_PATH)" \
		--run.checkpoint "$(CHECKPOINT_DIR)" \
		--run.checkpoint-top-k 3 \
		--learner.device cuda \
		--learner.cuda-device 0 \
		--learner.model-name balanced_reconstruction \
		--learner.batch-size 256 \
		--learner.num-channels 128 \
		--learner.repr-blocks 10 \
		--learner.dyn-blocks 1 \
		--learner.proj-dim 256 \
		--learner.lr 1e-4 \
		--learner.lr-min 1e-5 \
		--learner.lr-warmup-steps 1000 \
		--learner.weight-decay 1e-4 \
		--learner.amp-dtype bfloat16 \
		--learner.unroll-steps 5 \
		--learner.td-steps 32 \
		--learner.policy-loss-weight 1.0 \
		--learner.value-loss-weight 1.0 \
		--learner.reward-loss-weight 0.1 \
		--learner.consistency-loss-weight 0.0 \
		--learner.reconstruction-loss-weight 0.5 \
		--learner.grad-accum-steps 2 \
		--learner.grad-clip-norm 5 \
		--learner.recurrent-gradient-scale 0.5 \
		--learner.dataloader-workers 4 \
		--learner.compile-training \
		--learner.reanalyze-mcts-sims 8 \
		--learner.reanalyze-prob 0.02 \
		--learner.reanalyze-policy \
		--learner.reanalyze-start-step 5000 \
		$(TRAIN_ARGS) $(ARGS)

resume:
	$(MAKE) train CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
		TRAIN_ARGS='--load-model --load-checkpoint-dir "$(CHECKPOINT_DIR)" --load-checkpoint-file latest.pth.tar $(TRAIN_ARGS)'

download-pgn-data:
	@mkdir -p "$(PGN_DATA_DIR)"
	@if test -f "$(PGN_DATA_PATH)"; then \
		printf '%s  %s\n' "$(PGN_DATA_SHA256)" "$(PGN_DATA_PATH)" | sha256sum --check --status; \
	else \
		curl --fail --location --continue-at - --output "$(PGN_DATA_PATH).part" "$(PGN_DATA_URL)"; \
		printf '%s  %s\n' "$(PGN_DATA_SHA256)" "$(PGN_DATA_PATH).part" | sha256sum --check --status; \
		mv "$(PGN_DATA_PATH).part" "$(PGN_DATA_PATH)"; \
	fi

verify-pgn-data:
	@test -f "$(PGN_DATA_PATH)" || { echo "PGN dataset not found; run make download-pgn-data" >&2; exit 2; }
	@printf '%s  %s\n' "$(PGN_DATA_SHA256)" "$(PGN_DATA_PATH)" | sha256sum --check --status || { \
		echo "PGN dataset SHA-256 mismatch: $(PGN_DATA_PATH)" >&2; exit 2; }

# Starts the supervised warm start once and resumes it automatically after a
# process or host failure. The trainer validates the source/data provenance.
pretrain-pgn: _train-env-preflight verify-pgn-data
	@if test -f "$(PGN_PRETRAIN_CHECKPOINT_DIR)/latest.pth.tar" || \
		find "$(PGN_PRETRAIN_CHECKPOINT_DIR)" -maxdepth 1 -type f -name 'pretrain_step_*.pth.tar' -print -quit 2>/dev/null | grep -q .; then \
		set -- --resume-checkpoint "$(PGN_PRETRAIN_CHECKPOINT_DIR)/latest.pth.tar"; \
	else \
		test -f "$(PGN_SOURCE_CHECKPOINT)" || { echo "PGN source checkpoint not found: $(PGN_SOURCE_CHECKPOINT)" >&2; exit 2; }; \
		set -- --source-checkpoint "$(PGN_SOURCE_CHECKPOINT)"; \
	fi; \
	uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python src/pretrain_pgn.py \
		--dataset-path "$(PGN_DATA_PATH)" \
		--output-dir "$(PGN_PRETRAIN_CHECKPOINT_DIR)" \
		"$$@" \
		--total-steps 10000 \
		--chunk-steps 1000 \
		--checkpoint-top-k 10 \
		--dataset.min-player-elo 2000 \
		--dataset.max-positions 300000 \
		--dataset.validation-fraction 0.05 \
		--dataset.max-game-plies 256 \
		--learner.cuda-device 0 \
		--learner.dataloader-workers 4 \
		--wandb-project "$(WANDB_PROJECT)" \
		--wandb-run-id "$(PGN_PRETRAIN_WANDB_RUN_ID)" \
		--wandb-run-name "$(PGN_PRETRAIN_WANDB_RUN_NAME)" \
		--wandb-resume allow \
		$(ARGS)

# Benchmarks one immutable PGN milestone at the first real Fairy-Stockfish Elo.
eval-pgn-warmstart: _fairy-stockfish-preflight
	@test -n "$(PGN_EVAL_CHECKPOINT)" || { echo "PGN_EVAL_CHECKPOINT is required" >&2; exit 2; }
	@test -f "$(PGN_EVAL_CHECKPOINT)" || { echo "PGN checkpoint not found: $(PGN_EVAL_CHECKPOINT)" >&2; exit 2; }
	uv run --frozen python src/eval_vs_stockfish.py \
		--checkpoint "$(PGN_EVAL_CHECKPOINT)" \
		--opponent fairy \
		--learner.device cuda \
		--learner.cuda-device 0 \
		--run.search-mode gumbel \
		--run.num-mcts-sims 32 \
		--run.evaluation-num-mcts-sims 32 \
		--run.gumbel-max-considered-actions 8 \
		--run.ladder-eval-games 20 \
		--run.ladder-start-elo 500 \
		--run.ladder-depth 10 \
		--run.ladder-path "$(FAIRY_STOCKFISH_PATH)" \
		--run.ladder-eval-max-ply 256 \
		$(ARGS)

# Starts online learning only from an explicitly benchmarked immutable PGN
# milestone. Repeated calls resume the same optimizer/checkpoint/W&B lineage.
train-pgn-warmstart: _train-env-preflight _fairy-stockfish-preflight
	@if test -f "$(PGN_RL_CHECKPOINT_DIR)/latest.pth.tar" || \
		find "$(PGN_RL_CHECKPOINT_DIR)" -maxdepth 1 -type f -name 'checkpoint_*.pth.tar' -print -quit 2>/dev/null | grep -q .; then \
		$(MAKE) resume CHECKPOINT_DIR="$(PGN_RL_CHECKPOINT_DIR)" \
			TRAIN_ARGS='--wandb-project "$(WANDB_PROJECT)" --wandb-run-id "$(PGN_RL_WANDB_RUN_ID)" --wandb-run-name "$(PGN_RL_WANDB_RUN_NAME)" --wandb-resume allow'; \
	else \
		test -n "$(PGN_SELECTED_CHECKPOINT)" || { \
			echo "PGN_SELECTED_CHECKPOINT must name a benchmarked immutable checkpoint" >&2; exit 2; }; \
		test -f "$(PGN_SELECTED_CHECKPOINT)" || { \
			echo "Selected PGN checkpoint not found: $(PGN_SELECTED_CHECKPOINT)" >&2; exit 2; }; \
		selected_checkpoint="$(PGN_SELECTED_CHECKPOINT)"; \
		source_dir="$$(dirname -- "$$selected_checkpoint")"; \
		source_file="$$(basename -- "$$selected_checkpoint")"; \
		$(MAKE) train CHECKPOINT_DIR="$(PGN_RL_CHECKPOINT_DIR)" \
			TRAIN_ARGS="--new-training-phase --load-checkpoint-dir \"$$source_dir\" --load-checkpoint-file \"$$source_file\" --wandb-project \"$(WANDB_PROJECT)\" --wandb-run-id \"$(PGN_RL_WANDB_RUN_ID)\" --wandb-run-name \"$(PGN_RL_WANDB_RUN_NAME)\" --wandb-resume allow"; \
	fi

download-lc0-data:
	@mkdir -p "$(LC0_DATA_DIR)"
	@if test -f "$(LC0_DATA_PATH)"; then \
		printf '%s  %s\n' "$(LC0_DATA_SHA256)" "$(LC0_DATA_PATH)" | sha256sum --check --status; \
	else \
		curl --fail --location --continue-at - --output "$(LC0_DATA_PATH).part" "$(LC0_DATA_URL)"; \
		printf '%s  %s\n' "$(LC0_DATA_SHA256)" "$(LC0_DATA_PATH).part" | sha256sum --check --status; \
		mv "$(LC0_DATA_PATH).part" "$(LC0_DATA_PATH)"; \
	fi

verify-lc0-data:
	@test -f "$(LC0_DATA_PATH)" || { echo "LC0 dataset not found; run make download-lc0-data" >&2; exit 2; }
	@printf '%s  %s\n' "$(LC0_DATA_SHA256)" "$(LC0_DATA_PATH)" | sha256sum --check --status || { \
		echo "LC0 dataset SHA-256 mismatch: $(LC0_DATA_PATH)" >&2; exit 2; }

# Fits the shared prediction heads using observed root states only. Keeping the
# latent model fixed protects recurrent MCTS until a milestone passes an arena.
pretrain-lc0: _train-env-preflight verify-lc0-data
	@if test -f "$(LC0_PRETRAIN_CHECKPOINT_DIR)/latest.pth.tar" || \
		find "$(LC0_PRETRAIN_CHECKPOINT_DIR)" -maxdepth 1 -type f -name 'lc0_step_*.pth.tar' -print -quit 2>/dev/null | grep -q .; then \
		set -- --resume-checkpoint "$(LC0_PRETRAIN_CHECKPOINT_DIR)/latest.pth.tar"; \
		wandb_resume=must; \
	else \
		test -f "$(LC0_SOURCE_CHECKPOINT)" || { \
			echo "LC0 source checkpoint not found: $(LC0_SOURCE_CHECKPOINT)" >&2; exit 2; }; \
		printf '%s  %s\n' "$(LC0_SOURCE_SHA256)" "$(LC0_SOURCE_CHECKPOINT)" | sha256sum --check --status || { \
			echo "LC0 source checkpoint SHA-256 mismatch: $(LC0_SOURCE_CHECKPOINT)" >&2; exit 2; }; \
		set -- --source-checkpoint "$(LC0_SOURCE_CHECKPOINT)"; \
		wandb_resume=allow; \
	fi; \
	uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python src/pretrain_lc0.py \
		--dataset-path "$(LC0_DATA_PATH)" \
		--output-dir "$(LC0_PRETRAIN_CHECKPOINT_DIR)" \
		"$$@" \
		--total-steps 1000 \
		--chunk-steps 250 \
		--checkpoint-top-k 8 \
		--validation-positions 20000 \
		--dataset.validation-fraction 0.02 \
		--dataset.min-visits 1 \
		--dataset.shuffle-buffer-size 8192 \
		--learner.lr 1e-4 \
		--learner.lr-min 1e-5 \
		--learner.lr-warmup-steps 50 \
		--learner.cuda-device 0 \
		--learner.dataloader-workers 0 \
		--wandb-project "$(WANDB_PROJECT)" \
		--wandb-run-id "$(LC0_PRETRAIN_WANDB_RUN_ID)" \
		--wandb-run-name "$(LC0_PRETRAIN_WANDB_RUN_NAME)" \
		--wandb-resume "$$wandb_resume" \
		$(ARGS)

eval-lc0-warmstart: _fairy-stockfish-preflight
	@test -n "$(LC0_EVAL_CHECKPOINT)" || { echo "LC0_EVAL_CHECKPOINT is required" >&2; exit 2; }
	@test -f "$(LC0_EVAL_CHECKPOINT)" || { echo "LC0 checkpoint not found: $(LC0_EVAL_CHECKPOINT)" >&2; exit 2; }
	uv run --frozen python src/eval_vs_stockfish.py \
		--checkpoint "$(LC0_EVAL_CHECKPOINT)" \
		--opponent fairy \
		--learner.device cuda \
		--learner.cuda-device 0 \
		--run.search-mode gumbel \
		--run.num-mcts-sims 32 \
		--run.evaluation-num-mcts-sims 32 \
		--run.gumbel-max-considered-actions 8 \
		--run.ladder-eval-games 20 \
		--run.ladder-start-elo 500 \
		--run.ladder-depth 10 \
		--run.ladder-path "$(FAIRY_STOCKFISH_PATH)" \
		--run.ladder-eval-max-ply 256 \
		$(ARGS)

# A benchmarked LC0 milestone starts a new online optimizer and W&B lineage.
train-lc0-warmstart: _train-env-preflight _fairy-stockfish-preflight
	@if test -f "$(LC0_RL_CHECKPOINT_DIR)/latest.pth.tar" || \
		find "$(LC0_RL_CHECKPOINT_DIR)" -maxdepth 1 -type f -name 'checkpoint_*.pth.tar' -print -quit 2>/dev/null | grep -q .; then \
		$(MAKE) resume CHECKPOINT_DIR="$(LC0_RL_CHECKPOINT_DIR)" \
			TRAIN_ARGS='--wandb-project "$(WANDB_PROJECT)" --wandb-run-id "$(LC0_RL_WANDB_RUN_ID)" --wandb-run-name "$(LC0_RL_WANDB_RUN_NAME)" --wandb-resume must'; \
	else \
		test -n "$(LC0_SELECTED_CHECKPOINT)" || { \
			echo "LC0_SELECTED_CHECKPOINT must name a benchmarked immutable checkpoint" >&2; exit 2; }; \
		test -f "$(LC0_SELECTED_CHECKPOINT)" || { \
			echo "Selected LC0 checkpoint not found: $(LC0_SELECTED_CHECKPOINT)" >&2; exit 2; }; \
		selected_checkpoint="$(LC0_SELECTED_CHECKPOINT)"; \
		source_dir="$$(dirname -- "$$selected_checkpoint")"; \
		source_file="$$(basename -- "$$selected_checkpoint")"; \
		$(MAKE) train CHECKPOINT_DIR="$(LC0_RL_CHECKPOINT_DIR)" \
			TRAIN_ARGS="--new-training-phase --load-checkpoint-dir \"$$source_dir\" --load-checkpoint-file \"$$source_file\" --wandb-project \"$(WANDB_PROJECT)\" --wandb-run-id \"$(LC0_RL_WANDB_RUN_ID)\" --wandb-run-name \"$(LC0_RL_WANDB_RUN_NAME)\" --wandb-resume allow"; \
	fi

_train-env-preflight:
	@test -f "$(TRAIN_ENV_FILE)" || { echo "Training environment file not found: $(TRAIN_ENV_FILE)" >&2; exit 2; }
	@uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python -c \
		'import os, sys; names = ("WANDB_API_KEY", "WANDB_ENTITY"); missing = [n for n in names if not os.environ.get(n)]; sys.exit("Missing training environment variables: " + ", ".join(missing)) if missing else None'

_fairy-stockfish-preflight:
	@test -x "$(FAIRY_STOCKFISH_PATH)" || { \
		echo "Fairy-Stockfish binary not found: $(FAIRY_STOCKFISH_PATH); run make install-fairy-stockfish" >&2; exit 2; }

# Start a distinct optimizer/LR phase from validated v2 weights. The target
# directory must not contain any files; src/main.py enforces that contract.
train-phase: _train-env-preflight _fairy-stockfish-preflight
	@test -f "$(NEW_PHASE_SOURCE_DIR)/$(NEW_PHASE_SOURCE_FILE)" || { \
		echo "Phase source checkpoint not found: $(NEW_PHASE_SOURCE_DIR)/$(NEW_PHASE_SOURCE_FILE)" >&2; exit 2; }
	@printf '%s  %s\n' "$(NEW_PHASE_SOURCE_SHA256)" "$(NEW_PHASE_SOURCE_DIR)/$(NEW_PHASE_SOURCE_FILE)" \
		| sha256sum --check --status || { echo "Phase source checkpoint SHA-256 mismatch" >&2; exit 2; }
	uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python src/main.py \
		--new-training-phase \
		--load-checkpoint-dir "$(NEW_PHASE_SOURCE_DIR)" \
		--load-checkpoint-file "$(NEW_PHASE_SOURCE_FILE)" \
		--wandb-resume never \
		$(NEW_PHASE_IDENTITY_ARGS) \
		$(PHASE_TRAIN_ARGS) $(ARGS)

# Resume the same phase contract after an interruption without resetting
# optimizer, scaler, counters, or the LR schedule horizon.
resume-phase: _train-env-preflight _fairy-stockfish-preflight
	@test -f "$(NEW_PHASE_CHECKPOINT_DIR)/latest.pth.tar" || \
		find "$(NEW_PHASE_CHECKPOINT_DIR)" -maxdepth 1 -type f -name 'checkpoint_*.pth.tar' -print -quit | grep -q . || { \
		echo "No phase resume checkpoint found in $(NEW_PHASE_CHECKPOINT_DIR)" >&2; exit 2; }
	uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python src/main.py \
		--load-model \
		--load-checkpoint-dir "$(NEW_PHASE_CHECKPOINT_DIR)" \
		--load-checkpoint-file latest.pth.tar \
		--wandb-resume must \
		$(NEW_PHASE_IDENTITY_ARGS) \
		$(PHASE_TRAIN_ARGS) $(ARGS)

# Continue the complete optimizer/checkpoint state in the new ladder contract.
# The target must be absent or empty on first use; a pre-checkpoint retry may contain only validated evaluation sidecars.
migrate-ladder-phase: _train-env-preflight _fairy-stockfish-preflight
	@test -f "$(MIGRATION_SOURCE_DIR)/latest.pth.tar" || { \
		echo "Migration source not found: $(MIGRATION_SOURCE_DIR)/latest.pth.tar" >&2; exit 2; }
	uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python src/main.py \
		--load-model \
		--initialize-evaluation-state \
		--load-checkpoint-dir "$(MIGRATION_SOURCE_DIR)" \
		--load-checkpoint-file latest.pth.tar \
		--wandb-resume never \
		$(MIGRATION_PHASE_IDENTITY_ARGS) \
		$(PHASE_TRAIN_ARGS) $(ARGS)

# Resume the complete-state migration lineage without reusing the weights-only
# phase target or W&B identity.
resume-migrated-phase: _train-env-preflight _fairy-stockfish-preflight
	@test -f "$(MIGRATION_CHECKPOINT_DIR)/latest.pth.tar" || \
		find "$(MIGRATION_CHECKPOINT_DIR)" -maxdepth 1 -type f -name 'checkpoint_*.pth.tar' -print -quit | grep -q . || { \
		echo "No migrated phase resume checkpoint found in $(MIGRATION_CHECKPOINT_DIR)" >&2; exit 2; }
	uv run --frozen --env-file "$(TRAIN_ENV_FILE)" python src/main.py \
		--load-model \
		--load-checkpoint-dir "$(MIGRATION_CHECKPOINT_DIR)" \
		--load-checkpoint-file latest.pth.tar \
		--wandb-resume must \
		$(MIGRATION_PHASE_IDENTITY_ARGS) \
		$(PHASE_TRAIN_ARGS) $(ARGS)

# Short end-to-end run with phase timings and a compact profiler trace.
profile-smoke:
	uv run --frozen python src/main.py \
		--run.num-iters 1 \
		--run.num-episodes 2 \
		--run.parallel-games 2 \
		--run.num-mcts-sims 4 \
		--run.max-ply 40 \
		--run.train-steps-per-iter 12 \
		--run.stockfish-eval-every 0 \
		--run.checkpoint "" \
		--run.profile \
		--run.profile-torch-steps 8 \
		--run.profile-dir ./profiles \
		--run.profile-tensorboard-logdir ./profiles/tb \
		--learner.batch-size 8 \
		--learner.num-channels 32 \
		--learner.repr-blocks 2 \
		--learner.dyn-blocks 1 \
		--learner.proj-dim 64 \
		--learner.dataloader-workers 0 \
		$(ARGS)

serve:
	uv run --frozen python src/web_app.py --checkpoint "$(CHECKPOINT_PATH)" $(ARGS)

serve-cpu:
	uv run --frozen python src/web_app.py --checkpoint "$(CHECKPOINT_PATH)" --device cpu --search-simulations 25 --no-compile-inference $(ARGS)

serve-mps:
	uv run --frozen python src/web_app.py --checkpoint "$(CHECKPOINT_PATH)" --device mps --search-simulations 50 --no-compile-inference $(ARGS)

release-web-model:
	@test -n "$(RELEASE_ID)" || { echo "RELEASE_ID is required, for example RELEASE_ID=iteration-400" >&2; exit 2; }
	@case "$(RELEASE_ID)" in *[!A-Za-z0-9._-]*) echo "RELEASE_ID may contain only letters, digits, dots, underscores, and hyphens" >&2; exit 2;; esac
	@test -f "$(RELEASE_SOURCE)" || { echo "Evaluated checkpoint not found: $(RELEASE_SOURCE)" >&2; exit 2; }
	@mkdir -p "$(RELEASE_DIR)"
	@test ! -e "$(RELEASE_DIR)/luna-$(RELEASE_ID).pth.tar" || { echo "Release already exists: $(RELEASE_DIR)/luna-$(RELEASE_ID).pth.tar" >&2; exit 2; }
	@install -m 0444 "$(RELEASE_SOURCE)" "$(RELEASE_DIR)/luna-$(RELEASE_ID).pth.tar"
	@cd "$(RELEASE_DIR)" && sha256sum "luna-$(RELEASE_ID).pth.tar" > "luna-$(RELEASE_ID).pth.tar.sha256"
	@echo "Created immutable web release: $(RELEASE_DIR)/luna-$(RELEASE_ID).pth.tar"
	@cat "$(RELEASE_DIR)/luna-$(RELEASE_ID).pth.tar.sha256"

verify-web-model:
	@test -n "$(RELEASE_ID)" || { echo "RELEASE_ID is required" >&2; exit 2; }
	@cd "$(RELEASE_DIR)" && sha256sum --check "luna-$(RELEASE_ID).pth.tar.sha256"

web-config:
	$(WEB_COMPOSE) config --quiet

web-build: web-config
	$(WEB_COMPOSE) build luna-web

web-up: web-config
	$(WEB_COMPOSE) up --detach --build luna-web

web-down:
	$(WEB_COMPOSE) down

web-logs:
	$(WEB_COMPOSE) logs --follow --tail=100 luna-web

web-public-config:
	$(PUBLIC_COMPOSE) config --quiet

web-public-up: web-public-config
	$(PUBLIC_COMPOSE) up --detach --build

web-public-down:
	$(PUBLIC_COMPOSE) down

web-public-logs:
	$(PUBLIC_COMPOSE) logs --follow --tail=100 luna-web cloudflared

uci:
	uv run --frozen luna-uci --checkpoint "$(CHECKPOINT_PATH)" $(ARGS)

lichess-config:
	uv run --frozen luna-lichess-config --checkpoint "$(abspath $(CHECKPOINT_PATH))" $(ARGS)

eval-stockfish:
	uv run --frozen python src/eval_vs_stockfish.py --checkpoint "$(CHECKPOINT_PATH)" $(ARGS)

eval-checkpoints:
	@test -n "$(ARENA_CHECKPOINT_A)" || { echo "ARENA_CHECKPOINT_A is required" >&2; exit 2; }
	@test -n "$(ARENA_CHECKPOINT_B)" || { echo "ARENA_CHECKPOINT_B is required" >&2; exit 2; }
	@test -f "$(ARENA_CHECKPOINT_A)" || { echo "Checkpoint A not found: $(ARENA_CHECKPOINT_A)" >&2; exit 2; }
	@test -f "$(ARENA_CHECKPOINT_B)" || { echo "Checkpoint B not found: $(ARENA_CHECKPOINT_B)" >&2; exit 2; }
	uv run --frozen python src/eval_vs_checkpoint.py \
		--checkpoint-a "$(ARENA_CHECKPOINT_A)" \
		--checkpoint-b "$(ARENA_CHECKPOINT_B)" \
		$(ARGS)

ablate-search-contempt:
	@test -n "$(SEARCH_CONTEMPT_CHECKPOINT)" || { echo "SEARCH_CONTEMPT_CHECKPOINT is required" >&2; exit 2; }
	@test -f "$(SEARCH_CONTEMPT_CHECKPOINT)" || { echo "Checkpoint not found: $(SEARCH_CONTEMPT_CHECKPOINT)" >&2; exit 2; }
	uv run --frozen python src/ablate_search_contempt.py \
		--checkpoint "$(SEARCH_CONTEMPT_CHECKPOINT)" \
		--output "$(SEARCH_CONTEMPT_REPORT)" \
		$(ARGS)

install-fairy-stockfish:
	bash scripts/install-fairy-stockfish.sh

test-pipeline-cpu:
	uv run --frozen python src/main.py \
		--run.num-iters 2 \
		--run.num-episodes 2 \
		--run.parallel-games 2 \
		--run.num-mcts-sims 4 \
		--run.max-ply 32 \
		--run.train-steps-per-iter 4 \
		--run.stockfish-eval-every 0 \
		--run.checkpoint "" \
		--learner.device cpu \
		--learner.batch-size 4 \
		--learner.num-channels 16 \
		--learner.repr-blocks 1 \
		--learner.dyn-blocks 1 \
		--learner.proj-dim 32 \
		--learner.dataloader-workers 0 \
		$(ARGS)

test-pipeline-mps:
	$(MAKE) test-pipeline-cpu ARGS="--learner.device mps $(ARGS)"

.PHONY: _fairy-stockfish-preflight _train-env-preflight ablate-search-contempt audit bench check download-lc0-data download-pgn-data \
	eval-checkpoints eval-lc0-warmstart eval-pgn-warmstart eval-stockfish fmt format-check install-fairy-stockfish \
	lichess-config lint \
	migrate-ladder-phase pretrain-lc0 pretrain-pgn profile-smoke release-web-model resume resume-migrated-phase resume-phase \
	serve serve-cpu serve-mps test test-pipeline-cpu test-pipeline-mps train train-lc0-warmstart train-phase train-pgn-warmstart \
	types uci verify-lc0-data verify-pgn-data verify-web-model web-build web-config web-down web-logs web-public-config \
	web-public-down web-public-logs web-public-up web-up
