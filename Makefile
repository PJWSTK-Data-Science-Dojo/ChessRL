ARGS ?=
TRAIN_ARGS ?=
CHECKPOINT_DIR ?= ./runs/luna-main
CHECKPOINT_PATH = $(CHECKPOINT_DIR)/latest.pth.tar
PUBLIC_ENV ?= .env.public
RELEASE_DIR ?= ./release
RELEASE_ID ?=
RELEASE_SOURCE ?= $(CHECKPOINT_DIR)/best.pth.tar

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

check: format-check lint types test

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

# A bounded, high-throughput single-accelerator training recipe. ARGS is
# appended last so every setting can be overridden without editing this file.
train:
	uv run --frozen python src/main.py \
		--run.search-mode gumbel \
		--run.gumbel-max-considered-actions 16 \
		--run.num-iters 400 \
		--run.num-episodes 32 \
		--run.parallel-games 32 \
		--run.num-mcts-sims 32 \
		--run.recurrent-policy-topk 256 \
		--run.temp-threshold 20 \
		--run.max-ply 256 \
		--run.train-steps-per-iter 150 \
		--run.replay-capacity 150000 \
		--run.stockfish-eval-every 25 \
		--run.stockfish-eval-games 8 \
		--run.stockfish-elo 1320 \
		--run.checkpoint "$(CHECKPOINT_DIR)" \
		--run.checkpoint-top-k 3 \
		--learner.device cuda \
		--learner.cuda-device 0 \
		--learner.batch-size 64 \
		--learner.num-channels 128 \
		--learner.repr-blocks 8 \
		--learner.dyn-blocks 3 \
		--learner.proj-dim 256 \
		--learner.lr 2e-4 \
		--learner.lr-warmup-steps 1000 \
		--learner.grad-accum-steps 2 \
		--learner.grad-clip-norm 5 \
		--learner.recurrent-gradient-scale 0.5 \
		--learner.compile-inference \
		--learner.reanalyze-mcts-sims 16 \
		--learner.reanalyze-prob 0.25 \
		--learner.reanalyze-start-step 15000 \
		$(TRAIN_ARGS) $(ARGS)

resume:
	$(MAKE) train CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
		TRAIN_ARGS='--load-model --load-checkpoint-dir "$(CHECKPOINT_DIR)" --load-checkpoint-file latest.pth.tar $(TRAIN_ARGS)'

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

.PHONY: audit bench check eval-stockfish fmt format-check lichess-config lint profile-smoke \
	release-web-model resume serve serve-cpu serve-mps test test-pipeline-cpu test-pipeline-mps \
	train types uci verify-web-model web-build web-config web-down web-logs web-public-config \
	web-public-down web-public-logs web-public-up web-up
