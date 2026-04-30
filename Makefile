fmt:
	uv run --extra dev ruff format .
	uv run --extra dev ruff check --fix .

lint:
	uv run --extra dev ruff check .

types:
	uv run --extra dev mypy src

check: lint types

test:
	uv run --extra dev pytest tests/ -v

bench:
	uv run python tests/bench_throughput.py

# Example: one iter, small run, phase timings + 15-step CUDA trace in ./profiles/
# Run `make torch-fix` first if PyTorch does not see CUDA on your machine.
# On Volta+ GPUs (sm_70+), append `--learner.compile-inference` for faster MCTS forwards.
profile-smoke:
	uv run python src/main.py \
		--run.num-iters 1 \
		--run.num-episodes 2 \
		--run.parallel-games 2 \
		--run.num-mcts-sims 4 \
		--run.max-ply 40 \
		--run.arena-compare 2 \
		--run.train-steps-per-iter 30 \
		--run.batch-size 8 \
		--run.profile \
		--run.profile-torch-steps 15 \
		--run.profile-dir ./profiles \
		--run.profile-tensorboard-logdir ./profiles/tb \
		--run.profile-with-stack \
		--learner.batch-size 8 \
		--learner.num-channels 32 \
		--learner.repr-blocks 2 \
		--learner.dyn-blocks 1 \
		--learner.proj-dim 64 \
		--learner.grad-accum-steps 1 \
		--learner.dataloader-workers 0

# Extra CLI flags for main.py, e.g. make train ARGS='--num-mcts-sims 20'
ARGS ?=

# Extra args for training, e.g. make train TRAIN_ARGS='--num-mcts-sims 20'
TRAIN_ARGS ?=

train:
	uv run python src/main.py $(TRAIN_ARGS) $(ARGS)

# Reinstall a stable torch range that supports older and newer CUDA GPUs.
torch-fix:
	uv pip install --python .venv/bin/python "torch>=2.4,<2.6"

serve:
	uv run python src/web_app.py

.PHONY: fmt lint types check test bench train serve torch-fix profile-smoke
