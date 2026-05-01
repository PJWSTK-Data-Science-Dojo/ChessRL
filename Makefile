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

# Optimized RTX 4090 config: Maximal GPU utilization (24GB VRAM)
# Target: 2000-2200 Elo with 128ch (40M params), 200 MCTS sims, 8+4 blocks
# Increased parallel games (48→64), arena parallelism (12→16), dataloader workers (4→6)
train-rtx4090-balanced:
	uv run python src/main.py \
		--run.num-iters 400 \
		--run.num-episodes 50 \
		--run.parallel-games 64 \
		--run.num-mcts-sims 200 \
		--run.recurrent-policy-topk 1024 \
		--run.train-steps-per-iter 300 \
		--run.batch-size 96 \
		--run.arena-compare 40 \
		--run.arena-num-mcts-sims 200 \
		--run.arena-parallel-games 16 \
		--run.replay-capacity 300000 \
		--learner.device cuda \
		--learner.batch-size 96 \
		--learner.num-channels 128 \
		--learner.repr-blocks 8 \
		--learner.dyn-blocks 4 \
		--learner.support-size 15 \
		--learner.proj-dim 512 \
		--learner.lr 2e-4 \
		--learner.grad-accum-steps 2 \
		--learner.dataloader-workers 6 \
		--learner.compile-inference \
		--learner.reanalyze-mcts-sims 25 \
		--learner.reanalyze-prob 0.25 \
		--learner.mixed-value-td-until-step 5000 \
		$(ARGS)

# Reinstall a stable torch range that supports older and newer CUDA GPUs.
torch-fix:
	uv pip install --python .venv/bin/python "torch>=2.4,<2.6"

serve:
	uv run python src/web_app.py

# Serve on CPU (for MacBook or systems without GPU)
serve-cpu:
	uv run python src/web_app.py --device cpu --mcts-sims 25

# Serve on MPS (for Apple Silicon Macs)
serve-mps:
	uv run python src/web_app.py --device mps --mcts-sims 50

# MacBook pipeline testing: minimal config for testing training loop on CPU/MPS
test-pipeline-macbook:
	uv run python src/main.py \
		--run.num-iters 3 \
		--run.num-episodes 4 \
		--run.parallel-games 2 \
		--run.num-mcts-sims 8 \
		--run.max-ply 60 \
		--run.train-steps-per-iter 20 \
		--run.batch-size 8 \
		--run.arena-compare 4 \
		--learner.device cpu \
		--learner.batch-size 8 \
		--learner.num-channels 32 \
		--learner.repr-blocks 2 \
		--learner.dyn-blocks 1 \
		--learner.proj-dim 64 \
		--learner.dataloader-workers 2 \
		$(ARGS)

# Same as above but with MPS backend for Apple Silicon Macs
test-pipeline-macbook-mps:
	uv run python src/main.py \
		--run.num-iters 3 \
		--run.num-episodes 4 \
		--run.parallel-games 2 \
		--run.num-mcts-sims 8 \
		--run.max-ply 60 \
		--run.train-steps-per-iter 20 \
		--run.batch-size 8 \
		--run.arena-compare 4 \
		--learner.device mps \
		--learner.batch-size 8 \
		--learner.num-channels 32 \
		--learner.repr-blocks 2 \
		--learner.dyn-blocks 1 \
		--learner.proj-dim 64 \
		--learner.dataloader-workers 2 \
		$(ARGS)

.PHONY: fmt lint types check test bench train serve serve-cpu serve-mps torch-fix profile-smoke train-rtx4090-balanced test-pipeline-macbook test-pipeline-macbook-mps
