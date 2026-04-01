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

train:
	uv run python src/main.py

serve:
	uv run python src/luna_html_wrapper.py

.PHONY: fmt lint types check test bench train serve
