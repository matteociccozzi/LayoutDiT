test:
	@uv run pytest tests/

lint: 
	@uvx ruff format src/
	@uvx ruff check --fix src/

sync:
	@uv sync --locked --all-groups --all-extras

lock:
	@uv lock