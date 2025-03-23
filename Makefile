test:
	@uv run pytest tests/

lint: 
	@uvx ruff format src/
	@uvx ruff check src/

