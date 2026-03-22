
help: ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	uv run pytest

fast-test: ## Run tests that are not slow
	uv run pytest -m "not slow"

pre-commit: ## Install pre commit hooks
	uv run pre-commit install
	uv run pre-commit install-hooks

format: ## Format with pre commit
	uv run pre-commit run --all-files

bump: ## Sync environment
	uv sync

train: ## Run training (single GPU)
	uv run python train_gpt.py

train-rtx: ## Run distributed training (2x3090)
	uv run torchrun --standalone --nproc_per_node=2 train_gpt.py

