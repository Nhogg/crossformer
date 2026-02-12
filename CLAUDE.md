# CLAUDE.md

## Project Overview

CrossFormer is a JAX/Flax transformer-based robot policy for cross-embodied learning, trained on 900K trajectories across 20 robot embodiments. Built on the [Octo codebase](https://github.com/octo-models/octo).

## Package Layout

```
crossformer/           # Main package
  model/               # Flax model architecture (CrossFormerModel, transformer, action heads)
  data/                # Data pipelines (legacy TF/RLDS + newer Grain)
    grain/             # Google Grain pipeline (active migration target)
    arec/              # ArrayRecord tools
    oxe/               # Open X-Embodiment dataset configs
  cn/                  # Config system (tyro-based dataclasses)
  utils/               # JAX utils, training, specs, tree ops, type checking
  viz/                 # Visualization
scripts/               # Training, finetuning, data prep, debug, configs
tests/                 # pytest suite (unit, integration, hypothesis property tests)
docs/                  # Documentation and plans
roadmap/               # Active roadmap items (remove when completed)
```

## Build & Run

- **Package manager**: `uv` (with `uv_build` backend)
- **Python**: >=3.11 (target 3.11)
- **Install deps**: `uv sync --group dev`
- **Run scripts**: `uv run script.py` (never `python` directly or activate venv)
- **JAX**: 0.5.3 with CUDA 12

## Linting & Formatting

- **Ruff** (line-length 120, target py311): `uv run ruff check .` / `uv run ruff format .`
- **Pre-commit** runs ruff lint+format, check-yaml, check-ast, trailing-whitespace, nb-clean
- **Required import**: every `.py` file must have `from __future__ import annotations`
- **Type checking**: `uv run pyright` (basic mode)
- **isort**: via ruff, force-sort-within-sections, no order-by-type

## Testing

- **Run**: `uv run pytest` (defaults: `-q --maxfail=1 --disable-warnings`)
- **With coverage**: `uv run pytest --cov` (fail_under=80, branch coverage)
- **Markers**: `unit` (fast), `integration` (requires GPU), `multinode`
- **Frameworks**: pytest + hypothesis (property-based tests)
- Tests use real JAX/TF/NumPy -- no mocking of numerical backends
- Seed explicitly: `np.random.default_rng(seed)`, `jax.random.PRNGKey(n)`
- Shape assertions are the primary contract (`assert output.shape == (...)`)

## Code Style

### General
- Concise code and docstrings; purpose should be clear from reading the code
- Short, meaningful variable names (avoid long names that clutter)
- Keep functions small and single-purpose (cyclomatic complexity <= 8, prefer 4-5)
- DRY -- no duplicated code
- Flat over nested; avoid excessive nesting
- Feature-based file organization, not layer-based
- OOP patterns for components; decorators/wrappers when appropriate
- `config.create(*args, **kwargs)` pattern for component instantiation

### Python
- Type hints on public API; `X | None` union syntax (via `__future__` annotations)
- Google-style docstrings with inline field comments for dataclasses
- snake_case functions/variables, PascalCase classes, UPPER_CASE constants
- Private functions prefixed with `_`
- `functools.partial` for currying, `einops.rearrange` for tensor reshaping

### JAX/Flax
- `nn.Module` with `@nn.compact` for neural network components
- `@flax.struct.dataclass` for PyTree-compatible data containers
- `module.apply({"params": params}, ...)` for inference
- `jax.tree.map()` for tree operations; nested dict trees throughout
- `@partial(jax.jit, static_argnames=(...))` for JIT
- `jax.lax.scan` for iterative processes, `jax.vmap` for batching
- `ModuleSpec` system (`crossformer/utils/spec.py`) for config-driven instantiation

### Data Pipeline
- Two parallel systems: legacy TF/RLDS (`data/dataset.py`) and Grain (`data/grain/`)
- Grain is the active migration target
- Data stored as nested dict trees: `{"observation": {"image": ..., "proprio": ...}, "task": ..., "action": ...}`
- `flat()`/`unflat()` helpers for dot-separated key access

## Git Conventions

- Main branch: `main`
- Feature branches: `wip-*` or topic names
- Commit messages: short, lowercase, descriptive (conventional commits optional)
- CI: pre-commit hooks only (GitHub Actions on PRs and pushes to main)
- No force-push to main

## Things to Avoid

- Excessive nesting
- Long variable names
- Long functions with multiple responsibilities
- Duplicated code
- Mocking JAX/TF/NumPy in tests (use real backends)
- Happy-path-only tests (write negative tests, edge cases)
- Adding features or refactoring beyond what was asked
- Shims unless explicitly needed
