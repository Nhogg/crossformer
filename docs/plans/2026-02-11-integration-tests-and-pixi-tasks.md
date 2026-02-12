# Integration Tests + Pixi Tasks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-backed integration tests for model checkpoint round-trips and Grain pipeline contracts, plus pixi tasks for test types.

**Architecture:** Introduce a minimal CrossFormer config for fast integration tests, validating save/load behavior and action sampling determinism. Add a Grain pipeline integration test using synthetic trajectories to validate end-to-end data flow. Register pytest markers and define pixi tasks that map to unit/integration/multinode scopes.

**Tech Stack:** JAX, Flax, Orbax, pytest, Grain, pixi

---

### Task 1: Add model checkpoint integration test

**Files:**
- Create: `tests/integration/ckpt/test_model_checkpoint.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```python
@pytest.mark.integration
def test_model_checkpoint_roundtrip_gpu(tmp_path):
    model = _build_tiny_model()
    actions = model.sample_actions(...)
    model.save_pretrained(...)
    reloaded = CrossFormerModel.load_pretrained(...)
    reloaded_actions = reloaded.sample_actions(...)
    assert jnp.allclose(actions, reloaded_actions)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/ckpt/test_model_checkpoint.py::test_model_checkpoint_roundtrip_gpu -v`
Expected: FAIL with missing file / missing helper / not implemented errors.

**Step 3: Write minimal implementation**

Implement helpers in the same test file:

- `_require_gpu()` that raises if no CUDA devices.
- `_make_example_batch()` with small shapes and `timestep_pad_mask`.
- `_make_config()` with a `LowdimObsTokenizer` + `L1ActionHead` and tiny transformer settings.

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/ckpt/test_model_checkpoint.py::test_model_checkpoint_roundtrip_gpu -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/ckpt/test_model_checkpoint.py pyproject.toml
git commit -m "test: add model checkpoint integration test"
```

### Task 2: Add Grain pipeline integration test

**Files:**
- Create: `tests/integration/ckpt/test_grain_pipeline.py`

**Step 1: Write the failing test**

```python
@pytest.mark.integration
def test_grain_pipeline_contract(tmp_path):
    result = pipelines.make_single_dataset(...)
    frame = next(iter(result.dataset))
    assert frame["observation"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/ckpt/test_grain_pipeline.py::test_grain_pipeline_contract -v`
Expected: FAIL with missing helper / bad import errors.

**Step 3: Write minimal implementation**

Add synthetic trajectory helpers in the test file (no mocks), configure a tiny `GrainDatasetConfig`, run
`pipelines.make_single_dataset`, and assert:

- `dataset_name` matches config
- `action` has expected `(window, action_horizon, action_dim)`
- `pad_mask_dict` exists and is boolean

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/ckpt/test_grain_pipeline.py::test_grain_pipeline_contract -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/ckpt/test_grain_pipeline.py
git commit -m "test: add grain pipeline integration test"
```

### Task 3: Add pixi tasks for test scopes

**Files:**
- Modify: `pixi.toml`
- Modify: `pyproject.toml`

**Step 1: Write the failing config**

Add `unit`, `integration`, `multinode`, and `ckpt` tasks under `[tasks]` in `pixi.toml`.

**Step 2: Run command to verify markers are recognized**

Run: `pytest -m integration --collect-only`
Expected: No warnings about unknown markers.

**Step 3: Write minimal implementation**

Update `pyproject.toml` to register markers:

```toml
[tool.pytest.ini_options]
markers = [
  "unit: fast, pure tests",
  "integration: GPU-backed integration tests",
  "multinode: multi-node tests run via scripts",
]
```

**Step 4: Run verification**

Run: `pytest -m integration --collect-only`
Expected: PASS with no marker warnings.

**Step 5: Commit**

```bash
git add pixi.toml pyproject.toml
git commit -m "build: add pixi test tasks and pytest markers"
```

---

Plan complete and saved to `docs/plans/2026-02-11-integration-tests-and-pixi-tasks.md`.
Two execution options:

1. Subagent-Driven (this session) - I dispatch a fresh subagent per task and review between tasks
2. Parallel Session (separate) - Open a new session with executing-plans and batch execution with checkpoints

Which approach?
