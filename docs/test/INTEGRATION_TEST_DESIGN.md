# Integration Test Design Plan

Based on codebase analysis and spec in `docs/test/integration.md`, this document details the structure, helpers, test cases, and refactoring needs for comprehensive integration testing.

---

## 1. Test File Organization

### Primary Files
```
tests/integration/
├── conftest.py                    # Existing: model configs, fake batches, fixtures
├── test_train_step.py             # Training: single step, 20 steps, weight updates
├── test_checkpoint.py             # Save/load: round-trip, resume from checkpoint
├── test_eval.py                   # Evaluation: metrics finite, determinism
├── test_config_instantiation.py   # Config: dict-based and tyro-based creation
├── test_multihead.py              # Multi-head: multi-action-head training
└── helpers.py                     # NEW: Reusable test utilities
```

### Supporting Files
- `conftest.py` — augment existing with new fixtures for different head types
- Update `pyproject.toml` to ensure pytest markers are set correctly (already done: integration, unit, multinode)

---

## 2. Required Helper Functions (in `helpers.py`)

### Pure Step Boundaries (Core Refactoring)

These functions must be **extracted/created** to make training logic testable:

```python
def init_train_state(
    rng: PRNGKey,
    example_batch: Data,
    config: dict,
    text_processor: TextProcessor | None = None,
    dataset_statistics: Data | None = None,
    optimizer_kwargs: dict | None = None,
) -> TrainState:
    """Initialize TrainState from scratch.

    Returns:
        TrainState with initialized model, optimizer, and RNG.
    """
    # 1. Create model from config
    # 2. Initialize params from example_batch
    # 3. Create optimizer
    # 4. Initialize TrainState
    # No global state reads; all config passed in

def train_step(
    state: TrainState,
    batch: Data,
    rng: PRNGKey,
) -> tuple[TrainState, dict[str, Any]]:
    """Single training step.

    Args:
        state: Current training state
        batch: Input batch (observation, task, action, masks)
        rng: RNG for dropout

    Returns:
        (new_state, metrics_dict)
    """
    # 1. Bind module with params and RNG
    # 2. Run forward pass (transformer + heads)
    # 3. Compute loss for each head
    # 4. Backprop and update params
    # 5. Return new state and metrics

def eval_step(
    state: TrainState,
    batch: Data,
    rng: PRNGKey,
) -> dict[str, Any]:
    """Evaluation forward pass (no gradient update).

    Returns:
        metrics_dict with loss and per-head metrics
    """
    # 1. Bind module in eval mode (train=False)
    # 2. Run forward and loss computation
    # 3. No gradient update
    # 4. Return metrics

def save_ckpt(state: TrainState, path: str | Path) -> None:
    """Save checkpoint to disk."""
    # Use model.save_pretrained() + save optimizer state

def load_ckpt(path: str | Path) -> TrainState:
    """Load checkpoint from disk."""
    # Use CrossFormerModel.load_pretrained() + restore optimizer state
```

### Synthetic Data Generation

```python
def make_tiny_config(
    transformer_size: str = "dummy",
    head_names: list[str] | None = None,
    action_horizon: int = 4,
) -> dict:
    """Create a minimal model config for fast testing.

    Args:
        transformer_size: "dummy" (1 layer, 256 dim) or "vanilla" (4 layers, 256 dim)
        head_names: List of action head names; default ["single"]
        action_horizon: Action horizon; default 4

    Returns:
        Config dict with model, optimizer, and eval specs
    """
    # Already mostly implemented in conftest.py
    # Extend to support multiple head types and sizes

def make_fake_batch(
    batch_size: int = 2,
    window_size: int = 1,
    img_size: int = 64,
    action_dim: int = 7,
    action_horizon: int = 4,
    head_names: list[str] | None = None,
    seed: int = 0,
) -> Data:
    """Generate a synthetic batch matching config shape.

    Args:
        batch_size: Number of samples
        window_size: Sequence length
        img_size: Image resolution
        action_dim: Action dimension per head
        action_horizon: Action horizon (lookahead)
        head_names: List of head names (default ["single"])
        seed: Random seed for determinism

    Returns:
        Batch dict with observation, task, action, masks
    """
    # Already mostly implemented in conftest.py
    # Ensure it matches conftest.py structure

def make_optimizer_kwargs(
    learning_rate: float = 1e-4,
    clip_gradient: float | None = None,
    weight_decay: float = 0.01,
) -> dict:
    """Create optimizer kwargs for testing.

    Returns:
        Dict for passing to create_optimizer()
    """
```

### Assertion Helpers

```python
def assert_finite(x: Any, name: str = "") -> None:
    """Assert all array values are finite."""
    # JAX tree.map over all leaves, check isfinite

def assert_loss_decreased(
    loss_0: float,
    loss_1: float,
    rtol: float = 0.01,  # 1% improvement
) -> None:
    """Assert that loss decreased by at least rtol after one step."""

def assert_params_changed(
    params_old: Params,
    params_new: Params,
    min_changed_fraction: float = 0.1,
) -> None:
    """Assert that at least min_changed_fraction of params changed."""

def assert_metrics_keys(
    metrics: dict,
    expected_heads: list[str],
) -> None:
    """Assert metrics dict has expected structure."""
    # Check for each head_name key, and within it: loss, mse, lsign
```

### GPU/Device Management

```python
def has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    # Existing in conftest.py as _has_gpu()

def get_gpu_memory_gb() -> float:
    """Get available GPU memory."""

@contextmanager
def require_gpu(reason: str = ""):
    """Context that checks GPU availability and logs reason if skipped."""
```

---

## 3. Test Case Structure and Assertions

### File: `test_train_step.py`

**Scope**: Single GPU, single-process training with fixed batch

```python
@pytest.mark.integration
@pytest.mark.skipif(not has_gpu(), reason="Integration tests require GPU")
class TestTrainStep:
    """Test single and multi-step training."""

    def test_single_train_step_produces_finite_loss(self, model_config, example_batch):
        """One gradient step should produce finite loss."""
        # 1. init_train_state(rng, example_batch, model_config)
        # 2. train_step(state, batch, rng)
        # 3. assert_finite(loss)
        # 4. assert loss in reasonable range (e.g., 0, 100)

    def test_single_train_step_updates_params(self, model_config, example_batch):
        """At least some parameters should change after one step."""
        # 1. init_train_state()
        # 2. Store old params
        # 3. train_step()
        # 4. assert_params_changed(params_old, params_new, min_changed_fraction=0.1)

    def test_train_10_steps_loss_finite(self, model_config, example_batch):
        """Train for 10 steps; loss should stay finite."""
        # 1. init_train_state()
        # 2. Loop 10 times: train_step(), store loss
        # 3. assert all losses are finite
        # 4. assert final step == 10

    def test_train_10_steps_loss_decreasing(self, model_config, example_batch):
        """Loss should generally decrease over steps (with high learning rate)."""
        # 1. init_train_state(optimizer_kwargs={"learning_rate": 1e-2})
        # 2. Loop 10 steps, record losses
        # 3. Fit to trend (exponential decay or linear)
        # 4. assert overall trend is decreasing (allow noise)

    def test_multihead_single_step(self, multihead_config, example_batch_multihead):
        """Single step with multiple heads should compute loss for each."""
        # 1. init_train_state(multihead_config)
        # 2. train_step()
        # 3. assert metrics has keys: ["single", "bimanual", "mano", "total_loss"]
        # 4. assert each head loss is finite

    def test_train_step_jit_matches_non_jit(self, model_config, example_batch):
        """JIT-compiled step should match eager execution."""
        # 1. init_train_state()
        # 2. Run step twice, once with @jax.jit, once without
        # 3. assert_allclose(jit_loss, eager_loss)
        # 4. assert_allclose(jit_params, eager_params)
```

**Assertions per spec**:
- ✅ Gradients are finite
- ✅ Optimizer updates weights
- ✅ Loss changes (ideally decreases)
- ✅ Metrics are measurable

---

### File: `test_checkpoint.py`

**Scope**: Save/load and resume semantics

```python
@pytest.mark.integration
@pytest.mark.skipif(not has_gpu(), reason="Integration tests require GPU")
class TestCheckpoint:
    """Test checkpoint save/load round-trips and resume."""

    def test_save_and_load_params_identical(self, model_config, example_batch, tmp_path):
        """Saved and loaded params should yield identical outputs."""
        # 1. init_train_state() → state_0
        # 2. save_ckpt(state_0, tmp_path / "ckpt_0")
        # 3. load_ckpt(tmp_path / "ckpt_0") → loaded_state
        # 4. Run transformer forward on both
        # 5. assert_allclose(output_0, output_loaded, atol=1e-5)

    def test_save_after_step_and_resume(self, model_config, example_batch, tmp_path):
        """Training can resume from checkpoint and continue."""
        # 1. init_train_state() → state_0
        # 2. train_step() → state_1
        # 3. save_ckpt(state_1, tmp_path / "ckpt_1")
        # 4. train_step() → state_2
        # 5. load_ckpt(tmp_path / "ckpt_1") → resumed_state
        # 6. train_step(resumed_state) → state_2_resumed
        # 7. assert params_allclose(state_2, state_2_resumed)
        # 8. assert steps match (state_2.step == 2, state_2_resumed.step == 2)

    def test_optimizer_state_persists(self, model_config, example_batch, tmp_path):
        """Optimizer state (e.g., Adam momentum) should be restored."""
        # 1. Create state with adamw optimizer
        # 2. train_step(state) multiple times
        # 3. save_ckpt()
        # 4. load_ckpt()
        # 5. Compare opt_state (internal structure, just assert not None)
        # 6. Next step should use restored momentum

    def test_multiple_checkpoint_versions(self, model_config, example_batch, tmp_path):
        """Save multiple checkpoints and load them correctly."""
        # 1. Save at steps 0, 5, 10
        # 2. Load each and verify step count
        # 3. Verify outputs differ (params have changed)

    def test_checkpoint_across_heads(self, multihead_config, example_batch_multihead, tmp_path):
        """Checkpoint should include all head parameters."""
        # 1. init_train_state(multihead_config)
        # 2. save_ckpt()
        # 3. load_ckpt() and verify all heads present
        # 4. train_step() on loaded state should work for all heads
```

**Assertions per spec**:
- ✅ Checkpoint round-trip preserves outputs
- ✅ State dict reload yields identical outputs (within tolerance)

---

### File: `test_eval.py`

**Scope**: Evaluation-only runs (no gradients)

```python
@pytest.mark.integration
@pytest.mark.skipif(not has_gpu(), reason="Integration tests require GPU")
class TestEval:
    """Test evaluation loop and metrics."""

    def test_eval_produces_finite_metrics(self, model_config, example_batch):
        """Eval forward pass should produce finite metrics."""
        # 1. init_train_state()
        # 2. eval_step(state, batch, rng)
        # 3. assert_finite(metrics)
        # 4. Check keys: ["loss", "mse", "lsign"] for each head

    def test_eval_deterministic(self, model_config, example_batch):
        """Eval with same RNG should be deterministic (no dropout)."""
        # 1. init_train_state()
        # 2. eval_step(state, batch, rng_fixed)
        # 3. eval_step(state, batch, rng_fixed) again
        # 4. assert_allclose(metrics_1, metrics_2, atol=1e-6)

    def test_eval_no_gradient_flow(self, model_config, example_batch):
        """Eval should not compute or accumulate gradients."""
        # 1. init_train_state()
        # 2. Use JAX tracer to detect if any backward() is called
        # 3. eval_step() should not raise tracer errors

    def test_eval_metrics_structure_multihead(self, multihead_config, example_batch_multihead):
        """Eval metrics should reflect all heads."""
        # 1. init_train_state(multihead_config)
        # 2. eval_step()
        # 3. assert "single" in metrics and "bimanual" in metrics
        # 4. assert each has loss, mse, lsign

    def test_eval_after_train_shows_improvement(self, model_config, example_batch):
        """Eval loss should improve after training steps."""
        # 1. init_train_state(high_learning_rate)
        # 2. eval_step() → eval_loss_0
        # 3. Loop 10 train_step()
        # 4. eval_step() → eval_loss_1
        # 5. assert eval_loss_1 < eval_loss_0
```

**Assertions per spec**:
- ✅ Evaluation metrics are measurable
- ✅ Metrics within expected ranges

---

### File: `test_config_instantiation.py` (NEW)

**Scope**: Config-driven model creation

```python
@pytest.mark.integration
@pytest.mark.skipif(not has_gpu(), reason="Integration tests require GPU")
class TestConfigInstantiation:
    """Test that models can be created from various config formats."""

    def test_dict_config_to_model(self):
        """Dict-based config (Octo style) should create a model."""
        # 1. Create dict config with ModuleSpec entries
        # 2. CrossFormerModule.create(**config["model"])
        # 3. Initialize and verify

    def test_tyro_dataclass_config_to_model(self):
        """Tyro-based dataclass (cn.Train) should create a model."""
        # 1. Create cn.Train config
        # 2. Extract model part and create
        # 3. Initialize and verify

    def test_config_with_different_head_types(self):
        """Config should support different action head types."""
        # For each head type: ContinuousActionHead, DiffusionActionHead, FlowMatchingActionHead
        # 1. Create config with that head
        # 2. Initialize model
        # 3. train_step() should work

    def test_example_batch_shape_validation(self):
        """Model should validate batch shapes match config."""
        # 1. Create config for (B, W, H, W, 3) images
        # 2. Pass batch with wrong shape
        # 3. run_transformer() should raise AssertionError with helpful message
```

---

### File: `test_multihead.py` (NEW)

**Scope**: Multi-head training (multiple action spaces)

```python
@pytest.mark.integration
@pytest.mark.skipif(not has_gpu(), reason="Integration tests require GPU")
class TestMultiHead:
    """Test multi-head action prediction."""

    def test_multihead_loss_aggregation(self, multihead_config, example_batch_multihead):
        """Multiple heads should aggregate losses correctly."""
        # 1. init_train_state(multihead_config with 3 heads)
        # 2. train_step()
        # 3. assert total_loss == sum of head losses (weighted)
        # 4. Check that each head gets non-zero gradient

    def test_multihead_independent_action_dims(self, multihead_config, example_batch_multihead):
        """Each head should have its own action_dim."""
        # 1. Config: single=7, bimanual=14, mano=16
        # 2. init_train_state()
        # 3. Verify action head output shapes match

    def test_embodiment_mask_selects_heads(self, multihead_config, example_batch_multihead):
        """embodiment mask should select which heads are active."""
        # 1. Create batch where only "single" head is embodied (mask=1)
        # 2. train_step()
        # 3. Verify "single" gets loss, others are masked out
```

---

## 4. GPU Availability and Failure Modes

### Skip Conditions
- All integration tests skip if GPU unavailable (via `@pytest.mark.skipif(not has_gpu(), reason=...)`)
- Use decorator pattern from existing `requires_gpu` in `conftest.py`

### Graceful Degradation
- If OOM: catch `RuntimeError` with "out of memory" and skip with reason
- If compilation timeout: set `jax.config.jax_compilation_cache_dir` and warn
- If data loading fails: log and skip specific data test

### Logging
- Capture GPU memory usage before/after each test
- Log compile times for first forward/backward pass
- Log wall-clock time per test

---

## 5. Performance Budget Strategies

**Each test < 30s on single GPU; total suite < 10 minutes**

### Strategies to stay under budget:

1. **Model size**: Use "dummy" transformer (1 layer, 2 heads, 256 dim)
   - Initialization: ~1-2s
   - Forward: ~0.5s per batch
   - Backward: ~1-2s per step
   - Total per test: ~5-10s

2. **Batch size**: `batch_size=2, window_size=1`
   - Keeps activation memory low
   - Still exercises batching logic

3. **Sequence length**: `window_size=1`
   - Single timestep (no temporal modeling in test)
   - Reduces memory and compute

4. **Image size**: `img_size=64` (not 224)
   - ResNet26FILM still works, but ~4x faster

5. **Number of steps**:
   - Single-step tests: 1 step
   - Multi-step tests: 10 steps (not 100)

6. **JIT compilation caching**:
   - Set `JAX_COMPILATION_CACHE_DIR`
   - Second test run should be faster

### Expected timings:
- test_single_train_step: ~8s (JIT compile + 1 step)
- test_train_10_steps_loss_finite: ~15s (10 steps, JIT cache hit)
- test_checkpoint_round_trip: ~8s (2 forward passes)
- test_eval_*: ~5s each (no backward)
- **Total**: ~8-10 minutes

---

## 6. Refactoring Needed in `crossformer/` (Do NOT implement yet)

### High Priority (Required for testable architecture)

1. **Extract `init_train_state()` function**
   - **Location**: `crossformer/utils/train_utils.py`
   - **Current code**: Scattered in `scripts/finetune.py:177-203`
   - **Impact**: Currently, model + optimizer + TrainState init logic is in script; needs to be reusable
   - **Signature**: `init_train_state(rng, example_batch, config, text_processor, dataset_statistics, optimizer_kwargs) -> TrainState`

2. **Extract `train_step()` function**
   - **Location**: `crossformer/utils/train_utils.py` (or new `crossformer/utils/train_loop.py`)
   - **Current code**: Pattern in `scripts/finetune.py:285`, logic in finetune's loss_fn
   - **Impact**: Hardcoded in script; needs to be a pure function
   - **Signature**: `train_step(state, batch, rng) -> (TrainState, dict[str, Any])`
   - **Note**: Loss function logic (transformer + all heads) can stay as helper or be part of train_step

3. **Extract `eval_step()` function**
   - **Location**: `crossformer/utils/train_utils.py`
   - **Current code**: `scripts/finetune.py:306` (val_fn), but incomplete
   - **Signature**: `eval_step(state, batch, rng) -> dict[str, Any]`

4. **Extract `save_ckpt() / load_ckpt()` functions**
   - **Location**: `crossformer/utils/train_utils.py`
   - **Current code**: `CrossFormerModel.save_pretrained()` handles model, but not full TrainState or optimizer
   - **Issue**: Currently orbax calls are scattered; need unified checkpoint API
   - **Signature**:
     ```python
     save_ckpt(state: TrainState, path: str) -> None
     load_ckpt(path: str) -> TrainState
     ```

### Medium Priority (Cleaner test support)

5. **Modularize loss function**
   - **Location**: Create `crossformer/utils/losses.py`
   - **Current code**: Loss logic in `scripts/finetune.py:242-276`
   - **Issue**: Hard to test separately from train_step
   - **Function**: `compute_loss(params, batch, module, rng, train=True) -> (loss, metrics)`

6. **Standardize batch format in conftest**
   - **Location**: `tests/integration/conftest.py`
   - **Issue**: `make_fake_batch()` exists but could support more variations (multihead, different head types)
   - **Extensions**: Add params for `head_names`, `action_dims_per_head`, `has_language`

7. **Create optimizer helper**
   - **Location**: Already in `crossformer/utils/train_utils.py` as `create_optimizer()`
   - **Issue**: Works, but documentation could be clearer on signature
   - **Task**: Ensure it returns `(tx, lr_callable, param_norm_callable)` and document

### Low Priority (Nice-to-have)

8. **Add type hints to TrainState**
   - **Location**: `crossformer/utils/train_utils.py:28`
   - **Current code**: Fields not annotated with full types
   - **Task**: Add `Params`, `OptState` type aliases

9. **Metrics aggregation helper**
   - **Location**: `crossformer/utils/metrics.py` (new)
   - **Task**: Collect per-head metrics into aggregated dict; handle multi-process case

---

## 7. Fixtures and Utilities in conftest.py

### New fixtures to add:

```python
@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Whether GPU is available."""
    return len(jax.devices("gpu")) > 0

@pytest.fixture(scope="module")
def tiny_config_flow() -> dict:
    """Config with FlowMatchingActionHead (default)."""
    return make_tiny_config(head_names=["single"])

@pytest.fixture(scope="module")
def tiny_config_continuous() -> dict:
    """Config with ContinuousActionHead."""
    # Similar to flow, but use ContinuousActionHead

@pytest.fixture(scope="module")
def tiny_config_diffusion() -> dict:
    """Config with DiffusionActionHead."""
    # Diffusion requires more setup (betas, alphas)

@pytest.fixture(scope="module")
def multihead_config() -> dict:
    """Config with 3 heads: single (7d), bimanual (14d), mano (16d)."""
    return make_tiny_config(
        head_names=["single", "bimanual", "mano"],
        action_dims={"single": 7, "bimanual": 14, "mano": 16}
    )

@pytest.fixture(scope="module")
def example_batch_multihead() -> Data:
    """Multihead batch with corresponding actions."""
    return make_fake_batch(
        head_names=["single", "bimanual", "mano"],
        action_dims={"single": 7, "bimanual": 14, "mano": 16}
    )

@pytest.fixture(scope="function")
def tmp_ckpt_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    return tmp_path / "checkpoints"
```

### Update existing fixtures:

```python
@pytest.fixture(scope="module")
def tiny_config():
    """Make it more flexible with params."""
    def _make(transformer_size="dummy", head_names=None, **kwargs):
        return make_tiny_config(transformer_size, head_names, **kwargs)
    return _make

@pytest.fixture(scope="module")
def example_batch():
    """Make it more flexible with params."""
    def _make(batch_size=2, window_size=1, head_names=None, **kwargs):
        return make_fake_batch(batch_size, window_size, head_names=head_names, **kwargs)
    return _make
```

---

## 8. File-by-File Structure Summary

### `tests/integration/conftest.py`
```python
# Existing
- _has_gpu()
- requires_gpu marker
- make_tiny_config()
- make_fake_batch()
- tiny_config fixture
- example_batch fixture
- model_and_batch fixture

# New
- gpu_available() fixture
- tiny_config_continuous, tiny_config_diffusion fixtures
- multihead_config, example_batch_multihead fixtures
- Helper: make_optimizer_kwargs()
```

### `tests/integration/helpers.py` (NEW)
```python
# Pure functions
- init_train_state()
- train_step()
- eval_step()
- save_ckpt()
- load_ckpt()
- make_optimizer_kwargs()

# Assertions
- assert_finite()
- assert_loss_decreased()
- assert_params_changed()
- assert_metrics_keys()

# Device utilities
- has_gpu()
- get_gpu_memory_gb()
```

### `tests/integration/test_train_step.py`
```python
class TestTrainStep:
  - test_single_train_step_produces_finite_loss
  - test_single_train_step_updates_params
  - test_train_10_steps_loss_finite
  - test_train_10_steps_loss_decreasing
  - test_multihead_single_step
  - test_train_step_jit_matches_non_jit
```

### `tests/integration/test_checkpoint.py`
```python
class TestCheckpoint:
  - test_save_and_load_params_identical
  - test_save_after_step_and_resume
  - test_optimizer_state_persists
  - test_multiple_checkpoint_versions
  - test_checkpoint_across_heads
```

### `tests/integration/test_eval.py`
```python
class TestEval:
  - test_eval_produces_finite_metrics
  - test_eval_deterministic
  - test_eval_no_gradient_flow
  - test_eval_metrics_structure_multihead
  - test_eval_after_train_shows_improvement
```

### `tests/integration/test_config_instantiation.py` (NEW)
```python
class TestConfigInstantiation:
  - test_dict_config_to_model
  - test_tyro_dataclass_config_to_model
  - test_config_with_different_head_types
  - test_example_batch_shape_validation
```

### `tests/integration/test_multihead.py` (NEW)
```python
class TestMultiHead:
  - test_multihead_loss_aggregation
  - test_multihead_independent_action_dims
  - test_embodiment_mask_selects_heads
```

---

## 9. Refactoring Implementation Order

**Do NOT implement until task #3 approval**

1. **Phase 1** (blocking):
   - Extract `init_train_state()` from finetune.py
   - Extract `train_step()` logic
   - Extract `eval_step()` logic
   - Create `tests/integration/helpers.py` with these functions

2. **Phase 2** (supporting):
   - Extract `save_ckpt() / load_ckpt()` wrapper
   - Modularize loss function
   - Update conftest.py with new fixtures

3. **Phase 3** (polish):
   - Add type hints
   - Create metrics helpers
   - Document refactored functions

---

## 10. Testing Matrix

| Test | Model | Heads | Steps | Timeout | GPU Mem |
|------|-------|-------|-------|---------|---------|
| single_train_step | dummy | single | 1 | 15s | ~2GB |
| train_10_steps | dummy | single | 10 | 20s | ~2GB |
| loss_decreasing | dummy | single | 10 | 20s | ~2GB |
| multihead_step | dummy | 3 heads | 1 | 15s | ~3GB |
| eval_finite | dummy | single | 0 | 5s | ~1GB |
| eval_deterministic | dummy | single | 0 | 5s | ~1GB |
| checkpoint_round_trip | dummy | single | 0 | 10s | ~2GB |
| checkpoint_resume | dummy | single | 2 | 20s | ~2GB |
| config_dict_to_model | dummy | single | 0 | 10s | ~2GB |
| config_tyro_to_model | dummy | single | 0 | 10s | ~2GB |
| **TOTAL** | | | | ~10m | |

---

## 11. Success Criteria for Task #2 Design

✅ **File organization** — Clear structure with separate test files
✅ **Helper functions** — Pure train_step, eval_step, init_train_state with signatures
✅ **Test cases** — At least 15 specific tests covering train, eval, checkpoint, config
✅ **Assertions** — Clear assertion patterns per spec (finite, decreased loss, param updates, shape matching)
✅ **GPU handling** — Documented skip conditions and failure modes
✅ **Performance** — Detailed strategies to stay under 30s per test
✅ **Refactoring** — Listed 9 refactoring tasks with locations and impact
✅ **Fixtures** — New fixtures for different configs and heads

---

## Next Steps (Task #3)

Once this design is approved:
1. Implement refactoring tasks in Phase 1 (helpers extraction)
2. Create test files with all test cases
3. Update conftest.py with new fixtures
4. Run tests and iterate
