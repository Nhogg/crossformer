# Test Coverage Roadmap

## Current Coverage Snapshot

### Well Exercised Areas
- **Action heads** (`tests/test_action_heads.py`)
  - Validates forward passes, clipping, sampling, and loss masking logic for `ContinuousActionHead`, `DiffusionActionHead`, and `FlowMatchingActionHead` in `crossformer/model/components/action_heads.py`.
  - Confirms error handling for missing diffusion inputs and ensures stochastic sampling respects configured bounds.
- **Trajectory transforms** (`tests/test_traj_transforms.py`)
  - Uses property-based testing to cover `chunk_act_obs`, including pre-chunked actions, override window sizes, and failure paths when horizons are inconsistent.
  - Exercises downstream helpers such as `pad_actions_and_proprio`, `add_head_action_mask`, `subsample`, `zero_out_future_proprio`, and `add_pad_mask_dict` in `crossformer/data/traj_transforms.py` across a wide range of shapes and boolean mask combinations.

### Under-tested or Missing Coverage
- **Core transformer assembly** (`crossformer/model/components/block_transformer.py`, `crossformer/model/components/transformer.py`, and `crossformer/model/crossformer_module.py`)
  - No automated checks for assembling prefix/timestep tokens, attention mask generation, positional encodings, or causal rule enforcement.
- **Diffusion and flow utilities** (`crossformer/model/components/diffusion.py`)
  - Lack of tests for `cosine_beta_schedule`, residual MLP blocks, and `create_diffusion_model` wiring, leaving numerical stability and shape expectations unguarded.
- **High-level model facade** (`crossformer/model/crossformer_model.py`)
  - `create_tasks`, `run_transformer`, `sample_actions`, checkpoint load/save, and normalization pathways are untested, including interaction with multiple heads and text processors.
- **Dataset pipeline** (`crossformer/data/dataset.py`, `crossformer/data/obs_transforms.py`, and `crossformer/data/utils/*`)
  - No regression tests around trajectory/frame transform orchestration, goal relabeling, task augmentation, normalization utilities, or TF dataset threading helpers.
- **Utilities and infrastructure** (`crossformer/utils/spec.py`, `crossformer/utils/jax_utils.py`, `crossformer/utils/train_utils.py`, `crossformer/utils/train_callbacks.py`)
  - Absent coverage for configuration helpers, sharding utilities, training state transitions, scheduling helpers, and callback behaviors.
- **Tokenizers and visualization helpers** (`crossformer/model/components/tokenizers.py`, `crossformer/viz/*`)
  - Rendering and tokenizer composition logic currently rely solely on manual testing.

## Coverage Improvement Plan

### Phase 1 – Foundation Unit Tests
1. Add focused unit tests for small utilities:
   - `ModuleSpec.create/instantiate/to_string` round-trips and error handling.
   - `TokenGroup.create/concatenate` shape assertions and mask broadcasting.
   - `cosine_beta_schedule` monotonicity and range properties.
2. Extend `tests/test_action_heads.py` with cases for `pool_strategy="use_map"` and invalid strategy errors to lock down the remaining branches.

### Phase 2 – Transformer Assembly Validation
1. Construct lightweight dummy tokenizers to exercise `BlockTransformer.assemble_input_tokens`, `split_output_tokens`, and `generate_attention_mask`, verifying causal masking and prefix/timestep splits.
2. Cover `CrossFormerTransformer` end-to-end with a tiny configuration, asserting that requested readouts are produced, task repetition works, and attention rules respect `repeat_task_tokens`.

### Phase 3 – Data Pipeline Integration
1. Build synthetic `dl.DLataset` fixtures to test `apply_trajectory_transforms` and `apply_frame_transforms`, confirming that map/filters compose correctly, head masks propagate, and subsampling respects training flags.
2. Add regression tests for `goal_relabeling`, `task_augmentation`, and normalization utilities to prevent silent data-shape regressions.

### Phase 4 – Model and Training Facade
1. Introduce tests for `CrossFormerModel.create_tasks`, `sample_actions` (including normalization/unnormalization paths), and checkpoint load/save routines using mocked heads and minimal params.
2. Validate `TrainState` transitions, `Timer`, `format_name_with_config`, learning-rate schedulers, and multihost helpers in `train_utils`/`jax_utils` with host-local stubs.

### Phase 5 – Optional UX & Visualization Checks
1. Snapshot-based tests for visualization utilities (e.g., `_oikit.py`, `viz/utils.py`) using deterministic mock data.
2. Smoke tests for tokenizer modules to ensure expected token shapes per configuration.

Progressively implementing these phases will extend coverage from the current action-head and trajectory-transform focus to the broader model, data, and utility layers, reducing regression risk throughout the stack.
