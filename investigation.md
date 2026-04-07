# CrossFormer Investigation Report

## Scope and objective

This investigation reviewed the full repository with focus on:

- The main training loop and call chain.
- Data loading and preprocessing paths (Grain primary, legacy TF/RLDS secondary).
- Bugs, reliability risks, performance issues, and cleanup opportunities.

The goal is to surface high-signal issues and provide practical remediation steps.

## Executive summary

The codebase is functional but currently fragile in a few critical areas, especially around training/data consistency and pipeline correctness. The most important risks are:

- Train/val data path mismatch (validation may run on effectively train-shaped data).
- Reproducibility drift in action masking (unseeded per-sample RNG in Grain embody transform).
- Checkpoint step/accounting mismatch between loop counter and `TrainState.step`.
- Dead/unreachable filter logic in Grain pipeline and duplicated pipeline code paths increasing drift risk.

Addressing the high-severity items first will materially improve training reliability and reproducibility.

## Main call graph traces

### Training loop (primary)

Entry point:

- `scripts/train/xflow.py:337` (`main(cfg)`).

High-level flow:

1. Device/mesh/sharding setup (`xflow.py:339-343`).
2. Grain data creation (`xflow.py:367-371`).
3. Example batch introspection + key resolution (`xflow.py:372-387`).
4. Model config/build (`xflow.py:388-438`).
5. Optimizer + `TrainState` + compiled step function (`xflow.py:446-467`).
6. Callback wiring (save/viz/val callbacks) (`xflow.py:468-497`).
7. Core loop over `range(cfg.steps)` (`xflow.py:509-643`):
   - Pull batch.
   - Normalize obs.
   - Apply lowdim/guidance dropout.
   - Extract bundled actions.
   - Run `train_step`.
   - Periodic logging + prediction/viz.
   - Periodic val callback.
   - Periodic checkpoint save.
8. Final save + summary (`xflow.py:644-663`).

Train step implementation:

- `crossformer/run/train_step.py:29-99`.
- JIT step computes loss via `bound.heads["xflow"].loss(...)`, computes grads, applies updates, increments `TrainState.step` via `apply_gradients` (`crossformer/utils/train_utils.py:50-59`).

### Data loading (Grain primary path)

Factory entry:

- `crossformer/data/grain/loader.py:317-391` (`GrainDataFactory.make`).

High-level flow:

1. Build source configs (`loader.py:333-334`).
2. Compute `max_a` across embodiments (`loader.py:335-338`).
3. Build dataset(s) through `source2ds` / `make_single_dataset` (`loader.py:271-284`, `194-254`).
4. Optional embody transform (`loader.py:279-282`).
5. Mix/pad if multi-dataset (`loader.py:286-315`).
6. Seed/shuffle/repeat, map/add_mask, iter conversion, batch, mp prefetch (`loader.py:351-366`).
7. Thread prefetch, `np2jax`, optional shard, frame transforms, compatibility map (`loader.py:376-387`).

Trajectory build/normalization:

- `crossformer/data/grain/builders.py:200-238`:
  - Restructure samples.
  - Compute/load stats.
  - Apply normalization.

Trajectory/frame transforms:

- `crossformer/data/grain/pipelines.py:85-145` and `147-205`.

Legacy TF pipeline (still present):

- `crossformer/data/dataset.py` with richer transform implementations (`apply_trajectory_transforms`, `apply_frame_transforms`, and RLDS dataset creation).

## Findings and recommendations

### High severity

#### H1 — Validation dataset is built through train-only source path

Evidence:

- `xflow.py` requests val loader with `train=False` (`scripts/train/xflow.py:370`).
- But `GrainDataFactory.source2ds(...)` hardcodes `train=True` when invoking `make_single_dataset` (`crossformer/data/grain/loader.py:274`).
- Also `make_single_dataset(...)` currently ignores its `train` argument for trajectory transforms and always calls `apply_trajectory_transforms(ds, seed=seed, config=config)` without train-sensitive kwargs (`loader.py:240`).

Risk:

- Validation data may inherit train-time behavior and not reflect true eval conditions.
- Metrics can be optimistic/noisy and regressions can be masked.

Remediation:

1. Thread `train` from `GrainDataFactory.make(..., train=...)` into `source2ds` and down into `make_single_dataset`.
2. Ensure trajectory/frame transform choices honor `train` consistently.
3. Add a unit/integration test asserting train and val pipelines differ where intended (e.g., repeat/shuffle/augment behavior).

---

#### H2 — Unseeded RNG in embody transform breaks reproducibility

Evidence:

- `embody_transform` creates a fresh RNG with `np.random.default_rng()` per sample (`crossformer/data/grain/embody.py:239`).
- This RNG is not tied to global seed, step, worker id, or sample id.

Risk:

- Non-deterministic action masking/order across runs even with fixed config/seed.
- Harder debugging and benchmarking; potential train/val distribution drift.

Remediation:

1. Inject deterministic randomness via Grain RNG plumbing (sample-level seed input).
2. Derive per-sample RNG from stable ids (episode/step) + global seed if needed.
3. Add reproducibility test: two runs with same seed produce identical `act.id`/`mask.act` for fixed input.

---

#### H3 — Training loop step counter and state step can diverge semantically

Evidence:

- Loop uses `for step in range(cfg.steps)` and logs/saves with loop index (`scripts/train/xflow.py:509`, `625`, `640-643`).
- Optimizer/lr scheduling inside train step uses `state.step` (`crossformer/run/train_step.py:68`, `92`) and increments there.

Risk:

- Step references can diverge after resume or nonstandard loop control, creating ambiguous checkpoint/log alignment.

Remediation:

1. Use `int(state.step)` as the canonical step for logging/checkpoint intervals.
2. Keep loop index only for local iteration control.
3. Add test for resumed training ensuring `wandb step`, lr schedule step, and checkpoint step remain aligned.

---

#### H4 — Dead code in language-present filter indicates broken intent

Evidence:

- `_filter_language_present` returns early (`return language is not None`) and subsequent checks are unreachable (`crossformer/data/grain/pipelines.py:62-67`).

Risk:

- Intended non-empty language filtering is not happening.
- Data quality assumptions for language-conditioned training may be violated.

Remediation:

1. Remove dead code and implement intended semantics explicitly:
   - key exists, and
   - non-empty where required.
2. Add test coverage for empty-string, missing key, and valid language cases.

### Medium severity

#### M1 — Duplicate Grain pipeline implementations increase drift risk

Evidence:

- Similar/duplicated functionality exists in both:
  - `crossformer/data/grain/loader.py`
  - `crossformer/data/grain/pipelines.py` (contains another `make_single_dataset`, similar TODOs and flow).

Risk:

- Fixes can land in one path and miss the other.
- Behavioral drift and debugging overhead increase.

Remediation:

1. Choose one canonical pipeline implementation.
2. Move shared logic into small reusable helpers.
3. Decommission duplicate path with deprecation window and tests.

---

#### M2 — Multi-dataset frame transform config is taken from first source only

Evidence:

- In `GrainDataFactory.make`, frame transform config is selected from `sources[0]` regardless of dataset mix heterogeneity (`loader.py:383-386`).

Risk:

- Mixed datasets can be transformed with mismatched assumptions.

Remediation:

1. Merge frame transform requirements across all active sources.
2. Validate compatibility at startup and fail fast on incompatible mixes.

---

#### M3 — Broad use of hard-coded constants and TODO placeholders in hot path

Evidence:

- Examples:
  - `embody_transform(... mask_prob=0.25)` in loader (`loader.py:280`).
  - TODO warnings in production path (`loader.py:182`, `210`; `builders.py:235`; `pipelines.py:163-164`).

Risk:

- Important behavior is implicit and not config-driven.
- Maintenance burden and uncertainty remain high.

Remediation:

1. Move key constants to config.
2. Convert TODO/warnings in runtime paths into issues or tracked tasks.
3. Remove stale TODOs once addressed.

---

#### M4 — `np.bool` usage is legacy and brittle

Evidence:

- `np.bool` appears in `builders.note_embodiment` (`crossformer/data/grain/builders.py:146`).

Risk:

- Compatibility pitfalls across NumPy versions/tooling; avoid deprecated aliases.

Remediation:

1. Replace `np.bool` with `np.bool_` or builtin `bool`.
2. Add small test that covers embodiment-note shape/dtype expectations.

### Low severity / cleanup

#### L1 — Unused helper likely dead code

Evidence:

- `stable_hash_int` defined but not referenced elsewhere (`crossformer/data/grain/builders.py:94`, no other call sites found).

Remediation:

- Remove if truly unused, or integrate where intended with tests.

---

#### L2 — Test suite contains large `tests/broken/*` surface

Evidence:

- Multiple broken tests are parked under `tests/broken/`.

Risk:

- Reduced confidence in broad regressions; unclear health of older components.

Remediation:

1. Triage broken tests into:
   - obsolete/remove,
   - quick-fix/restore,
   - long-term backlog.
2. Promote restored tests into normal CI markers.

---

#### L3 — Roadmap artifacts include unrelated/placeholder items

Evidence:

- `roadmap/opencode_mcp.md` appears unrelated to CrossFormer training/data concerns.

Risk:

- Signal-to-noise loss for maintainers.

Remediation:

- Prune or relocate unrelated roadmap items.

## Suggested remediation plan (priority order)

### Phase 1 (stability/reproducibility first)

1. Fix train/val pipeline propagation (`train` flag threading end-to-end).
2. Seed embody randomness deterministically.
3. Align canonical step source to `state.step`.
4. Fix `_filter_language_present` dead/unreachable logic.

### Phase 2 (maintainability/perf)

1. Consolidate duplicate Grain pipeline paths.
2. Remove first-source-only assumptions in mixed datasets.
3. Replace legacy dtype aliases; remove dead code.

### Phase 3 (confidence)

1. Add tests for train-vs-val behavior, reproducibility, and filtering semantics.
2. Triage and recover selected `tests/broken/*` coverage.

## Potential quick wins

- Replace `np.bool` alias now (small, safe).
- Fix unreachable lines in `_filter_language_present`.
- Introduce a single helper for canonical training step value used by logging/checkpointing.

## Notes on uncertainty

Some risks (especially train/val parity and reproducibility) are strongly indicated by static inspection, but should be confirmed with short deterministic runtime checks once environment constraints allow full dependency sync.

---

Prepared from repository inspection with emphasis on:

- `scripts/train/xflow.py`
- `crossformer/run/train_step.py`
- `crossformer/utils/train_utils.py`
- `crossformer/utils/callbacks/save.py`
- `crossformer/data/grain/{loader.py,builders.py,pipelines.py,embody.py}`
- `crossformer/data/dataset.py`
- tests and roadmap context files
