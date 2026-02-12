"""Integration test helpers: pure training functions and assertion utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp

from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.mytyping import Data, PRNGKey
from crossformer.utils.train_utils import create_optimizer, TrainState

# ---- Training State and Steps ----


def init_train_state(
    rng: PRNGKey,
    example_batch: Data,
    config: dict,
    text_processor: Any = None,
    dataset_statistics: Any = None,
    optimizer_kwargs: dict | None = None,
) -> TrainState:
    """Initialize TrainState from scratch.

    Args:
        rng: JAX random key
        example_batch: Example batch for model initialization
        config: Model config dict (contains model and optimizer specs)
        text_processor: Optional text processor for language embeddings
        dataset_statistics: Optional dataset statistics for normalization
        optimizer_kwargs: Optional overrides for optimizer creation

    Returns:
        TrainState with initialized model, optimizer, and RNG.
    """
    model = CrossFormerModel.from_config(
        config,
        example_batch=example_batch,
        text_processor=text_processor,
        rng=rng,
        dataset_statistics=dataset_statistics,
    )

    optimizer_cfg = config.get("optimizer", {})
    if optimizer_kwargs:
        optimizer_cfg = {**optimizer_cfg, **optimizer_kwargs}

    tx, _, _ = create_optimizer(model.params, **optimizer_cfg)
    return TrainState.create(model=model, tx=tx, rng=rng)


def train_step(
    state: TrainState,
    batch: Data,
    rng: PRNGKey,
) -> tuple[TrainState, dict[str, Any]]:
    """Single training step: forward, loss, backward, update.

    Args:
        state: Current training state
        batch: Input batch (observation, task, action, masks)
        rng: RNG for dropout

    Returns:
        (new_state, metrics_dict)
    """

    def loss_fn(params, batch, dropout_rng, train=True):
        bound = state.model.module.bind({"params": params}, rngs={"dropout": dropout_rng})
        embeddings = bound.crossformer_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        total_loss = jnp.float32(0.0)
        metrics = {}
        for head_name, head in bound.heads.items():
            head_loss, head_metrics = head.loss(
                embeddings,
                batch["action"][head_name],
                batch["observation"]["timestep_pad_mask"],
                action_pad_mask=jnp.ones_like(batch["action"][head_name], dtype=jnp.bool_),
                action_head_mask=batch["embodiment"][head_name],
                train=train,
            )
            total_loss = total_loss + head_loss
            metrics[head_name] = head_metrics

        metrics["total_loss"] = total_loss
        return total_loss, metrics

    rng_dropout, rng_next = jax.random.split(rng)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.model.params, batch, rng_dropout, train=True
    )

    new_state = state.apply_gradients(grads=grads, rng=rng_next)
    return new_state, {"loss": loss, **metrics}


def eval_step(
    state: TrainState,
    batch: Data,
    rng: PRNGKey,
) -> dict[str, Any]:
    """Evaluation forward pass (no gradient update).

    Args:
        state: Training state
        batch: Input batch
        rng: RNG for dropout (though not used in eval mode)

    Returns:
        metrics_dict with loss and per-head metrics
    """
    bound = state.model.module.bind({"params": state.model.params}, rngs={"dropout": rng})
    embeddings = bound.crossformer_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=False,
    )

    total_loss = jnp.float32(0.0)
    metrics = {}
    for head_name, head in bound.heads.items():
        _, head_metrics = head.loss(
            embeddings,
            batch["action"][head_name],
            batch["observation"]["timestep_pad_mask"],
            action_pad_mask=jnp.ones_like(batch["action"][head_name], dtype=jnp.bool_),
            action_head_mask=batch["embodiment"][head_name],
            train=False,
        )
        total_loss = total_loss + head_metrics.get("loss", jnp.float32(0.0))
        metrics[head_name] = head_metrics

    metrics["total_loss"] = total_loss
    return metrics


# ---- Assertion Helpers ----


def assert_finite(x: Any, name: str = "") -> None:
    """Assert all array values in tree are finite.

    Args:
        x: Tree of arrays
        name: Optional name for error message
    """
    leaves = jax.tree.leaves(x)
    for i, leaf in enumerate(leaves):
        assert jnp.all(jnp.isfinite(leaf)), f"Non-finite values in {name} leaf {i}: {leaf}"


def assert_loss_decreased(
    loss_0: float,
    loss_1: float,
    rtol: float = 0.01,
) -> None:
    """Assert that loss decreased by at least rtol after training step.

    Args:
        loss_0: Initial loss
        loss_1: Loss after step
        rtol: Relative tolerance threshold (default 1% improvement)
    """
    improvement = (loss_0 - loss_1) / (jnp.abs(loss_0) + 1e-8)
    assert improvement >= -rtol, (
        f"Loss did not decrease enough: {loss_0:.6f} -> {loss_1:.6f} (improvement {improvement:.4f}, threshold {rtol})"
    )


def assert_params_changed(
    params_old: dict,
    params_new: dict,
    min_changed_fraction: float = 0.1,
) -> None:
    """Assert that at least min_changed_fraction of params changed.

    Args:
        params_old: Old params dict
        params_new: New params dict
        min_changed_fraction: Minimum fraction of params that must change
    """
    old_leaves = jax.tree.leaves(params_old)
    new_leaves = jax.tree.leaves(params_new)
    changed = sum(not jnp.allclose(o, n, atol=1e-8) for o, n in zip(old_leaves, new_leaves))
    total = len(old_leaves)
    frac = changed / total if total > 0 else 0
    assert frac >= min_changed_fraction, f"Only {frac:.2%} of params changed (threshold {min_changed_fraction:.2%})"


def assert_metrics_keys(
    metrics: dict,
    expected_heads: list[str],
) -> None:
    """Assert metrics dict has expected structure.

    Args:
        metrics: Metrics dict from loss computation
        expected_heads: List of expected head names
    """
    for head_name in expected_heads:
        assert head_name in metrics, f"Head '{head_name}' not in metrics: {metrics.keys()}"
        head_metrics = metrics[head_name]
        for key in ["loss", "mse", "lsign"]:
            assert key in head_metrics, f"Key '{key}' not in metrics['{head_name}']: {head_metrics.keys()}"


# ---- Device Management ----


def has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB (approximate via JAX device count)."""
    try:
        gpus = jax.devices("gpu")
        if gpus:
            # Rough estimate: typical GPU is ~8-24 GB
            return float(len(gpus) * 16)
        return 0.0
    except RuntimeError:
        return 0.0


@contextmanager
def require_gpu(reason: str = ""):
    """Context that checks GPU availability and logs reason if unavailable."""
    if not has_gpu():
        msg = f"GPU required: {reason}" if reason else "GPU required"
        raise RuntimeError(msg)
    yield
