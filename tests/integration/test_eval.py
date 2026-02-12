"""Integration: evaluation step produces valid metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from .conftest import HEAD_NAME, requires_gpu


def _eval_fn(params, batch, rng, module):
    """Forward pass in eval mode — no dropout."""
    bound = module.bind({"params": params}, rngs={"dropout": rng})
    embeddings = bound.crossformer_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=False,
    )

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
        metrics[head_name] = head_metrics

    return metrics


@requires_gpu
@pytest.mark.integration
class TestEval:
    def test_eval_metrics_finite(self, model_and_batch):
        """Eval forward pass should produce finite metric values."""
        model, batch = model_and_batch
        rng = jax.random.PRNGKey(99)

        metrics = _eval_fn(model.params, batch, rng, model.module)

        assert HEAD_NAME in metrics, f"Expected head '{HEAD_NAME}' in metrics"
        head_metrics = metrics[HEAD_NAME]
        assert "loss" in head_metrics
        assert "mse" in head_metrics
        assert jnp.isfinite(head_metrics["loss"])
        assert jnp.isfinite(head_metrics["mse"])

    def test_eval_deterministic(self, model_and_batch):
        """Eval (no dropout) should be deterministic across calls."""
        model, batch = model_and_batch
        rng = jax.random.PRNGKey(123)

        m1 = _eval_fn(model.params, batch, rng, model.module)
        m2 = _eval_fn(model.params, batch, rng, model.module)

        for key in m1:
            for metric_name in m1[key]:
                jnp.testing.assert_allclose(
                    m1[key][metric_name],
                    m2[key][metric_name],
                    atol=1e-6,
                )
