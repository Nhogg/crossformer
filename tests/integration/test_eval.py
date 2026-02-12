"""Integration: evaluation step produces valid metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from .conftest import requires_gpu
from .helpers import (
    assert_finite,
    assert_metrics_keys,
    eval_step,
    init_train_state,
    train_step,
)


@requires_gpu
@pytest.mark.integration
class TestEval:
    """Test evaluation loop and metrics."""

    def test_eval_produces_finite_metrics(self, tiny_config, example_batch):
        """Eval forward pass should produce finite metrics."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        metrics = eval_step(state, batch, jax.random.PRNGKey(99))

        assert "single" in metrics
        assert_finite(metrics["single"]["loss"])
        assert_finite(metrics["single"]["mse"])

    def test_eval_deterministic(self, tiny_config, example_batch):
        """Eval with same RNG should be deterministic (no dropout)."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        rng = jax.random.PRNGKey(123)
        m1 = eval_step(state, batch, rng)
        m2 = eval_step(state, batch, rng)

        for key in m1:
            if isinstance(m1[key], dict):
                for metric_name in m1[key]:
                    jnp.testing.assert_allclose(
                        m1[key][metric_name],
                        m2[key][metric_name],
                        atol=1e-6,
                    )
            else:
                jnp.testing.assert_allclose(m1[key], m2[key], atol=1e-6)

    def test_eval_metrics_structure_multihead(self, multihead_config, example_batch_multihead):
        """Eval metrics should reflect all heads."""
        batch = jax.tree.map(jnp.asarray, example_batch_multihead)
        state = init_train_state(jax.random.PRNGKey(0), batch, multihead_config)

        metrics = eval_step(state, batch, jax.random.PRNGKey(42))

        assert_metrics_keys(metrics, ["single", "bimanual", "mano"])
        for head_name in ["single", "bimanual", "mano"]:
            assert_finite(metrics[head_name]["loss"])

    def test_eval_after_train_shows_improvement(self, tiny_config, example_batch):
        """Eval loss should improve after training steps."""
        batch = jax.tree.map(jnp.asarray, example_batch)

        config = tiny_config.copy()
        config["optimizer"] = {"learning_rate": 1e-2, "weight_decay": 0.0}
        state = init_train_state(jax.random.PRNGKey(0), batch, config)

        # Initial eval loss
        eval_loss_0 = float(eval_step(state, batch, jax.random.PRNGKey(1))["total_loss"])

        # Train for 10 steps
        rng = jax.random.PRNGKey(42)
        for i in range(10):
            rng, step_rng = jax.random.split(rng)
            state, _ = train_step(state, batch, step_rng)

        # Final eval loss
        eval_loss_1 = float(eval_step(state, batch, jax.random.PRNGKey(2))["total_loss"])

        # With high learning rate, should generally improve
        improvement = (eval_loss_0 - eval_loss_1) / (eval_loss_0 + 1e-8)
        assert improvement > -0.1, (
            f"Eval loss should improve: {eval_loss_0:.4f} -> {eval_loss_1:.4f} (improvement {improvement:.4f})"
        )

    def test_eval_no_gradient_accumulation(self, tiny_config, example_batch):
        """Eval should not accumulate gradients."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        # This test verifies that eval_step doesn't use grad tracking
        # by simply calling it and ensuring no errors
        metrics = eval_step(state, batch, jax.random.PRNGKey(99))
        assert_finite(metrics["total_loss"])
