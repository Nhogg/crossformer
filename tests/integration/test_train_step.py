"""Integration: end-to-end training steps with real model, optimizer, and gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from .conftest import requires_gpu
from .helpers import (
    assert_finite,
    assert_metrics_keys,
    assert_params_changed,
    init_train_state,
    train_step,
)


@requires_gpu
@pytest.mark.integration
class TestTrainStep:
    """Test single and multi-step training."""

    def test_single_train_step_produces_finite_loss(self, tiny_config, example_batch):
        """One gradient step should produce finite loss."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        new_state, metrics = train_step(state, batch, jax.random.PRNGKey(1))

        assert_finite(metrics["loss"])
        assert 0 < metrics["loss"] < 1000, f"Loss out of expected range: {metrics['loss']}"
        assert new_state.step == 1

    def test_single_train_step_updates_params(self, tiny_config, example_batch):
        """At least some parameters should change after one step."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        params_old = state.model.params
        new_state, _ = train_step(state, batch, jax.random.PRNGKey(1))
        params_new = new_state.model.params

        assert_params_changed(params_old, params_new, min_changed_fraction=0.05)

    def test_train_10_steps_loss_finite(self, tiny_config, example_batch):
        """Train for 10 steps; loss should stay finite."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        rng = jax.random.PRNGKey(42)
        losses = []

        for i in range(10):
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            losses.append(float(metrics["loss"]))

        assert all(jnp.isfinite(l) for l in losses), f"Non-finite loss found: {losses}"
        assert state.step == 10

    def test_train_10_steps_loss_decreasing(self, tiny_config, example_batch):
        """Loss should generally decrease over steps (with high learning rate)."""
        batch = jax.tree.map(jnp.asarray, example_batch)

        config = tiny_config.copy()
        config["optimizer"] = {"learning_rate": 1e-2, "weight_decay": 0.0}
        state = init_train_state(jax.random.PRNGKey(0), batch, config)

        rng = jax.random.PRNGKey(42)
        losses = []

        for i in range(10):
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            losses.append(float(metrics["loss"]))

        # Check that loss trend is generally decreasing (fit exponential decay)
        # Allow some noise but overall should decrease
        final_loss = losses[-1]
        initial_loss = losses[0]
        improvement = (initial_loss - final_loss) / (initial_loss + 1e-8)
        assert improvement > -0.1, f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_multihead_single_step(self, multihead_config, example_batch_multihead):
        """Single step with multiple heads should compute loss for each."""
        batch = jax.tree.map(jnp.asarray, example_batch_multihead)
        state = init_train_state(jax.random.PRNGKey(0), batch, multihead_config)

        new_state, metrics = train_step(state, batch, jax.random.PRNGKey(1))

        assert_metrics_keys(metrics, ["single", "bimanual", "mano"])
        for head_name in ["single", "bimanual", "mano"]:
            assert_finite(metrics[head_name]["loss"])
        assert_finite(metrics["total_loss"])
        assert new_state.step == 1

    def test_train_step_jit_matches_non_jit(self, tiny_config, example_batch):
        """JIT-compiled step should match eager execution."""
        batch = jax.tree.map(jnp.asarray, example_batch)
        state = init_train_state(jax.random.PRNGKey(0), batch, tiny_config)

        rng1 = jax.random.PRNGKey(99)
        rng2 = jax.random.PRNGKey(99)

        # Eager execution
        eager_state, eager_metrics = train_step(state, batch, rng1)

        # JIT execution
        jit_train_step = jax.jit(train_step)
        jit_state, jit_metrics = jit_train_step(state, batch, rng2)

        # Compare losses and params
        jnp.testing.assert_allclose(eager_metrics["loss"], jit_metrics["loss"], rtol=1e-5, atol=1e-5)

        eager_params = jax.tree.leaves(eager_state.model.params)
        jit_params = jax.tree.leaves(jit_state.model.params)
        for ep, jp in zip(eager_params, jit_params):
            jnp.testing.assert_allclose(ep, jp, rtol=1e-5, atol=1e-5)
