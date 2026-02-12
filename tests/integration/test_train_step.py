"""Integration: end-to-end train step with real model, optimizer, and gradients."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax
import pytest

from crossformer.utils.train_utils import TrainState

from .conftest import requires_gpu


def _loss_fn(params, batch, rng, module, train=True):
    """Minimal loss function matching finetune.py structure."""
    bound = module.bind({"params": params}, rngs={"dropout": rng})
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


@requires_gpu
@pytest.mark.integration
class TestTrainStep:
    def test_single_train_step(self, model_and_batch):
        """One grad step should produce finite gradients and update params."""
        model, batch = model_and_batch

        tx = optax.adamw(learning_rate=1e-4)
        state = TrainState.create(rng=jax.random.PRNGKey(0), model=model, tx=tx)

        rng, dropout_rng = jax.random.split(state.rng)
        loss_fn = partial(_loss_fn, module=model.module)
        (loss, _info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng)

        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        grad_leaves = jax.tree.leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves), "Gradients contain non-finite values"

        new_state = state.apply_gradients(grads=grads, rng=rng)
        assert new_state.step == 1

    def test_train_20_steps_loss_finite(self, model_and_batch):
        """Run 20 train steps; loss should stay finite throughout."""
        model, batch = model_and_batch

        tx = optax.adamw(learning_rate=1e-4)
        state = TrainState.create(rng=jax.random.PRNGKey(42), model=model, tx=tx)

        loss_fn = partial(_loss_fn, module=model.module)

        @jax.jit
        def step(state, batch):
            rng, dropout_rng = jax.random.split(state.rng)
            (loss, _info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng)
            new_state = state.apply_gradients(grads=grads, rng=rng)
            return new_state, loss

        losses = []
        for _ in range(20):
            state, loss = step(state, batch)
            losses.append(float(loss))

        assert all(jnp.isfinite(l) for l in losses), f"Non-finite loss found in: {losses}"
        assert state.step == 20

    def test_optimizer_updates_weights(self, model_and_batch):
        """After a gradient step, at least some parameters should change."""
        model, batch = model_and_batch

        tx = optax.adamw(learning_rate=1e-3)
        state = TrainState.create(rng=jax.random.PRNGKey(0), model=model, tx=tx)

        loss_fn = partial(_loss_fn, module=model.module)
        rng, dropout_rng = jax.random.split(state.rng)
        (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng)
        new_state = state.apply_gradients(grads=grads, rng=rng)

        old_flat = jax.tree.leaves(state.model.params)
        new_flat = jax.tree.leaves(new_state.model.params)
        changed = sum(not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat))
        assert changed > 0, "No parameters were updated after a gradient step"
