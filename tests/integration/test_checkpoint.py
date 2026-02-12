"""Integration: checkpoint save/load round-trip."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from crossformer.model.crossformer_module import CrossFormerModule
from crossformer.utils.train_utils import TrainState

from .conftest import (
    requires_gpu,
)


def _save_checkpoint(state: TrainState, path: Path, config: dict, step: int):
    """Minimal save that mirrors SaveCallback + save_pretrained."""
    path.mkdir(parents=True, exist_ok=True)

    # save params via orbax
    mngr = ocp.CheckpointManager(path, ocp.StandardCheckpointer())
    mngr.save(step, args=ocp.args.StandardSave(item=state.model.params))
    mngr.wait_until_finished()

    # save config
    with open(path / "config.json", "w") as f:
        json.dump(config, f)

    # save example batch
    with open(path / "example_batch.msgpack", "wb") as f:
        f.write(flax.serialization.msgpack_serialize(state.model.example_batch))

    # save dataset statistics
    with open(path / "dataset_statistics.json", "w") as f:
        json.dump({}, f)


@requires_gpu
@pytest.mark.integration
class TestCheckpoint:
    def test_params_roundtrip(self, model_and_batch, tiny_config, tmp_path):
        """Save params, reload, and verify outputs match."""
        model, batch = model_and_batch

        tx = optax.adamw(learning_rate=1e-4)
        state = TrainState.create(rng=jax.random.PRNGKey(0), model=model, tx=tx)

        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(state, ckpt_dir, tiny_config, step=0)

        # reload params
        module = CrossFormerModule.create(**tiny_config["model"])
        init_args = (
            state.model.example_batch["observation"],
            state.model.example_batch["task"],
            state.model.example_batch["observation"]["timestep_pad_mask"],
        )
        params_shape = jax.eval_shape(partial(module.init, train=False), jax.random.PRNGKey(0), *init_args)["params"]
        target = jax.tree.map(jnp.zeros_like, params_shape)
        abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, target)

        mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer())
        restored_params = mngr.restore(0, args=ocp.args.StandardRestore(abstract))

        # compare outputs
        bound_orig = model.module.bind({"params": state.model.params})
        bound_restored = model.module.bind({"params": restored_params})

        obs = batch["observation"]
        task = batch["task"]
        mask = obs["timestep_pad_mask"]

        out_orig = bound_orig.crossformer_transformer(obs, task, mask, train=False)
        out_restored = bound_restored.crossformer_transformer(obs, task, mask, train=False)

        for key in out_orig:
            np.testing.assert_allclose(
                np.array(out_orig[key].tokens),
                np.array(out_restored[key].tokens),
                atol=1e-5,
                err_msg=f"Mismatch in transformer output '{key}' after checkpoint round-trip",
            )
