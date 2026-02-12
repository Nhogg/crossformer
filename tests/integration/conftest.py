"""Shared fixtures for integration tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.model.components.action_heads import FlowMatchingActionHead
from crossformer.model.components.tokenizers import ImageTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26FILM
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.spec import ModuleSpec


def _has_gpu() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


requires_gpu = pytest.mark.skipif(
    not _has_gpu(),
    reason="Integration tests require a GPU",
)

# ---- tiny model config ----

IMG_SIZE = 64
BATCH = 2
WINDOW = 1
ACTION_DIM = 7
ACTION_HORIZON = 4
HEAD_NAME = "single"


def make_tiny_config() -> dict:
    """Build a minimal CrossFormer config dict suitable for fast integration tests."""
    token_embedding_size, transformer_kwargs = common_transformer_sizes("dummy")

    encoder = ModuleSpec.create(ResNet26FILM)
    observation_tokenizers = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            task_film_keys=["language_instruction"],
            encoder=encoder,
        ),
    }
    heads = {
        HEAD_NAME: ModuleSpec.create(
            FlowMatchingActionHead,
            action_horizon=ACTION_HORIZON,
            action_dim=ACTION_DIM,
            readout_key=f"readout_{HEAD_NAME}",
            pool_strategy="use_map",
            clip_pred=False,
            loss_weight=1.0,
            constrain_loss_dims=True,
            num_preds=0,
            flow_steps=4,
        ),
    }
    readouts = {HEAD_NAME: ACTION_HORIZON}

    return {
        "model": {
            "observation_tokenizers": observation_tokenizers,
            "task_tokenizers": {},
            "heads": heads,
            "readouts": readouts,
            "token_embedding_size": token_embedding_size,
            "transformer_kwargs": transformer_kwargs,
            "max_horizon": WINDOW,
        },
    }


def make_fake_batch(rng: np.random.Generator | None = None) -> dict:
    """Generate a synthetic batch matching the tiny config."""
    if rng is None:
        rng = np.random.default_rng(0)

    obs = {
        "image_primary": rng.integers(0, 255, (BATCH, WINDOW, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
        "timestep_pad_mask": np.ones((BATCH, WINDOW), dtype=bool),
        "pad_mask_dict": {
            "image_primary": np.ones((BATCH, WINDOW), dtype=bool),
        },
    }
    task = {
        "image_primary": rng.integers(0, 255, (BATCH, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
        "pad_mask_dict": {
            "image_primary": np.ones((BATCH,), dtype=bool),
            "language_instruction": np.zeros((BATCH,), dtype=bool),
        },
        "language_instruction": np.zeros((BATCH, 512), dtype=np.float32),
    }
    action = rng.normal(size=(BATCH, WINDOW, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    action_pad_mask = np.ones((BATCH, WINDOW, ACTION_HORIZON, ACTION_DIM), dtype=bool)

    return {
        "observation": obs,
        "task": task,
        "action": {HEAD_NAME: action},
        "action_pad_mask": action_pad_mask,
        "action_head_masks": {HEAD_NAME: np.ones(BATCH, dtype=bool)},
        "embodiment": {HEAD_NAME: np.ones(BATCH, dtype=bool)},
    }


@pytest.fixture(scope="module")
def tiny_config():
    return make_tiny_config()


@pytest.fixture(scope="module")
def example_batch():
    return make_fake_batch()


@pytest.fixture(scope="module")
def model_and_batch(tiny_config, example_batch):
    """Init model and return (model, batch) — expensive, so module-scoped."""
    batch = jax.tree.map(jnp.asarray, example_batch)
    model = CrossFormerModel.from_config(
        tiny_config,
        example_batch=batch,
        text_processor=None,
        rng=jax.random.PRNGKey(0),
        dataset_statistics=None,
    )
    return model, batch
