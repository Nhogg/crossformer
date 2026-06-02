from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING

from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
from rich.table import Table
from tips.scenic.utils import checkpoint as tips_checkpoint

from crossformer.model.dream import DreamTIPS, DreamVGG
from crossformer.model.load import resolve_checkpoint_path

if TYPE_CHECKING:
    from .config import Config


def net_out_size(cfg: Config) -> tuple[int, int]:
    h, w = cfg.net_in_size
    if cfg.variant == "full":
        return h, w
    if cfg.variant == "half":
        return h // 2, w // 2
    if cfg.variant == "quarter":
        return h // 4, w // 4
    raise ValueError(f"unknown variant: {cfg.variant}")


def make_model(cfg: Config, num_keypoints: int):
    if cfg.encoder == "vgg":
        return DreamVGG(
            num_keypoints=num_keypoints,
            variant=cfg.variant,
            decoder=cfg.decoder,
            deconv_decoder=cfg.deconv_decoder,
            full_output=cfg.full_output,
            skip_connections=cfg.skip_connections,
            n_stages=cfg.n_stages,
            internalize_spatial_softmax=cfg.internalize_spatial_softmax,
            learned_beta=cfg.learned_beta,
            initial_beta=cfg.initial_beta,
        )
    if cfg.encoder == "tips":
        if cfg.n_stages != 1:
            raise ValueError("TIPS encoder currently supports n_stages=1")
        if cfg.internalize_spatial_softmax:
            raise ValueError("TIPS encoder does not support internalize_spatial_softmax")
        return DreamTIPS(
            num_keypoints=num_keypoints,
            variant=cfg.variant,
            decoder=cfg.decoder,
            tips_variant=cfg.tips_variant,
            freeze_encoder=not cfg.tips_trainable,
        )
    raise ValueError(f"unknown encoder: {cfg.encoder}")


def load_tips_params(cfg: Config, params):
    if cfg.encoder != "tips":
        return params
    ckpt_path = resolve_checkpoint_path(cfg.tips_variant, cfg.tips_checkpoint)
    p = unfreeze(params)
    p["tips"] = tips_checkpoint.load_checkpoint(ckpt_path, p["tips"])
    print(f"  tips_checkpoint: {ckpt_path}")
    return freeze(p)


def frozen_keys(cfg: Config) -> tuple[str, ...]:
    if cfg.encoder == "tips" and not cfg.tips_trainable:
        return ("tips.*",)
    return ()


def _image_to_float(image: jax.Array) -> jax.Array:
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def _count_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def _count_trainable_params(params, frozen: tuple[str, ...]) -> int:
    flat = freeze(params).unfreeze() if hasattr(params, "unfreeze") else params
    flat = jax.tree_util.tree_flatten_with_path(flat)[0]
    n = 0
    for path, x in flat:
        key = ".".join(str(p.key) for p in path)
        if any(fnmatch(key, pattern) for pattern in frozen):
            continue
        n += x.size
    return n


def _print_shapes(shapes):
    table = Table("stage", "shape")
    for name, shape in shapes:
        table.add_row(name, str(shape))
    print(table)
