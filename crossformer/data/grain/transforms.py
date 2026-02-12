"""Trajectory and frame level transforms implemented for Google Grain.

The original TensorFlow pipeline exposes a fairly rich set of transforms that
operate on either complete trajectories or on frame level chunks.  Rewriting
every single transform for NumPy/JAX would be unnecessary for the initial
Grain migration, however the core operations - chunking, padding, head masks,
and simple subsampling - are required for parity.  This module provides these
rewritten utilities in a framework agnostic way.

All functions operate purely on nested dictionaries containing NumPy arrays or
values that are trivially convertible to NumPy arrays.  They are therefore
compatible with both ``grain.MapDataset`` and ``grain.IterDataset`` pipelines
without introducing TensorFlow as a dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import logging
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np
from scipy import ndimage

from crossformer.data.oxe import HEAD_TO_DATASET
from crossformer.utils.jax_utils import cpu, with_device
from crossformer.utils.mytyping import PRNGKey
from crossformer.utils.tree import merge

log = logging.getLogger(__name__)

ArrayDict = dict[str, Any]
Step = dict[str, Any]
Trajectory = dict[str, Step]


_INTERPOLATION_TO_ORDER = {"nearest": 0, "bilinear": 1}


@with_device(cpu)
def _mask(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def add_pad_mask_dict(step: Step, device=cpu) -> Step:
    """Annotates ``step`` with padding masks for observation/task dictionaries."""

    # every item says if mask at that time
    n = step["observation"]["timestep"].shape
    n = n[0] if n else 1

    def maybe_full(x):
        if x.shape and x.shape[0] == n:
            return x.astype(bool)
        return np.full((n,), x).astype(bool).reshape(-1)

    # is_mask = lambda x: ~ utils.is_padding(x)
    def is_mask(*args):
        return np.zeros((), dtype=bool)  # no native strings

    pad_masks = jax.tree.map(is_mask, step)
    pad_masks = jax.tree.map(maybe_full, pad_masks)

    step["observation"]["pad_mask_dict"] = pad_masks.get("observation", {})
    step["task"]["pad_mask_dict"] = pad_masks.get("task", {})
    return step


def add_head_action_mask(step: Step, name=None) -> Step:
    """Adds per-head action masks mirroring TensorFlow implementation."""

    heads = HEAD_TO_DATASET.keys()
    assert HEAD_TO_DATASET and name

    n = step["observation"]["timestep"].shape
    n = n[0] if n else 1

    masks = {head: (name in d) for head, d in HEAD_TO_DATASET.items()}
    masks = jax.tree.map(partial(np.array, dtype=bool), masks)
    masks = jax.tree.map(lambda x: np.full((n,), x, dtype=bool), masks)
    step["action_head_masks"] = masks
    return step


def pad_actions_and_proprio(
    traj: Trajectory,
    *,
    max_action_dim: int | None,
    max_proprio_dim: int | None,
) -> Trajectory:
    """Pads actions/proprio streams and records action padding mask."""

    actions = traj["action"]
    traj["action_pad_mask"] = jax.tree.map(lambda a: np.ones_like(a, dtype=bool), actions)

    raise NotImplementedError("plz transition to padding dict keys not pad array dim")
    if max_action_dim is not None:
        action_dim = actions.shape[-1]
        if action_dim > max_action_dim:
            raise ValueError(f"action_dim ({action_dim}) is greater than max_action_dim ({max_action_dim})")
        pad_width = [(0, 0)] * (actions.ndim - 1) + [(0, max_action_dim - action_dim)]
        traj["action"] = np.pad(actions, pad_width, mode="constant")
        traj["action_pad_mask"] = np.pad(traj["action_pad_mask"], pad_width, mode="constant", constant_values=False)

    if max_proprio_dim is not None and "proprio" in traj.get("observation", {}):
        proprio = traj["observation"]["proprio"]
        proprio_dim = proprio.shape[-1]
        if proprio_dim > max_proprio_dim:
            raise ValueError(f"proprio_dim ({proprio_dim}) is greater than max_proprio_dim ({max_proprio_dim})")
        pad_width = [(0, 0), (0, max_proprio_dim - proprio_dim)]
        traj["observation"]["proprio"] = np.pad(proprio, pad_width, mode="constant")

    return traj


log.debug("TODO chunk actions by jax.tree.map")


def batch_fn(xs: list[dict[Any]]):
    xs = jax.tree.map(lambda x: jax.device_put(x, cpu), xs)
    stack = lambda *a: jnp.asarray(a, device=cpu)
    tree = jax.tree.map(stack, *xs)
    # iscpu = jax.tree.map(lambda x: x.platform() == 'cpu' , tree)
    # iscpu_all = jax.tree.reduce(lambda x, y: x and y, iscpu)
    return tree


def chunk_action_and_observation(
    traj: Trajectory,
    *,
    window_size: int,
    action_horizon: int,
    override_window_size: int | None = None,
) -> Trajectory:
    """Chunks observations into histories and actions into windows."""

    actions = traj["action"]
    n = actions.shape[0]
    device = actions.device
    # pprint(spec({'action': actions}))

    def arange(*args, **kwargs):
        return jnp.asarray(np.arange(*args, **kwargs), device=cpu)

    # history_indices = jnp.arange(n)[:, None] + jnp.arange(-window_size + 1, 1)[None, :]
    # now, in numpy
    history_indices = arange(n)[:, None] + arange(-window_size + 1, 1)[None, :]

    timestep_pad_mask = history_indices >= 0
    if override_window_size is not None:
        valid_history = arange(window_size) >= window_size - override_window_size
        timestep_pad_mask = jnp.logical_and(timestep_pad_mask, valid_history)
    history_indices = jnp.maximum(history_indices, 0)
    # pprint(spec({'history_indices': history_indices}))

    chunked_obs = jax.tree.map(lambda x: x[history_indices], traj["observation"])
    chunked_obs["timestep_pad_mask"] = timestep_pad_mask
    traj["observation"] = chunked_obs

    if actions.ndim == 2:
        action_indices = arange(n)[:, None] + arange(action_horizon)[None, :]
        action_indices = jnp.minimum(action_indices, n - 1)
        actions = actions[action_indices]
    else:
        if actions.shape[1] < action_horizon:
            raise ValueError(
                f"Pre-chunked action does not have enough horizon to satisfy requested action_horizon={action_horizon}."
            )
        actions = actions[:, :action_horizon]

    actions = actions[history_indices]
    traj["action"] = actions
    # pprint(spec({'action': actions}))

    if "task" not in traj:
        traj["task"] = task = traj.pop("task", {})
    task = traj["task"]
    goal_timestep = task.get("timestep")
    goal_timestep = jnp.full((n,), n - 1, dtype=jnp.int32, device=cpu) if goal_timestep is None else goal_timestep
    # ezdiff(_t,traj['task'])

    t, w, h = jnp.meshgrid(arange(n), arange(window_size), arange(action_horizon), indexing="ij")
    relative_goal = goal_timestep[:, None, None] - (t - (window_size + 1) + w + h)
    traj["observation"]["task_completed"] = relative_goal <= 0

    action_pad_mask = traj.get("action_pad_mask")
    # pprint(spec({'action_pad_mask': action_pad_mask}))
    if action_pad_mask is None:
        action_pad_mask = jnp.ones((*actions.shape[:-1], actions.shape[-1]), dtype=bool)
    else:
        action_pad_mask = action_pad_mask
        if action_pad_mask.ndim == 2:
            action_pad_mask = action_pad_mask[:, None, None, :]
        elif action_pad_mask.ndim == 3:
            action_pad_mask = action_pad_mask[:, None, :, :]

    traj["action_pad_mask"] = jnp.logical_and(
        action_pad_mask,
        ~traj["observation"]["task_completed"][:, :, :, None],
    )
    return traj


def subsample(traj: Trajectory, *, length: int, rng: np.random.Generator) -> Trajectory:
    traj_len = traj["action"].shape[0]
    if traj_len <= length:
        return traj
    indices = rng.choice(traj_len, size=length, replace=False)
    indices.sort()

    def gather(value: Any):
        if isinstance(value, dict):
            return {key: gather(sub_value) for key, sub_value in value.items()}
        value = _ensure_array(value)
        return value[indices]

    for key in ("action", "observation", "task"):
        if key in traj:
            traj[key] = gather(traj[key])
    return traj


def zero_out_future_proprio(traj: Trajectory) -> Trajectory:
    """Removes all proprio inputs after first one to prevent causal confusion."""
    raise NotImplementedError()
    proprio = traj["observation"].get("proprio")
    if proprio is None:
        return traj
    proprio = _ensure_array(proprio)
    if proprio.ndim < 2:
        raise ValueError("Expected proprio to be at least 2D after chunking.")
    proprio[:, 1:] = 0
    traj["observation"]["proprio"] = proprio
    return traj


def _normalize_resize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("resize dimension must be positive")
        return size, size
    if len(size) != 2:
        raise ValueError("size must be an int or a length-2 tuple")
    height, width = (int(dim) for dim in size)
    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    return height, width


def _is_image_like(value: Any) -> bool:
    array = _ensure_array(value)
    return array.ndim >= 3 and array.shape[-1] in {1, 3, 4}


def _resize_image_array(
    array: Any,
    *,
    size: tuple[int, int],
    order: int,
) -> np.ndarray:
    if array.ndim < 3:
        raise ValueError("Expected image-like array with channel dimension")
    height_axis = array.ndim - 3
    width_axis = array.ndim - 2
    zoom_factors = [1.0] * array.ndim
    zoom_factors[height_axis] = size[0] / array.shape[height_axis]
    zoom_factors[width_axis] = size[1] / array.shape[width_axis]
    working_array = array
    orig_dtype = working_array.dtype
    if np.issubdtype(orig_dtype, np.integer):
        working_array = working_array.astype(np.float32)
    resized = ndimage.zoom(working_array, zoom_factors, order=order)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        resized = np.clip(np.rint(resized), info.min, info.max).astype(orig_dtype)
    else:
        resized = resized.astype(orig_dtype)
    return resized


def resize_frame_images(
    tree: dict[str, Any],
    *,
    size: int | tuple[int, int],
    keys: Sequence[str] | None = None,
    interpolation: str = "bilinear",
) -> dict[str, Any]:
    """Resizes image-like observations inside ``tree`` to ``size``.

    Parameters
    ----------
    tree:
        Frame dictionary emitted by :func:`flatten_trajectory`.
    size:
        Target spatial resolution.  Accepts an integer for square resizing or a
        ``(height, width)`` tuple for non-square outputs.
    keys:
        Optional observation keys to resize.  When ``None`` the transform
        automatically selects keys whose arrays end with a channel dimension of
        size ``1``, ``3`` or ``4`` (``rgb``/``rgba``/single channel images).
    interpolation:
        Either ``"nearest"`` or ``"bilinear"``.
    """

    order = _INTERPOLATION_TO_ORDER.get(interpolation)
    if order is None:
        raise ValueError(
            f"Unsupported interpolation '{interpolation}'. Expected one of {sorted(_INTERPOLATION_TO_ORDER)}"
        )
    target_size = _normalize_resize_size(size)
    obs = tree.get("observation")
    image = obs.get("image")
    tim = tree.get("task").get("image")
    if not isinstance(image, dict):
        raise TypeError("Expected 'observation' to contain an 'image' dictionary.")
        return tree
    if keys is None:
        raise NotImplementedError("Automatic key detection is disabled for now.")
        keys = [key for key, value in obs.items() if key != "pad_mask_dict" and _is_image_like(value)]
    for key in keys:
        image[key] = _resize_image_array(image[key], size=target_size, order=order)
        tim[key] = _resize_image_array(tim[key], size=target_size, order=order)
    tree["observation"]["image"] = image
    tree["task"]["image"] = tim
    return tree


def drop_empty_language(traj: Trajectory) -> Trajectory:
    language = traj.get("task", {}).get("language_instruction")
    if language is None:
        return traj
    language = np.asarray(language)
    keep = language != ""
    if np.all(~keep):
        raise ValueError("Trajectory does not contain any language annotation.")
    return traj


def uniform_goal_relabel(traj: Trajectory, *, rng: PRNGKey) -> Trajectory:
    obs = traj["observation"]
    n = obs["timestep"].shape[0]
    rand = jax.random.uniform(rng, shape=(n,))
    low = jnp.asarray(np.arange(int(n)), device=cpu)
    high = jnp.full(n, n, device=cpu)
    goal_indices = (rand * (high - low) + low).astype(int)
    goal_indices = jnp.clip(goal_indices, 0, n - 1)
    goal = jax.tree.map(lambda x: x[goal_indices], obs)
    _t = traj["task"]
    traj["task"] = merge(traj["task"], goal)

    iscpu = jax.tree.map(lambda x: x.platform() == "cpu", traj)
    iscpu_all = jax.tree.reduce(lambda x, y: x and y, iscpu)
    # pprint(iscpu_all)

    return traj


def maybe_cast_dtype(value: Any, dtype: np.dtype | type[np.number]) -> np.ndarray:
    array = _ensure_array(value)
    if array.dtype != dtype:
        array = array.astype(dtype)
    return array
