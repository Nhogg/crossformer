"""Builders converting raw data sources into Grain datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import fnmatch
from functools import partial
import hashlib
import json
import logging
from typing import Any

import grain.python as gp
import jax
import numpy as np

from crossformer.data.grain import metadata
from crossformer.utils.jax_utils import str2jax
from crossformer.utils.mytyping import DeprecatedError
from crossformer.utils.spec import ModuleSpec

log = logging.getLogger(__name__)


def _resolve_callable(spec_or_fn: ModuleSpec | Callable | None) -> Callable | None:
    if spec_or_fn is None:
        return None
    if isinstance(spec_or_fn, Mapping) and set(spec_or_fn.keys()) == {
        "module",
        "name",
        "args",
        "kwargs",
    }:
        return ModuleSpec.instantiate(spec_or_fn)  # type: ignore[arg-type]

    if not callable(spec_or_fn):
        raise TypeError(f"Expected callable or ModuleSpec, got {type(spec_or_fn)!r}")
    return spec_or_fn  # type: ignore[return-value]


def _resolve_source(
    source: Sequence[dict] | gp.RandomAccessDataSource | Callable[[], Any],
) -> Sequence[dict] | gp.RandomAccessDataSource:
    if callable(source):
        return _resolve_source(source())
    if isinstance(source, gp.RandomAccessDataSource):
        return source
    if isinstance(source, Sequence):
        return source
    raise TypeError("Data source must be a Sequence, RandomAccessDataSource, or callable returning one.")


def _sample_match_key(traj: Mapping[str, Any], template: str) -> Any:
    """Returns the first key in traj that matches the given template using fnmatch."""
    matches = [key for key in traj if fnmatch.fnmatch(key, template)]
    if not matches:
        raise ValueError(f"No keys match template {template!r}; available keys: {traj.keys()}")
    return traj[matches[0]]


@dataclass
class Keys:
    image: Mapping[str | Any] | Sequence[str] = field(default_factory=list)
    depth: Mapping[str | Any] | Sequence[str] = field(default_factory=list)
    proprio: Mapping[str | Any] = field(default_factory=dict)
    lang: str | None = None
    # lang: Mapping[str| Any] = field(default_factory=dict)


@dataclass
class GrainDatasetConfig:
    name: str
    source: Sequence[dict] | gp.RandomAccessDataSource | Callable[[], Any]
    standardize_fn: ModuleSpec | Callable | None = None
    episode_info: Any = None

    keys: Keys = field(default_factory=Keys)
    action_proprio_normalization_type: str = metadata.NormalizationType.NORMAL
    dataset_statistics: metadata.DatasetStatistics | Mapping[str, Any] | str | None = None
    statistics_save_dir: str | None = None
    force_recompute_dataset_statistics: bool = False
    action_normalization_mask: Sequence[bool] | None = None
    filter_fns: Sequence[ModuleSpec | Callable[[dict], bool]] = ()
    skip_norm: bool = False
    skip_norm_keys: Sequence[str] = ()
    seed: int = 0

    batch_size: int = 256
    debug: bool = False


def stable_hash_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) / 2**256


def _restructure_trajectory(
    step: dict,
    *,
    name: str,
    config: GrainDatasetConfig,
) -> dict:
    if "observation" not in step or "action" not in step:
        raise ValueError("Trajectory must contain 'observation' and 'action' keys.")

    # @codex add some action processing fn here.
    # in place of hard coded action definition

    info = step["info"]
    sid, eid = np.array(info["step"]).reshape(-1), np.array(info["episode"]).reshape(-1)
    step["info"]["id"] = {"step": sid, "episode": eid}  # patch
    step["observation"]["timestep"] = sid

    """
    step["observation"]["proprio"]["single_arm"] = np.concatenate(
        [
            step["observation"]["proprio"]["gripper"],
            step["observation"]["proprio"]["joints"],
            step["observation"]["proprio"]["position"],
        ],
        axis=-1,
    )


    # concat proprio joints and proprio gripper
    proprio = step["observation"]["proprio"]
    action = {
            'single': np.concatenate([proprio["joints"], proprio["gripper"]], axis=-1),
            'position': proprio["position"],
            }
    print(spec(step))
    quit()
    """

    task = {}
    task[config.keys.lang] = step[config.keys.lang]  # simple
    if False:
        raise DeprecatedError("opaque")
        if config.keys.lang is not None:
            language = _sample_match_key(step, config.keys.lang)
            language = np.asarray(language)
            if language.shape == ():
                language = np.repeat(language, traj_len)
            if language.shape[0] != traj_len:
                language = np.broadcast_to(language, (traj_len,))
            task[config.keys.lang] = language  # .astype(object)

    return {
        "observation": step["observation"],
        "task": task,
        "action": step["action"],  #  action,
        "dataset_name": str2jax(name),
        "info": {
            "dataset_name": str2jax(name),
            "id": {k.replace("_id", ""): np.array([v]).reshape(-1) for k, v in step.items() if "_id" in k},
        }
        | step.get("info", {}),
    }

    old_obs = traj["observation"]
    new_obs: dict[str, Any] = {}

    for new, old in config.image_obs_keys.items():
        key = f"image_{new}"
        if old is None:
            new_obs[key] = np.full((traj_len,), "", dtype=object)
        else:
            new_obs[key] = np.asarray(old_obs[old])
    for new, old in config.depth_obs_keys.items():
        key = f"depth_{new}"
        if old is None:
            new_obs[key] = np.full((traj_len,), "", dtype=object)
        else:
            new_obs[key] = np.asarray(old_obs[old])
    if config.proprio_obs_keys is not None:
        if config.proprio_obs_dims is None:
            raise ValueError("proprio_obs_dims must be provided when proprio_obs_keys is set.")
        for new, old in config.proprio_obs_keys.items():
            key = f"proprio_{new}"
            if old is None:
                new_obs[key] = np.zeros((traj_len, config.proprio_obs_dims[new]), dtype=np.float32)
            else:
                new_obs[key] = np.asarray(old_obs[old], dtype=np.float32)

    new_obs["timestep"] = np.arange(traj_len, dtype=np.int32)

    return {
        "observation": new_obs,
        "task": task,
        "action": action,
        "dataset_name": np.repeat(name, traj_len),
    }


def note_embodiment(x: dict):
    # mark the dataset's embodiments for mask later
    # for k in action: embodiment[k] = 1
    x["embodiment"] = {k: np.array(1, dtype=np.bool).reshape(-1) for k in x["action"]}
    return x


def _restructure_step_mano(x: dict, *, name: str, config: GrainDatasetConfig) -> dict:
    task = {}
    task[config.keys.lang] = x.pop(config.keys.lang)  # simple
    x["task"] = task

    x["observation"]["timestep"] = x["info"]["id"]["step"]
    x = jax.tree.map(lambda y: np.array(y), x)  # ensure numpy arrays
    return x


def _load_dataset_statistics(
    config: GrainDatasetConfig,
    ds: Sequence[dict],
) -> metadata.DatasetStatistics:
    if isinstance(config.dataset_statistics, metadata.DatasetStatistics):
        return config.dataset_statistics
    if isinstance(config.dataset_statistics, Mapping):
        return metadata.DatasetStatistics.from_json(config.dataset_statistics)
    if isinstance(config.dataset_statistics, str):
        with open(config.dataset_statistics, "r") as f:
            return metadata.DatasetStatistics.from_json(json.load(f))

    log.info("Computing dataset statistics from scratch...")
    hash_dependencies = [
        config.name,  # str(asdict(config))
        str(len(ds)),
        str(sorted(config.keys.image)),
        str(sorted(config.keys.depth)),
        str(sorted(config.keys.proprio.keys())),
    ]
    return metadata.compute_dataset_statistics(
        ds,
        proprio_keys=list(config.keys.proprio.keys()),
        hash_dependencies=hash_dependencies,
        save_dir=config.statistics_save_dir,
        force_recompute=config.force_recompute_dataset_statistics,
    )


def build_trajectory_dataset(
    config: GrainDatasetConfig,
) -> tuple[gp.MapDataset, metadata.DatasetStatistics]:
    """Builds a :class:`grain.MapDataset` emitting standardized trajectories."""

    ds = source = _resolve_source(config.source)

    filters = [_resolve_callable(fn) for fn in config.filter_fns if fn]
    for f in filters:
        ds = ds.filter(f)

    standardize = _resolve_callable(config.standardize_fn)

    r = _restructure_step_mano if "k3ds" in ds[0]["action"] else _restructure_trajectory
    restructure = ModuleSpec.create(r, name=config.name, config=config)
    ds = ds.map(ModuleSpec.instantiate(restructure))
    ds = ds.map(note_embodiment)

    stats = _load_dataset_statistics(config, ds)
    # stats = jax.tree.map(jax.device_get, stats)

    mask = config.action_normalization_mask
    norm: Callable = metadata.normalize_action_and_proprio
    norm = partial(
        norm,
        metadata=stats,
        normalization_type=config.action_proprio_normalization_type,
        proprio_keys=list(config.keys.proprio.keys()),
        action_mask=mask,
        skip_norm_keys=config.skip_norm_keys,
    )

    log.info("Applying lazy normalization to dataset...")
    log.info("TODO vectorize normalization fn on batch")
    ds = ds.map(norm) if not config.skip_norm else ds
    ds.dataset_statistics = stats  # type: ignore[attr-defined]
    return ds, stats
