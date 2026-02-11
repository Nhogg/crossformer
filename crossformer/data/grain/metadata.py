"""Dataset metadata utilities for the Grain pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import grain
import jax
import numpy as np
from tqdm import tqdm

from crossformer.utils.jax_utils import cpu
from crossformer.utils.mytyping import Data

log = logging.getLogger(__name__)

EPS = 1e-8


class OnlineStats:
    def __init__(self, shape, dtype=np.float64):
        self.n = 0
        self.mean = np.zeros(shape, dtype=dtype)
        self.M2 = np.zeros(shape, dtype=dtype)
        self.min = np.full(shape, np.inf, dtype=dtype)
        self.max = np.full(shape, -np.inf, dtype=dtype)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=self.mean.dtype)

        # min / max (elementwise)
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

        # Welford (vectorized)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self, sample_std=False):
        if self.n <= 1:
            std = np.zeros_like(self.mean)
        else:
            denom = (self.n - 1) if sample_std else self.n
            std = np.sqrt(self.M2 / denom)

        return {
            # "n": self.n,
            "mean": self.mean,
            "std": std,
            "minimum": self.min,
            "maximum": self.max,
        }


@dataclass
class ArrayStatistics:
    mean: jax.Array
    std: jax.Array
    maximum: jax.Array
    minimum: jax.Array
    p99: jax.Array | None = None
    p01: jax.Array | None = None

    def to_json(self) -> dict:
        return jax.tree.map(lambda x: x.tolist(), asdict(self))

    @classmethod
    def from_json(cls, data: Mapping[str, Sequence[float]]) -> ArrayStatistics:
        toarr = np.array  # partial(jnp.array, device=cpu)
        s = jax.tree.map(toarr, data, is_leaf=lambda x: not isinstance(x, Mapping))
        # s = jax.tree.map(jax.device_get, s)
        return cls(**s)

    @classmethod
    def compute(cls, arr: np.ndarray) -> ArrayStatistics:
        print("computing stats for array of shape", arr.shape)
        return cls(
            mean=arr.mean(axis=0),
            std=arr.std(axis=0),
            maximum=arr.max(axis=0),
            minimum=arr.min(axis=0),
            p99=np.quantile(arr, 0.99, axis=0),
            p01=np.quantile(arr, 0.01, axis=0),
        )


@dataclass
class DatasetStatistics:
    action: dict[str, ArrayStatistics]
    proprio: dict[str, ArrayStatistics]
    num_transitions: int
    num_trajectories: int

    def to_json(self) -> dict:
        return {
            "action": jax.tree.map(lambda x: x.to_json(), self.action),
            "proprio": jax.tree.map(lambda x: x.to_json(), self.proprio),
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> DatasetStatistics:
        action = {
            key: ArrayStatistics.from_json(value)  # type: ignore[arg-type]
            for key, value in data["action"].items()  # type: ignore[union-attr]
        }
        proprio = {
            key: ArrayStatistics.from_json(value)  # type: ignore[arg-type]
            for key, value in data.get("proprio", {}).items()  # type: ignore[union-attr]
        }
        return cls(
            action=action,
            proprio=proprio,
            num_transitions=int(data.get("num_transitions", 0)),
            num_trajectories=int(data.get("num_trajectories", 0)),
        )


class NormalizationType(str):
    NORMAL = "normal"
    BOUNDS = "bounds"


def _cache_path(hash_dependencies: Iterable[str], save_dir: str | Path | None) -> Path:
    key = "".join(hash_dependencies).encode("utf-8")
    unique_hash = hashlib.sha256(key, usedforsecurity=False).hexdigest()
    if save_dir is not None:
        return Path(save_dir) / f"dataset_statistics_{unique_hash}.json"
    cache_dir = Path.home() / ".cache" / "arrayrecords"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"dataset_statistics_{unique_hash}.json"


def compute_dataset_statistics(
    ds: Data,
    *,
    proprio_keys: Sequence[str],
    hash_dependencies: Sequence[str],
    save_dir: str | Path | None = None,
    force_recompute: bool = False,
) -> DatasetStatistics:
    """Computes statistics for steps or loads cached values."""

    log.info(f"checking cache for stats {save_dir}")
    cache_path = _cache_path(hash_dependencies, save_dir)
    if cache_path.exists() and not force_recompute:
        log.info("they were in the cache")
        with cache_path.open("r") as f:
            return DatasetStatistics.from_json(json.load(f))
    log.info("no stats found in cache")

    actions = []
    proprio: dict[str, list[np.ndarray]] = {key: [] for key in proprio_keys}
    num_transitions = 0

    # assumes you can have it all in RAM
    # but we can
    # items: list[Data]= list(ds)
    N = int(ds[-1]["info"]["id"]["episode"][0] + 1)
    assert N, "No steps found in dataset."
    t = len(ds)

    _bs = 1  # 1024
    streams = jax.tree.map(lambda x: OnlineStats(np.array(x).shape, dtype=np.array(x).dtype), ds[0])
    streams = {"action": streams["action"], "proprio": streams["observation"]["proprio"]}

    def _update(x):
        # update action stats
        jax.tree.map(lambda a, stream: stream.update(a), x["action"], streams["action"])
        # update proprio stats
        jax.tree.map(lambda p, stream: stream.update(p), x["observation"]["proprio"], streams["proprio"])
        return x

    # ds = ds.slice(slice(5))
    mpds = (
        ds.to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=128))
        # .batch(_bs)
        # .mp_prefetch(grain.MultiprocessingOptions(num_workers=0))
        .map(_update)
    )

    def take_keys(x):
        return {k: v for k, v in x.items() if k == "action" or k == "observation"}

    mpds = mpds.map(take_keys)

    mpit = iter(mpds)
    trees = list(tqdm(mpit, desc="Loading ds for stats computation...", total=t // _bs))
    stats = jax.tree.map(lambda s: ArrayStatistics(**s.finalize()), streams)

    print(stats)

    # trees = [next(mpit) for _ in tqdm(range(2), desc="Computing ds stats...", total=t)]

    # for ep-wise
    # trees = jax.tree.map(lambda *a: np.concatenate([*a], axis=0), *trees)
    # for step-wise
    # trees = jax.tree.map(lambda *a: np.stack([*a], axis=0), *trees)

    # actions = trees["action"]
    # assert t, "No transitions found in dataset."
    # proprio = trees["observation"]["proprio"]

    print("computing stats for dataset with", N, "transitions and", t, "trajectories")
    stats = DatasetStatistics(
        action=stats["action"],
        proprio=stats["proprio"],
        num_transitions=N,
        num_trajectories=t,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(stats.to_json(), f)
        print(f"Saved dataset statistics to {cache_path}")
    return stats


NORM_MASK = {
    "single": [True] * 8,
    "joints": [True] * 7,
    "orientation": [True] * 3,
    "position": [True] * 3,  # TODO rename to pose
    "pose": [True] * 6,
    "gripper": [False],  # raw is better if mostly one of open/closed
    "k3ds": [[True] * 4] * 21,
}
NORM_MASK = jax.tree.map(np.array, NORM_MASK, is_leaf=lambda x: isinstance(x, list))
NORM_MASK = jax.tree.map(lambda x: x.astype(bool), NORM_MASK)


def normalize_action_and_proprio(
    step: dict,
    *,
    metadata: DatasetStatistics,
    normalization_type: str,
    proprio_keys: Sequence[str],
    action_mask: Sequence[bool] | None = None,
    skip_norm_keys: Sequence[str] = (),
    device: jax.Device | None = cpu,
) -> dict:
    """Normalizes actions and proprioceptive observations."""

    def _norm(a: np.ndarray, ma: ArrayStatistics) -> np.ndarray:
        """Normalizes a single action array.
        a: action
        ma: action statistics
        """
        if normalization_type == NormalizationType.NORMAL:
            mean = ma.mean
            std = np.maximum(ma.std, EPS)
            normalized = (a - mean) / std
        elif normalization_type == NormalizationType.BOUNDS:
            span = np.maximum(ma.maximum - ma.minimum, EPS)
            normalized = 2.0 * (a - ma.minimum) / span - 1.0
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
        return normalized

    action = step["action"]
    normalized = jax.tree.map(lambda a, ma: _norm(a, ma), action, metadata.action)

    def do_norm_mask(_action, _normalized, _mask):
        # normalize them all if no mask, else dont norm certain dim, ie gripper
        if _mask is not None:
            if _mask.shape[-1] != _normalized.shape[-1]:
                raise ValueError(
                    f"Length of action mask {_mask.shape} does not match action dimension {_normalized.shape}."
                )
            return np.where(_mask, _normalized, _action)
        return _normalized

    step["action"] = {k: do_norm_mask(action[k], normalized[k], NORM_MASK[k]) for k in action}

    obs = step.get("observation")
    for key in proprio_keys:
        if key not in obs or key in skip_norm_keys:
            continue
        if key not in metadata.proprio:
            continue
        value = obs[key]
        stats = metadata.proprio[key]
        if normalization_type == NormalizationType.NORMAL:
            mean = stats.mean
            std = np.maximum(stats.std, EPS)
            obs[key] = (value - mean) / std
        else:
            span = np.maximum(stats.maximum - stats.minimum, EPS)
            obs[key] = 2.0 * (value - stats.minimum) / span - 1.0
    step["observation"] = obs
    return step
