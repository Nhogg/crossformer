"""High level utilities for constructing Grain based data pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import fnmatch
from functools import partial
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Sequence, TypeVar

import augmax
import grain
from grain._src.python import dataset as gd
import grain.experimental as ge
from grain.experimental import ThreadPrefetchIterDataset
import grain.python as gp
import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from rich.pretty import pprint
from tqdm import tqdm

from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain import builders, metadata, transforms
from crossformer.data.grain.datasets import (
    _postprocess_episode,
    drop,
    EpisodeInfo,
    unpack_record,
)
from crossformer.data.grain.transforms import batch_fn
from crossformer.data.grain.util.remap import _remap_lang, rekey
from crossformer.utils.deco import deprecate
from crossformer.utils.jax_utils import cpu, with_device_context
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.tree import flat, unflat

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


def _filter_language_present(traj: dict) -> bool:
    language = traj.get("task", {}).get("language_instruction")
    return language is not None
    language = np.asarray(language)
    return np.any(language != "")


def _proprio_within_bounds(traj: dict, max_proprio: float) -> bool:
    for key, value in traj.get("observation", {}).items():
        if key.startswith("proprio") and not jnp.all(jnp.abs(value) <= max_proprio):
            return False
    return True


@with_device_context(device=cpu)
def to_jax_key(r):
    if isinstance(r, np.random.Generator):
        return jax.device_put(jax.random.key(r.integers(2**32 - 1, dtype=np.uint32)), device=cpu)
    if isinstance(r, int | np.integer):
        return jax.device_put(jax.random.key(np.uint32(r)), device=cpu)
    return r  # already a JAX key (uint32[2])


def apply_trajectory_transforms(
    ds: gp.MapDataset,
    *,
    window_size: int = 1,
    action_horizon: int = 20,
    override_window_size: int | None = None,
    goal_relabeling_strategy: str | None = "uniform",
    goal_relabeling_kwargs: Mapping[str, Any] | None = None,
    subsample_length: int | None = None,
    skip_unlabeled: bool = False,
    max_action: float | None = None,
    max_proprio: float | None = None,
    max_action_dim: int | None = None,
    max_proprio_dim: int | None = None,
    post_chunk_transforms: Sequence[ModuleSpec | Callable] = (),
    seed: int = 0,
    config: builders.GrainDatasetConfig | None = None,
) -> gp.MapDataset:
    """Applies trajectory level transforms mirroring the TensorFlow pipeline."""

    # filters
    ds = ds.filter(_filter_language_present) if skip_unlabeled else ds
    _max_act_filter = lambda traj: jnp.all(jnp.abs(traj["action"]) <= max_action)
    ds = ds.filter(_max_act_filter) if max_action else ds
    ds = ds.filter(_proprio_within_bounds) if max_proprio else ds

    ds = ds.map(transforms.add_pad_mask_dict)
    # ds = ds.map(
    # lambda traj: transforms.pad_actions_and_proprio(
    # traj,
    # max_action_dim=max_action_dim,
    # max_proprio_dim=max_proprio_dim,
    # )
    # )

    """
    if goal_relabeling_strategy is not None:
        if goal_relabeling_strategy != "uniform":
            raise ValueError(f"Unsupported goal relabeling strategy: {goal_relabeling_strategy}")
        kwargs = goal_relabeling_kwargs or {}
        ds = ds.random_map(lambda traj, rng: transforms.uniform_goal_relabel(traj, rng=to_jax_key(rng), **kwargs))
    log.warning("TODO goal relabel kwargs not implemented")

    log.info("TODO chunk actions by jax.tree.map")
    chunk = partial(
        transforms.chunk_action_and_observation,
        window_size=window_size,
        action_horizon=action_horizon,
        override_window_size=override_window_size,
    )
    ds = ds.map(chunk)  # override grain.experimental.ConcatThenSplit
    """

    ds = ds.map(partial(transforms.add_head_action_mask, name=config.name))

    for transform in post_chunk_transforms:
        fn = _resolve_callable(transform)
        ds = ds.map(fn) if fn else ds

    return ds


def apply_frame_transforms(
    ds: gp.IterDataset,
    frame_transforms: Sequence[ModuleSpec | Callable] = (),
) -> gp.IterDataset:
    """Applies frame level transforms as simple map operations."""

    def onimg(tree, fn):
        """dont apply fn on trajectory, apply on image dict"""
        im = tree["observation"]["image"]
        im = fn(im)
        tree["observation"]["image"] = im
        return tree

    for transform in frame_transforms:
        fn = _resolve_callable(transform)
        # ds = ds.map(partial(onimg, fn=fn)) if fn else ds
        log.warn("TODO use augmax for frame transform")
        log.warn("TODO apply frame transforms only on image dict")
        ds = ds.map(fn) if fn else ds
    return ds


def do_frame_transforms(config, tfconfig, ds):
    # 3. do frame level transforms
    # 3.1. x decoding is already done
    # 3.2. resize frames if needed
    # 3.3. augmentations and dropout
    jd = partial(jax.jit, donate_argnums=0)
    frame_transform_aug = jax.jit(get_frame_transform(config, tfconfig))

    def squeeze(x, dim):
        return jax.tree.map(lambda y: jnp.squeeze(y, axis=dim), x)

    def unsqueeze(x, dim):
        return jax.tree.map(lambda y: jnp.expand_dims(y, axis=dim), x)

    @partial(jax.jit, donate_argnums=(0, 1))
    def frame_aug_with_reshape(rng, batch):
        # unsqueeze task images on extra dim=1
        # batch["task"]["image"] = unsqueeze(batch["task"]["image"], dim=1)
        batch = frame_transform_aug(rng, batch=batch)
        # batch["task"]["image"] = squeeze(batch["task"]["image"], dim=1)
        return batch

    ds = (
        # @todo is it better to use mp or to jit with constant size?
        ds.random_map(lambda x, rng: frame_aug_with_reshape(rng=to_jax_key(rng), batch=x))
    )
    return ds


@dataclass
class GrainDataLoader:
    dataset: gp.IterDataset
    statistics: metadata.DatasetStatistics | dict[str, metadata.DatasetStatistics]
    config: builders.GrainDatasetConfig

    @property
    def ds(self):
        return self.dataset

    @property
    @deprecate("for compatibility with tfds", strict=False)
    def dataset_statistics(self) -> Any:
        """return serialize stats for 1+ datasets"""
        if isinstance(self.statistics, metadata.DatasetStatistics):
            return jax.tree.map(jnp.array, self.statistics.to_json())
        else:
            return {k: jax.tree.map(jnp.array, v.to_json()) for k, v in self.statistics.items()}


@dataclass
class TransformConfig:
    traj_transform_kwargs: dict[str, Any] | None = None
    frame_transforms: Sequence[ModuleSpec | Callable] = ()
    resize_frames_to: int | tuple[int, int] | None = None
    resize_frame_keys: Sequence[str] | None = None
    resize_interpolation: str = "bilinear"


from typing import TypedDict

import jaxtyping as jt


class Batch(TypedDict, total=False):  # total=False makes extra keys allowed
    observation: jt.Float[jt.Array, "N 3"]
    action: jt.Array  # or Int[Array, "N"]
    meta: NotRequired[dict]


@deprecate("for compatibility with older dataset (TFDS)", strict=False)
def compatibility(tree: dict):
    """Ensures compatibility with older dataset formats by renaming keys."""

    # compatibility with current dataloader
    tree = flat(tree)

    # IMAGE
    # fix image keys
    side = fnmatch.filter(tree.keys(), "*image.side*")
    tree = rekey(tree, inp=side, out=[k.replace("image.side", "image_side") for k in side])
    worm = fnmatch.filter(tree.keys(), "*image.worm*")
    tree = rekey(tree, inp=worm, out=[k.replace("image.worm", "image_primary") for k in worm])
    low = fnmatch.filter(tree.keys(), "*image.low*")
    tree = rekey(tree, inp=low, out=[k.replace("image.low", "image_primary") for k in low])
    over = fnmatch.filter(tree.keys(), "*image.over*")
    tree = rekey(tree, inp=over, out=[k.replace("image.over", "image_over") for k in over])
    wrist = fnmatch.filter(tree.keys(), "*image.wrist*")
    tree = rekey(tree, inp=wrist, out=[k.replace("image.wrist", "image_left_wrist") for k in wrist])

    # pad_mask_dict = fnmatch.filter(tree.keys(), "*pad_mask_dict.image*")
    # tree = rekey(tree, inp=pad_mask_dict, out=[k.replace("pad_mask_dict.image", "pad_mask_dict") for k in pad_mask_dict])

    overhead = list(fnmatch.filter(tree.keys(), "*overhead*"))
    tree = drop(tree, overhead)
    worm = list(fnmatch.filter(tree.keys(), "*image_worm*"))
    tree = rekey(tree, inp=worm, out=[k.replace("image_worm", "image_primary") for k in worm])
    wrist = list(fnmatch.filter(tree.keys(), "*image_wrist*"))
    tree = rekey(tree, inp=wrist, out=[k.replace("image_wrist", "image_left_wrist") for k in wrist])

    # LANG
    language = fnmatch.filter(tree.keys(), "*language*")
    tree = rekey(tree, inp=language, out=[k.replace("language.embedding", "language_instruction") for k in language])

    # PROPRIO
    proprio = fnmatch.filter(tree.keys(), "*proprio.*")
    tree = rekey(tree, proprio, out=[k.replace("proprio.", "proprio_") for k in proprio])
    # drop all proprio_[gripper|joints|position]
    # noprop = fnmatch.filter(tree.keys(), "*proprio_*")
    # noprop = [k for k in noprop if "single" not in k]
    # tree = drop(tree, noprop)

    tree = unflat(tree)
    return tree


def _infer_observation_mappings(tree: dict) -> tuple[dict, dict, dict, dict] | None:
    obs = tree.get("observation", {})
    image_keys = set(obs.get("image", {}))
    depth_keys = set(obs.get("depth", {}))

    proprio = {k: v.shape[-1] for k, v in spec(obs.get("proprio", {}), simple=False).items()}
    # proprio_keys, proprio_dims = zip(*proprio.items())
    return image_keys, depth_keys, proprio


@dataclass
class ArecReader:
    path: Path  # path to arec files
    mix: Any
    ram_cache: bool = True


@with_device_context(device=cpu)
def dummy_data(*args, **kwargs):
    myspec = {
        "action": (8),
        "action_head_masks": {
            "mano": (),
            "single_arm": (),
        },
        "action_pad_mask": (8),
        "observation": {
            "image": {
                "overhead": (224, 224, 3),
                "side": (224, 224, 3),
                "worm": (224, 224, 3),
                "wrist": (224, 224, 3),
            },
            "pad_mask_dict": {
                "image": {
                    "overhead": (1,),
                    "side": (1,),
                    "worm": (1,),
                    "wrist": (1,),
                },
                "proprio": {
                    "gripper": (1,),
                    "joints": (1,),
                    "position": (1,),
                    "single_arm": (1,),
                },
                "timestep": (1,),
            },
            "proprio": {
                "gripper": (1),
                "joints": (7),
                "position": (6),
                "single_arm": (14),
            },
            "task_completed": (1, 50),
            "timestep": (1,),
            "timestep_pad_mask": (1,),
        },
        "task": {
            "image": {
                "overhead": (224, 224, 3),
                "side": (224, 224, 3),
                "worm": (224, 224, 3),
                "wrist": (224, 224, 3),
            },
            "language.embedding": (512,),
            "pad_mask_dict": {
                "image": {
                    "overhead": (),
                    "side": (),
                    "worm": (),
                    "wrist": (),
                },
                "language.embedding": (),
                "proprio": {
                    "gripper": (),
                    "joints": (),
                    "position": (),
                    "single_arm": (),
                },
                "timestep": (),
            },
            "proprio": {
                "gripper": (1,),
                "joints": (7,),
                "position": (6,),
                "single_arm": (14,),
            },
            "timestep": (),
        },
    }

    def is_leaf(y):
        return not isinstance(y, dict)

    def stack_em(*x):
        jnp.stack([*x], axis=0)

    key = jax.random.key(0)
    n = jax.random.randint(key, (), 500, 1000)
    return jax.tree.map(lambda x: jax.random.normal(key, x, dtype=np.float32), myspec, is_leaf=is_leaf)

    batch = [make_batch() for _ in range(n)]
    batch = jax.tree.map(stack_em, *batch)
    return batch


T = TypeVar("T")
S = TypeVar("S")


if multiprocessing.parent_process() is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"


def worker_init_fn(worker_id: int, worker_count: int):
    """Initialize each worker process."""
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL + 1)
    import os

    # 3. Force unified memory (usually a bad idea)
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # This allows paging to host RAM.
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_DEFAULT_DEVICE"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"

    import jax

    jax.default_device(cpu)
    pprint(jax.devices())

    import faulthandler
    import warnings

    faulthandler.enable()
    os.environ["PYTHONWARNINGS"] = "error"  # promote many warnings to errors
    warnings.simplefilter("error", ResourceWarning)  # unclosed files, etc.
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"  # full JAX traces
    # global cpu
    # cpu = jax.devices("cpu")[0]


def get_episode_lengths(path: Path) -> list[list[int]] | None:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [[int(idx) for idx in group] for group in payload]


def get_task_ids(ds, lengths):
    global_length = sum(l for l in lengths.values())

    offsets = {}  # quadratic but who cares
    for eid in lengths:
        eids = [x for x in lengths if x < eid]
        offsets[str(eid)] = sum(lengths[str(i)] for i in eids)

    # offsets = { int(eid): sum(lengths[str(i)] for i in range(int(eid))) for eid in lengths.keys() }
    def step2global_step(eid, id):
        return offsets[str(int(eid))] + id

    def sample_goal(x, rng):
        id = x["info"]["id"]
        # pprint((len(lengths),id))
        # pprint(id['episode_id']>=len(lengths))
        l = lengths[str(int(id["episode_id"]))]
        x["info"]["length"] = l
        x["info"]["global_length"] = global_length
        goal_id = jax.device_get(jax.random.randint(rng, (), 0, l - id["step_id"]))
        id["goal_rel_id"] = goal_id
        id["goal_abs_id"] = id["step_id"] + id["goal_rel_id"]
        id["goal_global_id"] = id["global_id"] + goal_id
        x["info"]["id"] = id
        return x

    def add_global_id(x):
        # handle if info.id is missing
        id = x.get("info", {}).get("id", {})
        id = id if id else {"episode_id": x.get("episode_id"), "step_id": x.get("step_id")}
        id["global_id"] = step2global_step(id["episode_id"], id["step_id"])
        x["info"] = x.get("info", {}) | {"id": id}
        return x

    ds = ds.map(add_global_id).random_map(lambda x, r: sample_goal(x, rng=to_jax_key(r)))
    return ds


def get_task_goals(ds):
    _ds = ds  # pin _ds so we can index into it without closure issues

    # search the ds for the global goal id and get the observation
    def make_goal(x):
        gid = int(x["info"]["id"]["goal_global_id"])
        task = {"task": _ds[gid]["observation"]}
        return x | task

    ds = ds.map(make_goal)
    return ds


def get_chunked_act(ds, a, o, chunk_fn):
    _ds = ds  # pin _ds so we can index into it without closure issues

    def make_chunk(x):
        eid = x["info"]["id"]["episode_id"]
        step_id = x["info"]["id"]["step_id"]
        gid = x["info"]["id"]["global_id"]
        chunk = [_ds[int(gid + i)] for i in range(a)]
        action = batch_fn([a["action"] for a in chunk])
        action_pad_mask = batch_fn([(c["info"]["id"]["episode_id"] == eid) for c in chunk])
        x = x | {
            "action": action,
            "action_pad_mask": action_pad_mask,
        }

    ds = ds.map(make_chunk)
    return ds


def make_data_source(cfg: cn.Train) -> grain.MapDataset:
    grain.config.update("py_debug_mode", True)

    mix = cfg.data.mix.value
    print(mix, mix.name)
    if not isinstance(mix, Arec):
        mix = Arec.from_name(mix.name)

    def exists(x: dict[Any] | None):
        return x is not None

    print(mix.source, len(mix.source))

    epinfo = EpisodeInfo(mix.source, mix)

    def pack_fn(xs):
        xs = [unpack_record(x) for x in xs]
        xs = _postprocess_episode(xs, steps=False)
        return xs  # batch_fn(xs)

    def sanity_check(traj: dict[jax.Array]) -> dict:
        eid = traj["episode_id"][0]
        same = jnp.all(traj["episode_id"] == eid)
        assert same, f"Episode ID mismatch in episode {eid}"
        return traj

    def maybe_init_lang(x: dict):
        if "language_instruction" not in x:
            x["language_instruction"] = ""
        if "language_embedding" not in x:
            x["language_embedding"] = np.zeros((512,), dtype=np.float32)
        return x

    ds = (
        grain.MapDataset.source(mix.source)
        .seed(42)
        .map(unpack_record)
        .map(maybe_init_lang)
        # .map(partial(_postprocess_episode, steps=False))
        # .filter(exists)
        .map(partial(drop, keys=["discount", "is_terminal", "reward", "is_first", "is_last"]))
        .map(
            partial(
                rekey,
                inp=["language_instruction", "language_embedding"],
                out=["language.instruction", "language.embedding"],
            )
        )
        .map(drop_str)
        # .map(sanity_check)
    )
    # ds = cache.CacheByKeyMapDataset(ds)

    dsit = iter(ds)
    example = next(dsit)

    print(spec(example))

    mappings = _infer_observation_mappings(example)
    assert mappings, "Trajectory missing observation key"

    lkey = "language.embedding"
    language = example.get(lkey)
    standardize_fn = partial(_remap_lang, k=lkey) if language is not None else None
    keys = builders.Keys(
        *mappings,
        lkey if language is not None else None,
    )

    dataset_config = builders.GrainDatasetConfig(
        name=mix.name,
        source=ds,
        episode_info=epinfo,
        keys=keys,
        standardize_fn=standardize_fn,
        skip_norm_keys=cfg.data.transform.skip_norm_keys,
        force_recompute_dataset_statistics=cfg.data.recompute,
    )

    traj_kwargs = cfg.data.traj.create(with_head_to_dataset=False)
    traj_kwargs["window_size"] = cfg.window_size or traj_kwargs.get("window_size", 1)
    log.warning("TODO move cfg.window_size to cfg.data")
    traj_kwargs.pop("task_augment_strategy")
    traj_kwargs.pop("task_augment_kwargs")
    tfconfig = TransformConfig(
        traj_transform_kwargs=traj_kwargs,
        frame_transforms={},
        resize_frames_to=64,
        resize_frame_keys=None,
    )
    return ds, dataset_config, tfconfig


def hwc2chw(img):
    return jnp.transpose(img, (2, 0, 1))  # HWC -> CHW


def get_frame_transform(
    config: builders.GrainDatasetConfig,
    tfconfig: TransformConfig,
) -> Callable:
    re = tfconfig.resize_frames_to
    re_wh: int = re[0] if isinstance(re, tuple) else re

    chain = augmax.Chain(
        augmax.Resize(re_wh),
        # augmax.ByteToFloat(),
        # augmax.RandomGrayscale(p= 0.5),
        # augmax.ChannelDrop(),
        # augmax.Rotate((-15, 15),p=0.3),
        # augmax.Warp(strength= 5, coarseness= 32),
        # augmax.Normalize(),
        # augmax.Blur(),
        # augmax.ChannelShuffle(),
        # augmax.RandomBrightness((-1.0, 1.0), p= 0.5),
        # augmax.RandomContrast(),
        # augmax.RandomGamma(),
        # augmax.RandomChannelGamma(),
        # augmax.ColorJitter(),
        # augmax.Solarization(),
    )

    v = jax.vmap
    # slots = [(c, k) for c in ("observation", "task") for k in config.keys.image]
    slots = [(c, k) for c in ("observation",) for k in config.keys.image]
    d5chw = v(v(hwc2chw))

    def parallelize_all_keys(rng, batch, f):
        imgs = [batch[c]["image"][k] for (c, k) in slots]  # each (B,T,C,H,W)
        if imgs[0].ndim == 4:  # in case no T
            imgs = [jnp.expand_dims(x, axis=(1)) for x in imgs]

        big = jnp.stack(imgs, axis=0)  # (N,B,T,C,H,W), N=8
        N, B, T, *_ = big.shape
        rngs = jax.random.split(rng, (N, B, T))
        big_out = v(v(v(f)))(rngs, big)  # (N,B,T,C,H,W)

        for i, (c, k) in enumerate(slots):  # scatter back
            batch[c]["image"][k] = big_out[i]
        return batch

    frame_transform_aug = partial(parallelize_all_keys, f=chain)
    return frame_transform_aug


def sharding_put(
    ds: grain.dataset.IterDataset,
    shard_fn,
    *,
    cpu_buffer_size: int = 4,
    device_buffer_size: int = 2,
) -> grain.dataset.IterDataset:
    """Moves the data to the given devices with prefetching.

    Stage 1: A CPU-side prefetch buffer.
    Stage 2: Per-device buffers for elements already transferred to the device.

    Args:
      ds: Dataset to prefetch.
      device: same arguments as in jax.device_put.
      cpu_buffer_size: Number of elements to prefetch on CPU.
      device_buffer_size: Number of elements to prefetch per device.

    Returns:
      Dataset with the elements prefetched to the devices.
    """
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cpu_buffer_size)
    # May raise ImportError if jax is not linked.

    if device_buffer_size > 0:
        ds = ds.map(lambda x: shard_fn(x))
        ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=device_buffer_size)
    return ds


def drop_str(batch):
    # if 'info' in batch: batch.pop('info')
    # pprint(spec(batch))

    if "language.instruction" in batch:
        batch.pop("language.instruction")
    if "dataset_name" in batch:
        batch.pop("dataset_name")
    if "dataset_name" in batch.get("info", {}):
        batch["info"].pop("dataset_name")
    return batch


def add_mask(x: dict):
    flag = x["info"]["id"]["episode"] % 2  # 95% of data
    x["mask"] = {
        "action_head_masks": x["action_head_masks"],
        # "action_pad_mask": x["action_pad_mask"],
        "timestep_pad_mask": np.ones_like(x["observation"]["timestep"]).astype(np.bool_),
        "only_adjustment": ~flag.astype(jnp.bool_),
    }

    # bwd compatibility
    x["observation"]["timestep_pad_mask"] = x["mask"]["timestep_pad_mask"]
    return x


def debug_dataset(ds, config, tracking: bool = True, n: int | float = 10):
    if tracking:
        options = ge.DatasetOptions(execution_tracking_mode=ge.ExecutionTrackingMode.STAGE_TIMING)
        ds = ge.WithOptionsIterDataset(ds, options)

    dsit = iter(ds)
    x = next(dsit)
    pprint(spec(x, simple=True))
    del x

    l = len(config.source) // config.batch_size
    for _ in tqdm(range(int(n))):
        x = next(dsit)
        del x

    if tracking:
        summary = gd.dataset.get_execution_summary(dsit)  # must run on iterator
        print(gd.stats.pretty_format_summary(summary))
    quit()


def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
    shard_fn: Callable,
    tfconfig: TransformConfig | None = TransformConfig(),
    shuffle_buffer_size: int | None = None,
    drop_remainder: bool = True,
    seed: int = 0,
) -> GrainDataLoader:
    """Builds a dataset of frames for a single dataset configuration.

    When ``resize_frames_to`` is provided the image observations within each
    frame are resized via :func:`transforms.resize_frame_images` before applying
    any additional ``frame_transforms``.
    """

    log.warning("TODO see update notes")
    # pack steps by episode_id
    # it makes action/obs chunk easier and goal_idx
    # try using a certain subsample length (500?) so that it can jit
    # ds = PackByKeyMapDataset(base, episode_len=episode_len_dict)
    # ds = ds.shuffle(seed=seed)           # shuffle episodes
    # define FlatMapFnIterDataset. pass shuffle idx fn to the flat map for random order
    # use grain.experimental.InterleaveIterDataset to interleave steps from different episodes
    ### its not a true shuffle but better than window shuffle
    ### and its deterministic and fast for IO & compute
    # use windowshuffle after
    # fix augmax to do image augmentations

    with jax.default_device(cpu):
        # 1. Build the trajectory dataset
        # 1.1. restructure keys
        # 1.2. compute / load statistics
        # 1.3. normalize with statistics and norm mask

        ds, stats = builders.build_trajectory_dataset(config)

        # 2. Apply trajectory transforms
        # 2.1. filter no lang
        # 2.2. maybe filter max action
        # 2.3. add pad mask and head masks
        # 2.4. seed
        # 2.5. goal relabel
        # 2.6. chunking and windowing
        # 2.7. maybe other transforms

        ds = apply_trajectory_transforms(ds, seed=seed, config=config)  # , **asdict(tfconfig))
        ds = ds.map(drop_str)

        # pprint(spec(next(iter(ds))))

        # ds = flatmap.PrivilegedFlatMapMapDataset(ds, transform=flatmap.PackByEpisode(key="info.id.episode_id", use_np=True))
        # ds = cache.CacheByKeyMapDataset(ds)
        ds = ds.seed(42).shuffle()  # shuffle before iter

        ds = ds.repeat() if train else ds  # repeat before iter ... repeat after shuffle
        ds = (
            # ds.map(add_mask)
            ds.map(lambda x: jax.tree.map(lambda y: np.array(y), x)).to_iter_dataset(  # to numpy
                grain.ReadOptions(num_threads=32)
            )  # iter before batch so that procs do batching and doesnt impede read threads
        )

        ds = ds.batch(config.batch_size, drop_remainder=drop_remainder)  # , batch_fn=batch_fn)
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=8, per_worker_buffer_size=10), worker_init_fn=worker_init_fn
        )
        # ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.id.episode_id", use_np=True))

        # debug_dataset(ds, config, n=1e3)

        def unbatch(items):
            for item in items:
                print("u")
                yield from item

        # ds = ds.pipe(unbatch)
        # ds = grain.experimental.WindowShuffleIterDataset(ds, window_size=10_00, seed=42)

        # this is as good as it gets with grain
        # now we need to move to jax arrays and shard to gpu
        # lastly, do frame aug on gpu en-batch for speed

        def np2jax(x):
            return jax.tree.map(lambda y: jnp.array(y, device=cpu), x)

        ds.dataset_statistics = stats  # type: ignore[attr-defined]
        #
        # blocks mp prefetch so that final jax ops can be main proc
        #
        ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=2)

        ds = ds.map(np2jax)  # dont use jax+grain yet... buggy
        ds = ds.map(shard_fn)
        ds = do_frame_transforms(config, tfconfig, ds)
        ds = ds.map(compatibility)

        log.info("returning final dataset")

        print("Dataset created... please be very patient while threads start up")
        return GrainDataLoader(dataset=ds, statistics=stats, config=config)

        log.warning("TODO add img dropout")
        log.warning("TODO add img or lang dropout")

        ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=config.batch_size)
        ds = (
            # @maddie is it better to use mp or to jit with constant size?
            ds.random_map(lambda x, rng: frame_aug_with_reshape(rng=to_jax_key(rng), batch=x)).map(
                jax.jit(compatibility)
            )
        )
        # pprint(spec(next(iter(ds))))
        # quit()

        # ds = sharding_put(ds, shard_fn=shard_fn, cpu_buffer_size=8, device_buffer_size=0)
        # debug_dataset(ds, config, n=1e2)

    ds.dataset_statistics = stats  # type: ignore[attr-defined]
    log.info("returning final dataset")

    print("Dataset created... please be very patient while threads start up")
    return GrainDataLoader(dataset=ds, statistics=stats, config=config)


def make_interleaved_dataset(
    configs: Sequence[builders.GrainDatasetConfig],
    *,
    train: bool,
    sample_weights: Sequence[float] | None = None,
    shuffle_buffer_size: int = 1,
    traj_transform_kwargs: dict[str, Any] | None = None,
    frame_transforms: Sequence[ModuleSpec | Callable] = (),
    resize_frames_to: int | tuple[int, int] | None = None,
    resize_frame_keys: Sequence[str] | None = None,
    resize_interpolation: str = "bilinear",
    batch_size: int | None = None,
    drop_remainder: bool = False,
    seed: int = 0,
) -> GrainDataLoader:
    """Creates a weighted mixture of datasets similar to the TensorFlow pipeline.

    Resizing is performed independently for every dataset when
    ``resize_frames_to`` is specified.
    """

    if not configs:
        raise ValueError("At least one dataset configuration must be provided.")

    datasets = []
    statistics = {}
    for cfg in configs:
        ds = make_single_dataset(
            cfg,
            train=train,
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transforms=frame_transforms,
            resize_frames_to=resize_frames_to,
            resize_frame_keys=resize_frame_keys,
            resize_interpolation=resize_interpolation,
            shuffle_buffer_size=None,
            batch_size=None,
            seed=seed,
        )
        datasets.append(ds.dataset)
        statistics[cfg.name] = ds.statistics

    if sample_weights is None:
        weights = np.ones(len(datasets), dtype=np.float32)
    else:
        weights = np.asarray(sample_weights, dtype=np.float32)
        if weights.shape[0] != len(datasets):
            raise ValueError("Number of sample weights must match number of datasets.")
    weights = weights / weights.sum()

    class _MixtureIterDataset(gp.IterDataset):
        def __init__(self, children: Sequence[gp.IterDataset], probs: np.ndarray):
            super().__init__(children)
            self._children = children
            self._probs = probs

        def __iter__(self):
            rng = np.random.default_rng(seed)
            child_iters = [iter(child) for child in self._children]
            while True:
                choice = rng.choice(len(child_iters), p=self._probs)
                try:
                    yield next(child_iters[choice])
                except StopIteration:
                    return

    mixture_dataset: gp.IterDataset = _MixtureIterDataset(datasets, weights)

    if shuffle_buffer_size and shuffle_buffer_size > 1:
        mixture_dataset = _BufferShuffleIterDataset(mixture_dataset, buffer_size=shuffle_buffer_size, seed=seed)

    if train:
        mixture_dataset = mixture_dataset.repeat()

    if batch_size is not None:
        mixture_dataset = mixture_dataset.batch(batch_size, drop_remainder=drop_remainder)

    mixture_dataset.dataset_statistics = statistics  # type: ignore[attr-defined]
    return GrainDataLoader(dataset=mixture_dataset, statistics=statistics)
