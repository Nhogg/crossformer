"""High level factory class for constructing Grain based data loader."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import fnmatch
from functools import partial
import logging
from typing import Any

import cv2
import grain
from grain.experimental import ThreadPrefetchIterDataset
import jax
import jax.numpy as jnp
import numpy as np
from rich import print

from crossformer import cn
from crossformer.cn.dataset.mix import Arec, MultiDataSource
from crossformer.data.grain import builders, transforms
from crossformer.data.grain.datasets import (
    _postprocess_episode,
    EpisodeInfo,
    unpack_record,
)
from crossformer.data.grain.pipelines import (
    _infer_observation_mappings,
    add_mask,
    apply_trajectory_transforms,
    compatibility,
    do_frame_transforms,
    drop_str,
    GrainDataLoader,
    TransformConfig,
)
from crossformer.data.grain.util.remap import _remap_lang, rekey
from crossformer.utils.jax_utils import cpu
from crossformer.utils.spec import diff, ezdiff, spec
from crossformer.utils.tree import do_fn_key, drop, flat, unflat

log = logging.getLogger(__name__)


def np2jax(x):
    return jax.tree.map(lambda y: jnp.array(y, device=cpu), x)


from crossformer.utils.type_checking import Image, jtyped, ShapeError, Windowed


@jtyped
def center_crop(img, h, w):
    H, W = img.shape[:2]
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w]


def mix_precompatibility(x):
    """ensure mixed datasets have same keys"""

    # GENERAL SHAPE and TYPE compatibility
    def shape_check(item: Image | Windowed[Image]):
        if item.ndim == 3:
            return item
        if len(item) == 1 and item.ndim == 4:
            return item[0]
        raise ShapeError(f"Expected image with shape {item.shape} to have 3 dims (H,W,C) or 4 dims (1,H,W,C)")

    x["observation"]["image"] = jax.tree.map(shape_check, x["observation"]["image"])
    # print(spec(x['observation']['image']))
    # quit()

    crop = partial(center_crop, h=480, w=480)
    x["observation"]["image"] = jax.tree.map(crop, x["observation"]["image"])
    x["observation"]["image"] = jax.tree.map(
        lambda img: np.array(cv2.resize(img, (224, 224))), x["observation"]["image"]
    )

    def _reshape_int(y):
        """reshape int arrays to (-1,)"""
        return np.array(y).reshape(-1).astype(np.int32)

    x["info"]["id"] = jax.tree.map(_reshape_int, x["info"]["id"])
    x["observation"]["timestep"] = _reshape_int(x["observation"]["timestep"])

    # RENAMING SOME IMAGE KEYS
    # worm -> low
    x = flat(x)
    worm = fnmatch.filter(x.keys(), "*worm*")
    x = rekey(x, inp=worm, out=[k.replace("worm", "low") for k in worm])
    # overhead -> over
    overhead = fnmatch.filter(x.keys(), "*overhead*")
    x = rekey(x, inp=overhead, out=[k.replace("overhead", "over") for k in overhead])
    x = unflat(x)

    return x


def mix_compatibility(x, diff: dict):
    x = flat(x)
    for k in diff:
        if k not in x:
            # add missing key as padding
            # print(k, diff[k].shape)
            x[k] = np.zeros(diff[k].shape, dtype=diff[k].dtype)
            assert x[k].shape == diff[k].shape, ("err", x[k].shape, diff[k].shape)
    x = unflat(x)
    return x


def make_source_by_mix(
    mix: Arec | MultiDataSource,
    cfg: cn.Train,
) -> tuple[grain.Dataset, builders.GrainDatasetConfig, transforms.TransformConfig]:
    # TODO reduce config scope by only passing cfg.data ?

    grain.config.update("py_debug_mode", True)

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
        .map(drop_str)  # TODO refactor to drop_type(typ=str)
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


def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
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
        ds = ds.map(mix_precompatibility)

        def unsqueeze_img_horizon(x):
            # unsqueeze proprio horizon dim to 1
            # in the current state of data transforms, we dont use many horizon steps
            # code still expects horizon dim
            return do_fn_key(x, "observation.proprio.*", lambda y: y[None])

        ds = ds.map(unsqueeze_img_horizon)

        return ds, stats


@dataclass
class GrainDataFactory:
    """in charge of making GrainDataLoader"""

    stats: dict[str, Any] = field(init=False, default_factory=dict)

    def source2ds(self, dconfig, tfconfig, cfg: cn.Train) -> GrainDataLoader:
        ds, stats = make_single_dataset(
            dconfig,
            train=True,
            tfconfig=tfconfig,
            shuffle_buffer_size=1,
            seed=cfg.seed,
        )
        self.stats[dconfig.name] = stats
        return ds

    def pad_and_mix(self, dsets: list[GrainDataLoader]) -> GrainDataLoader:
        """pad datasets to same keys and mix them by weight"""
        print("n dsets", len(dsets))

        samples = [next(iter(ds)) for ds in dsets]
        a = samples[0]
        d = {}
        for b in samples[1:]:
            # ezdiff(a, b, simple=False)

            _a, _b = spec(flat(a), simple=False), spec(flat(b), simple=False)
            _d = diff(_a, _b, simple=False)
            assert not _d.get("changed"), ("mismatched shape on the same key", {"changed": _d["changed"]})
            d["added"] = {**d.get("added", {}), **_d["added"]}
            d["removed"] = {**d.get("removed", {}), **_d["removed"]}

        d = {**d["added"], **d["removed"]}

        # double check
        dsets = [ds.map(partial(mix_compatibility, diff=d)) for ds in dsets]
        if False:
            samples = [next(iter(ds)) for ds in dsets]
            a = samples[0]
            for b in samples[1:]:
                ezdiff(a, b, simple=False)
            print(spec(a))

        ds = grain.MapDataset.mix(dsets, weights=[1.0] * len(dsets))
        return ds

    def make(
        self,
        cfg: cn.Train,
        shard_fn: Callable | None = None,
        train: bool = True,
    ) -> GrainDataLoader:
        print(cfg.data.mix.value)
        mix = cfg.data.mix.value.flatten()
        print(mix)
        mix = [Arec.from_name(m[0]) for m in mix]  # m[1] is weights
        print(mix)

        sources = [make_source_by_mix(m, cfg) for m in mix]

        # if single then make single source and single dataset
        if len(sources) == 1:
            _, dconfig, tfconfig = sources[0]
            ds = self.source2ds(dconfig, tfconfig, cfg)

        # if multi then make sources for each
        # then interleave them
        else:
            dsets = [self.source2ds(dc, tc, cfg) for s, dc, tc in sources]
            ds = self.pad_and_mix(dsets)

        ds = ds.seed(cfg.seed).shuffle()  # shuffle before iter
        ds = ds.repeat() if train else ds  # repeat before iter ... repeat after shuffle
        ds = (
            ds.map(add_mask)
            .map(lambda x: jax.tree.map(lambda y: np.array(y), x))  # to numpy
            .to_iter_dataset(
                grain.ReadOptions(num_threads=32)
            )  # iter before batch so that procs do batching and doesnt impede read threads
        )

        ds = ds.batch(cfg.data.loader.batch_size, drop_remainder=True)  # , batch_fn=batch_fn)
        ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=8, per_worker_buffer_size=10))

        batch = next(iter(ds))
        print(spec(batch))

        # then frame lvl transforms in jax

        ds.dataset_statistics = self.stats  # type: ignore[attr-defined]
        #
        # blocks mp prefetch so that final jax ops can be main proc
        #
        ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=2)

        ds = ds.map(np2jax)  # dont use jax+grain yet... buggy
        if shard_fn is not None:
            ds = ds.map(shard_fn)

        # quick hack with keys
        dconfig, tfconfig = sources[0][1], sources[0][2]
        dconfig.keys.image = list(batch["observation"]["image"].keys())
        ds = do_frame_transforms(dconfig, tfconfig, ds)
        ds = ds.map(compatibility)

        log.info("returning final dataset")

        print("Dataset created... please be very patient while threads start up")
        return GrainDataLoader(dataset=ds, statistics=self.stats, config=dconfig)
