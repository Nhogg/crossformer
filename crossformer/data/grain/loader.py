"""High level factory class for constructing Grain based data loader."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import logging
import resource
from typing import Any

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
    MultiArrayRecordSource,
    unpack_record,
)
from crossformer.data.grain.embody import embody_transform
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
from crossformer.utils.spec import diff, ModuleSpec, spec
from crossformer.utils.tree import drop, flat, unflat
from crossformer.utils.type_checking import jtyped

log = logging.getLogger(__name__)


def np2jax(x):
    return jax.tree.map(lambda y: jnp.array(y, device=cpu), x)


@jtyped
def center_crop(img, h, w):
    H, W = img.shape[:2]
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w]


def mix_precompatibility(x, *, resize: int | tuple[int, int] | None = (224, 224)):
    """ensure mixed datasets have same keys."""

    def _reshape_int(y):
        """reshape int arrays to (-1,)"""
        return np.array(y).reshape(-1).astype(np.int32)

    x["info"]["id"] = jax.tree.map(_reshape_int, x["info"]["id"])
    x["observation"]["timestep"] = _reshape_int(x["observation"]["timestep"])

    """
    # RENAMING SOME IMAGE KEYS
    # worm -> low
    x = flat(x)
    worm = fnmatch.filter(x.keys(), "*worm*")
    x = rekey(x, inp=worm, out=[k.replace("worm", "low") for k in worm])
    # overhead -> over
    overhead = fnmatch.filter(x.keys(), "*overhead*")
    x = rekey(x, inp=overhead, out=[k.replace("overhead", "over") for k in overhead])
    x = unflat(x)
    """

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


def make_tfconfig(cfg: cn.Train) -> TransformConfig:
    traj_kwargs = cfg.data.traj.create(with_head_to_dataset=False)
    traj_kwargs["window_size"] = cfg.window_size or traj_kwargs.get("window_size", 1)
    log.warning("TODO move cfg.window_size to cfg.data")
    traj_kwargs.pop("task_augment_strategy")
    traj_kwargs.pop("task_augment_kwargs")
    return TransformConfig(
        traj_transform_kwargs=traj_kwargs,
        frame_transforms={},
        resize_frames_to=64,
        resize_frame_keys=None,
    )


def make_source_by_mix(
    mix: Arec | MultiDataSource,
    cfg: cn.Train,
) -> tuple[grain.Dataset, builders.GrainDatasetConfig]:
    # TODO reduce config scope by only passing cfg.data ?

    grain.config.update("py_debug_mode", log.isEnabledFor(logging.DEBUG))

    log.debug("mix source: %s (%d)", mix.name, len(mix.source))

    def maybe_init_lang(x: dict):
        if "language_instruction" not in x:
            x["language_instruction"] = ""
        if "language_embedding" not in x:
            x["language_embedding"] = np.zeros((512,), dtype=np.float32)
        return x

    if isinstance(mix.source, MultiArrayRecordSource):
        ds = (
            grain.MapDataset.source(mix.source)
            .seed(42)
            .map(lambda x: x | {"action": x["proprio"].copy()})
            .map(
                lambda x: x | {"proprio": jax.tree.map(lambda y: y[0], x["proprio"])}
            )  # select first item from horizon
            .map(lambda x: x | {"observation": {k: x.pop(k) for k in ["image", "proprio"]}})
            .map(lambda x: x | {"language.embedding": np.zeros((512,), dtype=np.float32)})
            .map(lambda x: x | {"info": jax.tree.map(lambda y: y[0], x["info"])})
        )

    elif hasattr(mix, "restructure") and mix.restructure is not None:
        base_fn = ModuleSpec.instantiate(mix.restructure)  # partial(restructure_fn)
        r = partial(base_fn, name=mix.name, lang_key=None)
        ds = grain.MapDataset.source(mix.source).seed(42).map(unpack_record).map(r)

    else:
        ds = (
            grain.MapDataset.source(mix.source)
            .seed(42)
            .map(unpack_record)
            .map(maybe_init_lang)
            # .map(partial(_postprocess_episode, steps=False))
            .map(partial(drop, keys=["discount", "is_terminal", "reward", "is_first", "is_last"]))
            .map(
                partial(
                    rekey,
                    inp=["language_instruction", "language_embedding"],
                    out=["language.instruction", "language.embedding"],
                )
            )
            .map(drop_str)  # TODO refactor to drop_type(typ=str)
        )

    dsit = iter(ds)
    example = next(dsit)

    # log.debug("example spec: %s", spec(example))

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
        keys=keys,
        standardize_fn=standardize_fn,
        restructure_fn=getattr(mix, "restructure", None),
        skip_norm_keys=cfg.data.transform.skip_norm_keys,
        force_recompute_dataset_statistics=cfg.data.recompute,
    )
    return ds, dataset_config


def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
    shuffle_buffer_size: int | None = None,
    drop_remainder: bool = True,
    seed: int = 0,
    resize: int | tuple[int, int] | None = (224, 224),
) -> GrainDataLoader:
    """Builds a dataset of frames for a single dataset configuration."""

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
        ds = ds.map(partial(mix_precompatibility, resize=resize))

        def unsqueeze_img_horizon(x):
            # unsqueeze proprio horizon dim to 1
            # in the current state of data transforms, we dont use many horizon steps
            # code still expects horizon dim
            x["observation"]["proprio"] = jax.tree.map(lambda y: y[None], x["observation"]["proprio"])
            return x

        ds = ds.map(unsqueeze_img_horizon)

        return ds, stats


def _apply_fd_limit(limit: int) -> tuple[int, int]:
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, old_hard), old_hard))
    return old_soft, old_hard


@dataclass
class GrainDataFactory:
    """in charge of making GrainDataLoader"""

    stats: dict[str, Any] = field(init=False, default_factory=dict)
    mp: int = 8  # of processes for mp prefetch. set to 0 to disable mp prefetch
    shuffle: bool = True
    mask_slot: bool = True  # mask body-part slots in embody_transform; disable for eval/debug
    shuffle_slot: bool = True  # shuffle body-part slot order in embody_transform; disable for eval/debug
    imaug: bool = True  # apply augmax image augmentations (channel shuffle)
    rotate: bool = True  # apply augmax.Rotate((-15, 15), p=0.3); independent of imaug
    # final image size; controls both mix_precompatibility cv2.resize and augmax.Resize.
    # None disables both stages (image stays at native size, no center_crop).
    resize: int | tuple[int, int] | None = (64, 64)
    patch_prob: float = 0.0
    patch_min_frac: float = 0.05
    patch_max_frac: float = 0.5

    def source2ds(self, dconfig, cfg: cn.Train, dataset: Arec, max_a: int = 0) -> GrainDataLoader:
        ds, stats = make_single_dataset(
            dconfig,
            train=True,
            shuffle_buffer_size=1,
            seed=cfg.seed,
            resize=self.resize,
        )
        if max_a > 0:
            embody_fn = partial(
                embody_transform,
                embodiment=dataset.embodiment,
                max_a=max_a,
                mask_prob=0.25 if self.mask_slot else 0.0,
                shuffle_slot=self.shuffle_slot,
            )
            ds = ds.map(embody_fn)
            log.debug("applied embody transform: %s (max_a=%d)", dconfig.name, max_a)
        self.stats[dconfig.name] = stats
        return ds

    def pad_and_mix(self, dsets: list[GrainDataLoader]) -> GrainDataLoader:
        """pad datasets to same keys and mix them by weight"""
        log.debug("mixing %d datasets", len(dsets))

        samples = [next(iter(ds)) for ds in dsets]
        a = samples[0]
        d = {}
        simple = True
        for b in samples[1:]:
            # ezdiff(a, b, simple=False)

            _a, _b = spec(flat(a), simple=simple), spec(flat(b), simple=simple)
            _d = diff(_a, _b, simple=simple)
            # print("diff", _d)
            # assert not _d.get("changed"), ("mismatched shape on the same key", {"changed": _d["changed"]})
            d["added"] = {**d.get("added", {}), **_d["added"]}
            d["removed"] = {**d.get("removed", {}), **_d["removed"]}

        d = {**d["added"], **d["removed"]}

        # double check
        dsets = [ds.map(partial(mix_compatibility, diff=d)) for ds in dsets]
        if True:
            samples = [next(iter(ds)) for ds in dsets]
            a = samples[0]
            for b in samples[1:]:
                # ezdiff(a, b, simple=False)
                _a, _b = spec(flat(a), simple=simple), spec(flat(b), simple=simple)
                _d = diff(_a, _b, simple=simple)
                print(_d)
                assert not _d.get("changed"), ("mismatched shape on the same key", {"changed": _d["changed"]})
            # print(spec(a))

        ds = grain.MapDataset.mix(dsets, weights=[1.0] * len(dsets))
        return ds

    def make(
        self,
        cfg: cn.Train,
        shard_fn: Callable | None = None,
        train: bool = True,
    ) -> GrainDataLoader:
        lim = _apply_fd_limit(512**2)

        log.info("applied fd limit: %s", lim)

        log.debug("data mix: %s", cfg.data.mix.value)
        mix = cfg.data.mix.value.flatten()
        log.debug("flattened mix: %s", mix)
        mix = [Arec.from_name(m[0]) for m in mix]  # m[1] is weights
        log.debug("arec sources: %s", mix)

        tfconfig = make_tfconfig(cfg)
        tfconfig.resize_frames_to = self.resize
        sources = [make_source_by_mix(m, cfg) for m in mix]

        # compute max action dim across all embodiments in the mix
        embodiments = [m.embodiment for m in mix]
        max_a = max(e.action_dim for e in embodiments)
        log.info("embodiment max_a=%d from %s", max_a, [e.name for e in embodiments])

        # if single then make single source and single dataset
        if len(sources) == 1:
            _, dconfig = sources[0]
            ds = self.source2ds(dconfig, cfg, dataset=mix[0], max_a=max_a)

        # if multi then make sources for each
        # then interleave them
        else:
            dsets = [self.source2ds(dc, cfg, dataset=m, max_a=max_a) for (_, dc), m in zip(sources, mix)]
            ds = self.pad_and_mix(dsets)

        ds = ds.seed(cfg.seed)
        if self.shuffle:
            ds = ds.shuffle()  # shuffle before iter
        ds = ds.repeat() if train else ds  # repeat before iter ... repeat after shuffle
        ds = (
            ds.map(add_mask)
            .map(lambda x: jax.tree.map(lambda y: np.array(y), x))  # to numpy
            .to_iter_dataset(
                grain.ReadOptions(num_threads=8)
            )  # iter before batch so that procs do batching and doesnt impede read threads
        )

        ds = ds.batch(cfg.data.loader.batch_size, drop_remainder=True)  # , batch_fn=batch_fn)
        if self.mp > 0:
            ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=self.mp, per_worker_buffer_size=8))

        batch = next(iter(ds))
        log.debug("batch spec: %s", spec(batch))

        # then frame lvl transforms in jax

        ds.dataset_statistics = self.stats  # type: ignore[attr-defined]
        #
        # blocks mp prefetch so that final jax ops can be main proc
        #
        ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=2)

        if train and self.patch_prob > 0:
            ds = ds.random_map(
                lambda x, rng: transforms.patch_occlude(
                    x,
                    rng,
                    prob=self.patch_prob,
                    min_frac=self.patch_min_frac,
                    max_frac=self.patch_max_frac,
                ),
                seed=cfg.seed,
            )

        ds = ds.map(np2jax)  # dont use jax+grain yet... buggy
        if shard_fn is not None:
            ds = ds.map(shard_fn)

        # quick hack with keys
        dconfig = sources[0][1]
        dconfig.keys.image = list(batch["observation"]["image"].keys())
        ds = do_frame_transforms(dconfig, tfconfig, ds, imaug=self.imaug, rotate=self.rotate)
        ds = ds.map(compatibility)

        log.info("returning final dataset")

        log.info("dataset created, waiting for prefetch threads to start")
        return GrainDataLoader(dataset=ds, statistics=self.stats, config=dconfig)


if __name__ == "__main__":
    f = GrainDataFactory()
    f.make()
