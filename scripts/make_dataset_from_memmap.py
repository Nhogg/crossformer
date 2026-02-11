from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
from functools import partial
from pathlib import Path

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from xgym.rlds.util.trajectory import binarize_gripper_actions as binarize
from xgym.rlds.util.trajectory import scan_noop

from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.datasets import (
    drop,
)
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.util.mano import acroll_stacked
from crossformer.data.grain.util.remap import rekey
from crossformer.data.grain.utils import unflat
from crossformer.utils.io.memmap import read
from crossformer.utils.spec import spec


def rekey_match(tree: dict, match, src, tgt) -> dict:
    srckey = fnmatch.filter(tree.keys(), match)
    tgtkey = [x.replace(src, tgt) for x in srckey]
    return rekey(tree, inp=srckey, out=tgtkey)


@dataclass
class BuildConfig:
    path: Path  # path to memmap files

    name: str
    version: str
    branch: str = "to_step"
    shard_size: int = 1000

    builder: ArrayRecordBuilder = field(init=False)

    def __post_init__(self):
        # assert self.version.startswith("v"), "Version should start with 'v'"
        assert self.version.count(".") == 2, "Version should be in the format 'vX.Y.Z'"
        self.path = self.path.expanduser().resolve()

    def build(self, fn):
        print(self)
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            branch=self.branch,
            shard_size=self.shard_size,
        )
        print(self.builder.root)

        self.builder.prepare(fn)


def standardize(ep: dict, threshold=1e-3) -> dict:
    # rekey
    ep = rekey_match(ep, "/xgym/camera/*", "/xgym/camera/", "image.")
    ep = rekey_match(ep, "xarm_*", "xarm_", "proprio.")

    ### scale and binarize
    ep["proprio.pose"][:, :3] /= 1e3
    ep["proprio.position"] = ep["proprio.pose"][:, :3]
    ep["proprio.orientation"] = ep["proprio.pose"][:, 3:]

    ep["proprio.gripper"] /= 850
    _binarize = partial(binarize, open=0.95, close=0.4)  # doesnt fully close
    ep["proprio.gripper"] = np.array(_binarize(jnp.array(ep["proprio.gripper"])))

    ### filter noop cartesian
    pos = np.concatenate((ep["proprio.position"], ep["proprio.gripper"]), axis=1)
    noops = np.array(scan_noop(jnp.array(pos), threshold=threshold))
    mask = ~noops
    # filter noop joints
    jpos = np.concatenate([ep["proprio.joints"], ep["proprio.gripper"]], axis=1)
    ep["proprio.single"] = jpos
    jnoop = np.array(scan_noop(jnp.array(jpos), threshold=threshold))
    jmask = ~jnoop
    mask = np.logical_and(mask, jmask)

    ep = jax.tree.map(select := lambda x: x[mask], ep)

    # abs action is next proprio. cant select past last step

    # "language_instruction": self.task,
    # "language_embedding": self.lang,

    sid, n = np.arange(len(ep["time"])), len(ep["time"])
    actid = (sid + 1).clip(max=len(sid) - 1)

    ep = {"observation": unflat(ep), "info": {"step": sid}}
    ep["action"] = jax.tree.map(lambda x: x[actid], ep["observation"]["proprio"])

    c = 20
    chunk = acroll_stacked(sid, c)
    # print('chunk', chunk)
    ep["action"] = jax.tree.map(lambda x: x[chunk], ep["action"])
    return ep


def try_read(x: dict) -> dict | None:
    try:
        _info, data = read(x)
        return data
    except Exception as e:
        print(f"Error reading {x}: {e}")
        return None


def main(cfg: BuildConfig) -> None:
    print(cfg.path)
    files = list(cfg.path.rglob("*.dat"))

    lang = next(cfg.path.rglob("*.npy"))
    lang = np.load(lang)

    print(len(files))

    ds = grain.MapDataset.source(files).map(try_read).filter(lambda x: x is not None)
    ds = ds.map(partial(drop, keys=["gello_joints"]))
    ds = ds.map(standardize)

    def add_episode_id(i, x: dict) -> dict:
        x["info"]["episode"] = np.full((len(x["info"]["step"]),), i)
        return x

    ds = ds.map_with_index(add_episode_id)

    # def unsqueeze(x: dict) -> dict:
    # return jax.tree.map(lambda y: np.expand_dims(y, axis=1), x)
    # ds = ds.map(unsqueeze)  # for window dim

    print(spec(ds[0]))

    n_ep = len(ds)
    # cuda OOM with mp
    # ds  = ds .to_iter_dataset() .mp_prefetch( grain.multiprocessing.MultiprocessingOptions(num_workers=4, per_worker_buffer_size=2) )
    ckpt = list(tqdm(ds, total=n_ep))  # force loading
    n_step = sum([len(x["info"]["step"]) for x in tqdm(ckpt, total=n_ep)])

    ds = FlatMapIterDataset(
        grain.MapDataset.source(ckpt), transform=flatmap.UnpackFlatMap(key="info.step", use_np=True)
    )

    def add_lang(x: dict) -> dict:
        # x['info']['lang'] = lang
        x["language_embedding"] = lang
        return x

    ds = ds.map(add_lang)

    ds = grain.MapDataset.source(list(ds))

    bar = tqdm(total=n_step, desc="Building dataset")

    def build_progress(data: dict):
        """increments tqdm bar and yields data"""
        bar.update(1)
        return data

    ds = ds.map(build_progress)

    def build_fn():
        """yields one item from ds"""
        yield from ds

    cfg.build(build_fn)  # shouldn't need intermediate build_fn...


if __name__ == "__main__":
    main(tyro.cli(BuildConfig))
