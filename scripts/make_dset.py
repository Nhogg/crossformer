from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import json
from pathlib import Path
from typing import Callable

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from xclients.cli.preprocess import main as preprocess
from xclients.cli.preprocess import PrepConfig

from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.arec.arec import ArrayRecordBuilder, build_fn_per_episode
from crossformer.data.grain.datasets import (
    _postprocess_episode,
    drop,
    unpack_record,
)
from crossformer.data.grain.map import flatmap
from crossformer.utils.spec import spec


def test_prepare_empty_stream(tmp_path):
    builder = ArrayRecordBuilder(
        name="empty_ds",
        root=str(tmp_path),
        version="v0",
        shard_size=2,
    )

    builder.prepare(lambda: iter(()))

    meta = builder.meta
    assert meta["num_records"] == 0
    assert meta["name"] == "empty_ds"
    assert meta["version"] == "v0"

    spec_path = Path(tmp_path) / "empty_ds" / "spec.json"
    assert not spec_path.exists()

    meta_path = Path(tmp_path) / "empty_ds" / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        stored_meta = json.load(f)
    assert stored_meta["num_records"] == 0


def plot_3d_trajectories(
    sequences,
    out_path,
    cmap="hot",
    linewidth=2.0,
    figsize=(6, 6),
):
    """
    sequences: iterable of (n, 3) arrays
    out_path: path to save figure (e.g. 'traj.png')
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(0.0, 1.0)

    last_lc = None
    for seq in sequences:
        seq = np.asarray(seq)
        assert seq.ndim == 2 and seq.shape[1] == 3

        t = np.linspace(0.0, 1.0, len(seq))
        segments = np.stack([seq[:-1], seq[1:]], axis=1)

        lc = Line3DCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=linewidth,
            alpha=0.3,
        )
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        last_lc = lc

    # now plot mean sequence
    # pad them
    lens = np.array([len(seq) for seq in sequences])
    seqs = [
        np.pad(
            seq,
            ((0, lens.max() - len(seq)), (0, 0)),
            mode="edge",
        )
        for seq in sequences
    ]
    mean_seq = np.mean(np.stack(seqs, axis=0), axis=0)
    mean_segments = np.stack([mean_seq[:-1], mean_seq[1:]], axis=1)
    mean_lc = Line3DCollection(
        mean_segments,
        colors="blue",
        linewidth=linewidth * 1.5,
        alpha=0.7,
    )
    ax.add_collection(mean_lc)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # set limits
    all_points = np.concatenate(sequences, axis=0)
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    if last_lc is not None:
        fig.colorbar(last_lc, ax=ax, label="time")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_3d_points(
    sequences,
    out_path,
    cmap="hot",
    s=6,
    alpha=1.0,
    figsize=(6, 6),
):
    """
    sequences: iterable of (n, 3) arrays
    out_path: path to save figure (e.g. 'points.png')
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(0.0, 1.0)

    last_sc = None
    for seq in sequences:
        seq = np.asarray(seq)
        assert seq.ndim == 2 and seq.shape[1] == 3

        t = np.linspace(0.0, 1.0, len(seq))

        sc = ax.scatter(
            seq[:, 0],
            seq[:, 1],
            seq[:, 2],
            c=t,
            cmap=cmap,
            norm=norm,
            s=s,
            alpha=alpha,
        )
        last_sc = sc

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if last_sc is not None:
        fig.colorbar(last_sc, ax=ax, label="time")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


@dataclass
class BuildMGR:
    prep: PrepConfig

    name: str
    version: str
    shard_size: int = 1000

    fn: Callable = field(init=False)
    builder: ArrayRecordBuilder = field(init=False)

    def __post_init__(self):
        # assert self.version.startswith("v"), "Version should start with 'v'"
        assert self.version.count(".") == 2, "Version should be in the format 'vX.Y.Z'"

    def build(self, fn):
        print(self)
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            shard_size=self.shard_size,
        )
        print(self.builder.root)

        self.builder.prepare(fn)


def to_unified_structure(x: dict) -> dict:
    return {
        "info": {"id": x["episode_id"]},
        "action": x["k3ds"],
        "observation": {
            "image": {
                "low": x["low"],
                "over": x["over"],
                "side": x["side"],
            }
        },
    }


def main(cfg: BuildMGR):
    fn = partial(preprocess, cfg.prep)
    fn = partial(build_fn_per_episode, fn=fn)
    cfg.build(fn)
    # preprocess(cfg)

    mix = Arec.from_name(cfg.name)
    mix.upgrade = True
    shards = mix.get_shards()
    print(shards)
    source = grain.sources.ArrayRecordDataSource(shards)
    print(len(source))
    ds = grain.MapDataset.source(source).seed(42).map(unpack_record).map(partial(_postprocess_episode, steps=False))
    ds = ds.map(to_unified_structure)
    ds = ds.map(partial(drop, keys=["info"]))
    print(spec(ds[0]))
    ds = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="action", use_np=True))

    ds = grain.MapDataset.source(list(ds))

    for x in tqdm(ds):
        pass

    quit()
    print(source[0])

    X = list(tqdm(cfg.builder, leave=False))
    kp3d = [x["k3ds"][:, 0, :3] for x in X]
    lens = np.array([len(x["k3ds"]) for x in X])
    print([x.shape for x in kp3d])

    print({"max": lens.max(), "mean": lens.mean(), "std": lens.std()})
    # dict_keys(['episode_id', 'k3ds', 'low', 'over', 'side'])

    # plot_3d_points( kp3d, 'traj.png')
    # plot_3d_trajectories( kp3d, 'traj.png')


if __name__ == "__main__":
    main(tyro.cli(BuildMGR))
