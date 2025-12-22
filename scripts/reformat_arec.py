from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, TypeAlias

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
import jax
from make_dset import to_unified_structure
import numpy as np
from rich import print
import tyro

from crossformer.cn.dataset.mix import Arec
from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.datasets import (
    unpack_record,
)
from crossformer.data.grain.map import flatmap
from crossformer.utils.spec import spec


def tile(x: dict, key: str, shape):
    if isinstance(shape, str):
        shape = (len(x[shape]), 1)

    x[key] = np.tile(x[key], shape)
    return x


def write_step(x: dict):
    n = len(x["episode_id"])
    x["step_id"] = np.arange(n)
    return x


@dataclass
class ReformatConfig:
    arec: Arec
    fmt: Literal["episode", "step"] | None = None  # format


StepWise: TypeAlias = grain.MapDataset
EpWise: TypeAlias = grain.MapDataset


def main(cfg: ReformatConfig):
    ds: EpWise = (
        grain.MapDataset.source(cfg.arec.source).seed(42).map(unpack_record)
        # .map(partial(_postprocess_episode, steps=False))
    )

    ds = ds.map(partial(tile, key="episode_id", shape="k3ds")).map(write_step).map(to_unified_structure)

    print(spec(ds[0]))
    print(jax.tree.map(lambda x: type(x), ds[0]))
    ds: StepWise = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="action", use_np=True))

    dsit = iter(ds)
    first = next(dsit)

    print(spec(first))
    print(jax.tree.map(lambda x: type(x), first))

    def get_build_fn():
        if cfg.fmt == "step":

            def _build():
                yield from dsit

            return _build
        else:
            raise NotImplementedError(f"Reformat to {cfg.fmt} not implemented.")

    writer = ArrayRecordBuilder(
        name=cfg.arec.name,
        version=cfg.arec.version,
        branch=f"to_{cfg.fmt}",
        shard_size=cfg.arec.builder.shard_size,
        writer_options=cfg.arec.builder.writer_options,
    )
    build = get_build_fn()
    writer.prepare(build)
    print("done writing")


if __name__ == "__main__":
    main(tyro.cli(ReformatConfig))
