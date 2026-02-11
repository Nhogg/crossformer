from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, TypeAlias

import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
import jax
import numpy as np
from rich import print
import tyro

from crossformer.cn.dataset.mix import Arec
from crossformer.cn.dataset.types import Head
from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.datasets import (
    unpack_record,
)
from crossformer.data.grain.map import flatmap
from crossformer.data.grain.util.mano import add_step_id, tile, to_unified_structure
from crossformer.utils.spec import spec


@dataclass
class ReformatConfig:
    arec: Arec
    fmt: Literal["episode", "step"] | None = None  # format


Step: TypeAlias = dict[str, jax.Array | np.ndarray]
Episode: TypeAlias = dict[str, jax.Array | np.ndarray]

StepWise: TypeAlias = grain.MapDataset[Step]
EpWise: TypeAlias = grain.MapDataset[Episode]


def identity(x: dict) -> dict:
    return x


shapekeys = {Head.SINGLE: "action.single", Head.K3DS: "k3ds"}
standards: dict[Any, Callable] = {
    Head.SINGLE: identity,
    Head.K3DS: to_unified_structure,
}


def main(cfg: ReformatConfig):
    ds: EpWise = (
        grain.MapDataset.source(cfg.arec.source).seed(42).map(unpack_record)
        # .map(partial(_postprocess_episode, steps=False))
    )
    shapekey = shapekeys[cfg.arec.head]

    match cfg.arec.head:
        case Head.K3DS:
            ds = ds.map(partial(tile, key="episode_id", shape=shapekey)).map(add_step_id).map(standards[cfg.arec.head])
        case Head.SINGLE:
            ds = ds
    print(spec(ds[0]))
    quit()

    print(spec(ds[0]))
    print(jax.tree.map(lambda x: type(x), ds[0]))

    ds: StepWise = FlatMapIterDataset(ds, transform=flatmap.UnpackFlatMap(key="info.id.step", use_np=True))
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
