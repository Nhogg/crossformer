"""Debug utility for visualizing Grain datasets."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import grain
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from rich import print
from tqdm import tqdm
import tyro

from crossformer import cn
from crossformer.data.grain import pipelines
from crossformer.data.grain.loader import GrainDataFactory
from crossformer.utils.spec import spec

log = logging.getLogger(__name__)
grain.config.update("py_debug_mode", True)


@dataclass
class Config(cn.Train):
    arec_path: Path | None = None
    dataset_name: str | None = None
    recompute: bool = False  # recompute data stats? y/n
    once: bool = True

    log_level: Literal["debug", "info", "warning", "error"] = "warning"  # logging verbosity


def main(cfg: Config) -> None:
    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # @deprecate('let grain do it', strict=False)
    def do_shard(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    if True:
        dataset = GrainDataFactory().make(cfg, shard_fn=do_shard, train=True)

    if False:
        # jax.config.update("jax_default_device", jax.devices("cpu")[0])
        _ds, dataset_config, tfconfig = pipelines.make_data_source(cfg)

        # TODO try to load only and then jit the preprocessing loop
        # and jax jit the loop itself maybe too
        # read_options = grain.sources.ReadOptions(num_threads=4, prefetch_buffer_size=500)
        # ds = _ds.repeat().to_iter_dataset(read_options).flat_map(MyFlatMap()).batch(4096, drop_remainder=True)
        # from grain import multiprocessing as gmp # npqa
        # ds = ds.mp_prefetch( gmp.MultiprocessingOptions(num_workers=16, per_worker_buffer_size=4, ))

        dataset = pipelines.make_single_dataset(
            dataset_config,
            shard_fn=do_shard,
            train=True,
            tfconfig=tfconfig,
            shuffle_buffer_size=1,
            seed=cfg.seed,
        )

    print("Dataset created... please be very patient while threads start up")

    print(dataset)
    dsit = iter(dataset.dataset)
    batch = next(dsit)
    print(spec(batch, simple=True))
    isnp = jax.tree.map(lambda x: isinstance(x, np.ndarray), batch)
    print(isnp)
    iscpu = jax.tree.map(lambda x: x.platform() == "cpu", batch)
    iscpu_all = jax.tree.reduce(lambda x, y: x and y, iscpu)
    print(iscpu)

    if cfg.once:
        print("exiting")
        quit()

    dsit = iter(dataset.dataset)
    for i in tqdm(range(int(1e4)), miniters=100, mininterval=0.1):
        x = next(dsit)
        if i % 1000 == 0:
            print(spec(x))

    del dataset  # threads arent daemon


if __name__ == "__main__":
    main(tyro.cli(Config))
