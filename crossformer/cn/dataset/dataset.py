from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import jax
from rich.pretty import pprint

from crossformer.cn.base import CN, default
from crossformer.cn.dataset.action import DataPrep
from crossformer.cn.dataset.mix import DataSource, MultiDataSource
from crossformer.cn.dataset.transform import Transform
from crossformer.cn.dataset.transform.frame import FrameTransform
from crossformer.cn.dataset.transform.traj import TrajectoryTransform
from crossformer.data.dataset import make_interleaved_dataset

_ = Transform(name="default")


@dataclass
class Reader(CN):
    """Controls the nature of reading data from disk and preloading."""

    # location of tensorflow_datasets
    loc: Path | str = (Path().home() / "tensorflow_datasets").expanduser()

    load_camera_views: list[str] = default(
        [
            "primary",
            "side",
            # "high",
            "left_wrist",
        ]
    )
    load_proprio: bool = True
    load_depth: bool = False


@dataclass
class Loader(CN):
    """Controls the nature of loading size shuffle and parallelism."""

    use_grain: bool = True
    global_batch_size: int = 256  # to be divided by jax.devices()

    shuffle_buffer: int = 50_000
    balance_weights: bool = False

    threads_traj_transform: int = 48
    threads_traj_read: int = 48
    threads_frame_transform: int | None = 32  # None

    prefetch: int = 64  # number prefetch batches

    @property
    def batch_size(self):
        """retruns local batch size (per gpu)"""
        return self.global_batch_size // jax.process_count()


DataSourceE = Enum("DataSourceE", dict(DataSource.REGISTRY.items()))
TransformE = Enum("TransformE", dict(Transform.REGISTRY.items()))


@dataclass()
class Dataset(CN):
    """new dataset object consists of:
    - mix: a registered combination of datasets and sampling weights
    - transform: combination of data transforms
    - reader: data location and preloading selection rules
    - loader: batch size and parallelism settings
    """

    name = "default"

    # combination of datasets
    mix: DataSourceE = DataSourceE.xgym_stack_single
    # series of data transforms
    transform: Transform = Transform.REGISTRY["default"].field()

    reader: Reader = Reader().field()
    recompute: bool = False  # recompute data stats? y/n
    loader: Loader = Loader().field()

    @property
    def traj(self) -> TrajectoryTransform:
        return self.transform.traj

    @property
    def frame(self) -> FrameTransform:
        return self.transform.frame

    def get_mix(self):
        return MultiDataSource.REGISTRY[self.mix]

    @property
    def bs(self):
        """return the batch size for this dataset"""
        return self.loader.batch_size

    @property
    def gbs(self):
        """return the global batch size for this dataset"""
        return self.loader.global_batch_size

    def kwargs_list(self, oxe_fns: dict[str, Callable]):
        """makes just the dataset_kwargs_list in crossformer spec"""

        data_weights: list[tuple[str, float]] = self.mix.value.flatten()
        data_prep: list[DataPrep] = [
            DataPrep(
                name=d,
                weight=w,
                #
                loc=self.reader.loc,
                load_camera_views=self.reader.load_camera_views,
                load_depth=self.reader.load_depth,
                load_proprio=self.reader.load_proprio,
                skip_norm_keys=self.transform.skip_norm_keys,
            )
            for d, w in data_weights
        ]

        # monkey patch
        dks = [d.create(oxe_fns=oxe_fns) for d in data_prep]
        return dks, data_prep

    def kwargs(self):
        """all other "generic" kwargs"""
        frame = self.transform.frame.create(self.reader.load_camera_views)
        seq = self.transform.traj.create()
        out = {
            "shuffle_buffer_size": self.loader.shuffle_buffer,
            "batch_size": self.bs,
            #
            "traj_transform_kwargs": seq,
            "frame_transform_kwargs": frame,
            #
            # # balance_weights=None,
            "traj_transform_threads": self.loader.threads_traj_transform,
            "traj_read_threads": self.loader.threads_traj_read,
        }
        return out

    def create(self, oxe_fns: dict[str, Callable], train: bool = True):
        """create the dataset"""

        #
        # TODO: fix assert transform.traj.action_horizon to match model
        # TODO: why do we use transform.traj.task_augment_strategy: 'delete_task_conditioning'?
        #

        dks, data_prep = self.kwargs_list(oxe_fns=oxe_fns)
        frame = self.transform.frame.create(self.reader.load_camera_views)
        seq = self.transform.traj.create()

        pprint(dks)
        dataset = make_interleaved_dataset(
            dataset_kwargs_list=dks,
            sample_weights=[d.weight for d in data_prep],
            train=train,
            shuffle_buffer_size=self.loader.shuffle_buffer,
            batch_size=self.bs,
            #
            traj_transform_kwargs=seq,
            frame_transform_kwargs=frame,
            #
            # # balance_weights=None,
            traj_transform_threads=self.loader.threads_traj_transform,
            traj_read_threads=self.loader.threads_traj_read,
        )
        return dataset
