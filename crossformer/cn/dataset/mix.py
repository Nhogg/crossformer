from __future__ import annotations

from dataclasses import dataclass, field

# from crossformer.data.oxe.oxe_standardization_transforms import  OXE_STANDARDIZATION_TRANSFORMS
import logging
from pathlib import Path
from typing import ClassVar, Sequence

import tyro

from crossformer.cn.base import CN
from crossformer.cn.dataset.types import Head
from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
from crossformer.data.oxe.oxe_dataset_mixes import DATASET_TO_HEAD, HEAD_TO_DATASET

log = logging.getLogger(__name__)


@dataclass
class DataSource(CN):
    REGISTRY: ClassVar[dict[str, DataSource]] = {}

    name: str = tyro.MISSING
    head: Head = tyro.MISSING

    def __post_init__(self):
        self.REGISTRY[self.name] = self
        members = {dataset for datasets in HEAD_TO_DATASET.values() for dataset in datasets}
        assert self.name in members, f"{self.name} missing from HEAD_TO_DATASET"
        assert self.name in OXE_DATASET_CONFIGS, f"{self.name} missing OXE config"
        # assert ( self.name in OXE_STANDARDIZATION_TRANSFORMS), f"{self.name} missing OXE standardization"

    def flatten(self):
        return [(self.name, 1.0)]


@dataclass
class TFDS(DataSource):
    pass


@dataclass
class Arec(DataSource):
    version: str | None = None
    # upgrade: bool = False
    branch: str = "main"

    builder: ArrayRecordBuilder = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.version is None:
            self.version = self.infer_version()
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            branch=self.branch,
        )

    @property
    def source(self):
        return self.builder.source

    @property
    def root(self) -> Path:
        return self.builder.root

    @staticmethod
    def from_name(name: str) -> Arec:
        config = OXE_DATASET_CONFIGS.get(name)
        if config is None:
            raise ValueError(f"No OXE dataset config found for name: {name}")
        version = config.get("version")
        head = DATASET_TO_HEAD.get(name)
        return Arec(name=name, version=version, head=head)

    def create(self):
        pass

    def infer_version(self) -> str:
        log.info(f"No version specified for dataset {self.name}.")
        log.info(f"Inferring latest version for dataset {self.name}")
        path = self.root / self.name
        versions = [v.name for v in path.iterdir() if v.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No versions found for dataset {self.name} in {path}")
        return sorted(versions)[-1]

    def get_version(self):
        # TODO infer version at post init
        return self.version if self.version else self.infer_version()

    @property
    def loc(self):
        return Path(self.name) / self.get_version() / self.branch

    def get_shards(self):
        shards = sorted(self.root.glob("*.arrayrecord"))
        if not shards:
            raise FileNotFoundError(f"No ArrayRecord shards found in {path}")
        return shards


@dataclass
class LeRobot(DataSource):
    pass


@dataclass
class MultiDataSource(DataSource):
    """Data Mix Configuration"""

    data: Sequence[DataSource] = tyro.MISSING
    weights: list[float] = tyro.MISSING
    head: str = Head.MULTI

    def __post_init__(self):
        msg = "Datasets and weights must be same length."
        assert len(self.data) == len(self.weights), msg

    def flatten(self) -> list[tuple[str, float]]:
        """for each d in dataset, flatten recursively  and multiply the contents by its weight"""
        out = []
        for d, w in zip(self.data, self.weights):
            for name, weight in d.flatten():
                out.append((name, weight * w))
        return out


XGYM = [
    TFDS(name="xgym_duck_single", head=Head.SINGLE),
    TFDS(name="xgym_lift_single", head=Head.SINGLE),
    TFDS(name="xgym_stack_single", head=Head.SINGLE),
    TFDS(name="xgym_sweep_single", head=Head.SINGLE),
]

NEW = [
    Arec(name="my_dataset", head=Head.SINGLE, version="0.5.3"),
]

# multi source
MultiDataSource(
    name="xgym",
    data=XGYM,
    weights=[1.0] * len(XGYM),
)
