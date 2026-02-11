"""config nodes"""

from __future__ import annotations

from dataclasses import dataclass, Field, field
import enum
from enum import Enum
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import flax
from rich import print as pprint
from six import u
import tyro

from crossformer.cn import wab
from crossformer.cn.base import CN, default
from crossformer.cn.dataset import Dataset, DataSource, Head, MultiDataSource, transform
from crossformer.cn.dataset.action import DataPrep, DataSpec, HEAD2SPACE
from crossformer.cn.eval import Eval
from crossformer.cn.optim import Optimizer
from crossformer.cn.rollout import Rollout
from crossformer.cn.wab import Wandb
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.model.components.action_heads import (
    ActionHead,
    AdjFlowHead,
    DiffusionActionHead,
    FlowMatchingActionHead,
    L1ActionHead,
)
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26, ResNet26FILM
from crossformer.utils.spec import ModuleSpec

log = logging.getLogger(__name__)
log.info("Importing crossformer.cn")


class ModuleE(Enum):
    L1 = L1ActionHead
    DIFFUSION = DiffusionActionHead
    FLOW = FlowMatchingActionHead
    ADJFLOW = AdjFlowHead


@dataclass
class HeadFactory(CN):
    horizon: int = 20
    module: ModuleE = ModuleE.FLOW
    steps: int = 10  # diffusion/flow steps

    def __post_init__(self):
        if self.module == ModuleE.DIFFUSION:
            assert self.steps, "Diffusion steps must be 1+"
        if self.module == ModuleE.L1:
            pass
        if isinstance(self.module.value, FlowMatchingActionHead | AdjFlowHead):
            assert self.steps, "Flow steps must be 1+"

        h = Head[self.name.upper()]
        self.dim = HEAD2SPACE[h]

    def kwargs(self):
        d = {
            "pool_strategy": "use_map",
            "clip_pred": False,
            "loss_weight": 1.0,
            "constrain_loss_dims": True,
            "readout_key": f"readout_{self.name}",
            "num_preds": 0,  # force to use dim * horizon
        }
        if self.module == ModuleE.DIFFUSION:
            d["diffusion_steps"] = self.steps
        if isinstance(self.module.value, FlowMatchingActionHead | AdjFlowHead):
            d["flow_steps"] = self.steps

        return d

    def create(self):
        return ModuleSpec.create(
            self.module.value,
            action_horizon=self.horizon,
            action_dim=self.dim.value,
            **self.kwargs(),
        )


_SINGLE = "single_arm"
# updated to single going forward
# to match data loader and dynamic head selection
# and to retain readout keys
_SINGLE = "single"


class Size(Enum):
    DUMMY = "dummy"
    VANILLA = "vanilla"
    DETR = "detr"
    VIT_T = "vit_t"
    VIT_S = "vit_s"
    VIT_B = "vit_b"
    VIT_L = "vit_l"
    VIT_H = "vit_h"
    VINT = "vint"
    VIT_T_REPEAT = "vit_t_repeat"
    VIT_S_REPEAT = "vit_s_repeat"
    DETR_BIG = "detr_big"


@dataclass
class ModelFactory(CN):
    im: Sequence[str] = default(["primary", "side", "left_wrist"])
    proprio: Sequence[str] = default([_SINGLE])

    # which heads to create
    heads: Sequence[str] = default(
        [
            _SINGLE,
            "k3ds",
            # "bimanual",
            # "mano",
        ]
    )
    single: HeadFactory = HeadFactory(name=_SINGLE).field()
    bimanual: HeadFactory = HeadFactory(name="bimanual").field()
    mano: HeadFactory = HeadFactory(name="mano").field()
    k3ds: HeadFactory = HeadFactory(name="k3ds").field()

    size: Size = Size.DETR
    debug: bool = False  # y/n load model?

    def get_all_heads(self) -> dict[str, HeadFactory]:
        """get selected heads dynamically by searching dict"""

        # print({i: v for i, v in enumerate(self.asdict().values())})
        def _make(x):
            """see if x could be a HeadFactory
            for some reason they are serialized as dicts sometimes
            """
            try:
                return HeadFactory(**x)
            except Exception:
                return False

        heads = {k: _make(v) for k, v in self.asdict().items() if _make(v)}
        return heads

    def get_selected_heads(self) -> dict[str, HeadFactory]:
        """get selected heads based on self.heads"""
        all_heads = self.get_all_heads()
        heads = {k: v for k, v in all_heads.items() if k in self.heads and v.name in self.heads}
        return heads

    def create(self) -> dict[str, Any]:
        """create the model config"""

        token_embedding_size, transformer_kwargs = common_transformer_sizes(self.size.value)

        encoder = self.make_obs_im_encoder()
        # im = {k: self.make_obs_im(k) for k in self.im}
        im = {k: self.make_obs_im(k, encoder=encoder) for k in self.im}
        prop = {k: self.make_obs_proprio(k) for k in self.proprio}
        tok = im | prop

        heads = self.get_selected_heads()
        assert len(heads) > 0, "No heads selected"
        model = {
            "observation_tokenizers": tok,
            "heads": {k: v.create() for k, v in heads.items()},
            "readouts": {k: v.horizon for k, v in heads.items()},
            "token_embedding_size": token_embedding_size,
            "transformer_kwargs": transformer_kwargs,
        }
        return {"model": model}

    def spec(self) -> dict[str, Any]:
        "simplified keys that can be used to delete old model config"
        model = self.create()["model"]
        model = {
            "observation_tokenizers": {k: v["module"] for k, v in model["observation_tokenizers"].items()},
            "heads": {k: v["module"] for k, v in model["heads"].items()},
            "readouts": dict(model["readouts"].items()),
        }
        return {"model": model}

    def flatten(self) -> list[str]:
        """expand the delete config into a list of keys"""
        flattened = flax.traverse_util.flatten_dict(self.spec(), keep_empty_nodes=True)
        flattened = list(flattened.keys())
        return flattened

    def delete(self, flat):
        """delete keys from flax config tree based on flat spec"""
        """
        Delete keys from the model config based on the flattened list
        :param flat: list of keys to delete
        """

        def inside(a: list[str], b: list[str]):
            """see if a is inside b"""
            if len(a) > len(b):
                return False
            return all(_a == _b for _a, _b in zip(a, b[: len(a)]))

        mykeys = self.flatten()  # keep any keys already shared
        # delete model: observation_tokenizers, heads, readouts
        deletespec = {m[:2] for m in mykeys}

        for c in list(flat.keys()):
            if any(inside(m, c) for m in mykeys):
                continue
            if any(inside(d, c) for d in deletespec):
                print(f"del: {'.'.join(c)}")
                del flat[c]

        return flat

    def make_obs_proprio(self, key: str):
        """create observation tokenizer for proprio"""
        return ModuleSpec.create(LowdimObsTokenizer, obs_keys=[f"proprio_{key}"], dropout_rate=0.2)

    def make_obs_im(self, keys: str | Sequence[str], encoder=None):
        """create observation tokenizer for image"""
        if isinstance(keys, str):
            keys = [keys]
        if encoder is None:
            encoder = self.make_obs_im_encoder()

        return ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=[f"image_{k}" for k in keys],
            task_stack_keys=[f"image_{k}" for k in keys],
            task_film_keys=["language_instruction"],
            encoder=encoder,
        )

    def make_obs_im_encoder(self):
        """create image encoder"""
        return ModuleSpec.create(ResNet26FILM)

    def max_horizon(self) -> int:
        h = 0
        for head in self.get_selected_heads().values():
            h = max(h, head.horizon)
        return h

    def max_action_dim(self) -> int:
        d = 0
        for head in self.get_selected_heads().values():
            d = max(d, head.dim.value)
        return d


@dataclass()
class Train(CN):
    """Base Config"""

    debug: bool = False  # mostly turns off wandb

    steps: int = int(5e5)  # n training steps
    grad_acc = None

    modality: transform.Modality = transform.Modality.MULTI  # mode of observation
    window_size: int = 1
    head_name: str | None = None  # TODO why is this here ... see log warning

    pretrained_path: Path | str = "hf://rail-berkeley/crossformer"
    pretrained_step: int | None = None  # elapsed steps (if resume/restart)

    wandb: Wandb = Wandb().field()
    data: Dataset = Dataset().field()
    model: ModelFactory = ModelFactory().field()  # model specs
    optimizer: Optimizer = Optimizer().field()
    eval: Eval = Eval().field()
    rollout: Rollout = Rollout().field()

    log_level: Literal["debug", "info", "warning", "error"] = "warning"  # logging verbosity
    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 2000
    save_dir: str | Path = os.environ.get("BAFL_SAVE", Path().home())
    seed: int = 42

    def set_log_level(self):
        logging.basicConfig(level=self.log_level.upper(), force=True)
        log.info(f"Logging level set to {self.log_level.upper()}")

    def __post_init__(self):
        self.set_log_level()
        if self.data.transform.traj.action_horizon != self.model.max_horizon():
            log.warning(
                "WARNING: action horizon mismatch."
                f"data.transform.traj ({self.data.transform.traj.action_horizon}) "
                f"model.max_horizon ({self.model.max_horizon()})."
            )
            self.data.transform.traj.action_horizon = self.model.max_horizon()

        if self.data.transform.traj.max_action_dim != self.model.max_action_dim():
            log.warning(
                "WARNING: max action dim mismatch."
                f"data.transform.traj ({self.data.transform.traj.max_action_dim}) "
                f"model.max_action_dim ({self.model.max_action_dim()})."
            )
            self.data.transform.traj.max_action_dim = ActionDim(self.model.max_action_dim())

        if self.optimizer.lr.decay_steps is None:
            log.warning(f"WARNING: decay_steps is None, setting it to {self.steps}")
            self.optimizer.lr.decay_steps = self.steps

        log.warn("TODO: fix grad_acc and steps")
        log.warn("TODO: fix dataset")

        log.info("SUGGESTION: propogate from the main cfg down to children")
        log.warn("TODO: assert all keys that appear twice are the same")
        log.warn("TODO: is update_config for model arch only?")

    def transform_schema(self) -> dict[str, Any]:
        """
        Transform the schema of the config
        for use in crossformer repo
        """
        return {}


MODELS = []

# transform pipelines
TFORMS = []

DATAS = []

# experiment manager
MGRS = []


CONFIGS = []
CONFIGS += [Train(name="default")]

"""
# Single Arm Only
CONFIGS += [
    Train(name=x.name.replace("xgym", "bela"), data=x.name)
    for x in DataSource.REGISTRY.values()
    if x.head == Head.SINGLE
]

# # Human (Mano) Only
CONFIGS += [
    Train(name=x.name.replace("xgym", "bela"), data=x.name)
    for x in DataSource.REGISTRY.values()
    if x.head == Head.MANO
]
"""

CONFIGS = {x.name: x for x in CONFIGS}

# Train(
# name="bela_mano",
# # dataset=Dataset(),
# ),
# # Cross Embodiment
# Train(
# name="bela_cross",
# # dataset=Dataset(),
# ),


def cli() -> Train:
    """tyro.cli wrapper for subcommands (predefined experiment configs)"""
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in CONFIGS.items()})


# for tyro strong typing
ET = Enum("ExperimentTypes", {k: k for k, v in CONFIGS.items()})


@dataclass()
class Experiment(Train):
    # whether to use CLI or defined experiment
    exp: ET | None = None

    def __post_init__(self):
        if self.exp:
            d = CONFIGS[self.exp.value].asdict()
            raise NotImplementedError("TODO: reverse merge with self")
            # if you reverse the merge then you can still apply CLI args to the chosen Experiment
            # need to be careful about the defaults somehow though

            self.update(d)


TYP = Experiment


class Sweep(Train):
    # sweep parameters
    # -- expand into list of training runs
    # -- TODO how to compose them in a modular way?
    sweep: dict[str, list[Any]] | None = None


def main(cfg: TYP) -> None:  # experiment or sweep
    pprint(cfg)
    print()


if __name__ == "__main__":
    main(tyro.cli(TYP))
    # main(cli())

    # cmd = tyro.cli(None | Experiment)
    # print(cmd)
