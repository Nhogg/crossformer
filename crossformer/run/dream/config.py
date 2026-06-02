from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flax import struct
from flax.training.train_state import TrainState
import jax
import optax

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.utils.train_utils import create_optimizer

from .constants import (  # noqa: F401
    ADD_THRESHOLDS_MM,
    KP_CONF_THRESHOLD,
    KP_MISSING_VALUE,
    KP_PEAK_AMBIGUITY_GAP,
    KP_PEAK_THRESHOLD,
    KP_SMOOTH_RADIUS,
    KP_SMOOTH_SIGMA,
    SOURCE_REAL,
    SOURCE_SYNTH,
)


@dataclass
class DreamVizConfig:
    every: int = 100


@dataclass
class Optim:
    lr: float = 1.5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    lr_schedule: str = "constant"  # constant | cosine | rsqrt
    clip_gradient: float | None = 1.0
    acc: int | None = None  # gradient accumulation
    frozen_keys: tuple[str, ...] = ()

    def kwargs(self, steps: int, frozen_keys: tuple[str, ...] = ()) -> dict:
        learning_rate = self.lr
        if self.lr_schedule != "constant" or self.warmup_steps > 0:
            decay_steps = max(steps, self.warmup_steps + 1)
            learning_rate = {
                "name": self.lr_schedule,
                "init_value": 0.0,
                "peak_value": self.lr,
                "warmup_steps": self.warmup_steps,
                **({"decay_steps": decay_steps} if self.lr_schedule == "cosine" else {}),
            }
        all_frozen_keys = self.frozen_keys + frozen_keys
        return {
            "learning_rate": learning_rate,
            "weight_decay": self.weight_decay,
            "clip_gradient": self.clip_gradient,
            "grad_accumulation_steps": self.acc,
            "frozen_keys": list(all_frozen_keys) if all_frozen_keys else None,
        }

    def create(self, params, steps: int, frozen_keys: tuple[str, ...] = ()):
        return create_optimizer(params, **self.kwargs(steps, frozen_keys=frozen_keys))


@dataclass
class Config:
    """Smoke-test config for DREAM."""

    name: str = "dream"
    seed: int = 0
    steps: int = 1_000_000
    log_every: int = 100
    raw_size: tuple[int, int] = (480, 640)
    net_in_size: tuple[int, int] = (400, 400)
    image_c: int = 3
    num_keypoints: int = 0  # 0 = infer from batch
    encoder: str = "vgg"  # vgg | tips
    variant: str = "full"  # quarter | half | full
    decoder: str = "auto"  # auto | upsample | deconv | dpt
    tips_variant: str = "tips_v2_b14"
    tips_checkpoint: Path | None = None
    tips_trainable: bool = False
    deconv_decoder: bool | None = None
    full_output: bool | None = None
    skip_connections: bool = False
    n_stages: int = 1
    internalize_spatial_softmax: bool = False
    learned_beta: bool = True
    initial_beta: float = 1.0
    sigma_pct: float = 1.0  # Gaussian std dev as percent of belief-map size.
    mask_weight: float = 0.1
    optim: Optim = default(Optim())
    viz: DreamVizConfig = default(DreamVizConfig())
    wandb: cn.Wandb = default(cn.Wandb(project="bela-dream"))
    verbose: bool = False

    # Aug
    imaug: bool = True
    rotate: bool = True
    real_mix: Arec = default(Arec.from_name("xgym_sweep_single"))
    real_prob: float = 0.3
    min_visible_kp: int = 4

    # LOADER
    bs: int = 50
    mix: Arec = default(Arec.from_name("xarm_dream_100k"))
    irl_mix: Arec = default(Arec.from_name("xgym_sweep_single"))
    irl_image_keys: tuple[str, ...] = ("side",)
    mp: int = 16
    mp_buf: int = 4  # per worker buffer size
    n_preshard: int = 2  # prefetch sharded data

    coco_prob: float = 0.5
    coco_dir: Path = Path("/home/bela/datasets/coco/train2014/")

    # Checkpointing
    save_dir: Path | None = Path.home().expanduser()
    save_interval: int = 25_000


@struct.dataclass
class DreamCheckpointModel:
    params: dict


@struct.dataclass
class DreamCheckpointState:
    model: DreamCheckpointModel
    step: jax.Array
    opt_state: optax.OptState


def _save_path(cfg: Config) -> str:
    if cfg.save_dir is None:
        raise ValueError("save_dir is None")
    return str((Path(cfg.save_dir).expanduser() / cfg.wandb.project / (cfg.wandb.group or "") / cfg.name).resolve())


def _checkpoint_state(state: TrainState) -> DreamCheckpointState:
    return DreamCheckpointState(
        model=DreamCheckpointModel(params=state.params),
        step=state.step,
        opt_state=state.opt_state,
    )
