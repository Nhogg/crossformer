"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from functools import partial
import os
from pathlib import Path

from flax import struct
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState
import grain
from grain.experimental import ThreadPrefetchIterDataset
import jax
from jax.experimental import multihost_utils
import jax.image
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import numpy as np
import optax
from PIL import Image
from rich import print
from rich.pretty import pprint
from rich.rule import Rule
from rich.table import Table
from tips.scenic.utils import checkpoint as tips_checkpoint
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.datasets import MultiArrayRecordSource, unpack_record
from crossformer.data.grain.loader import _apply_fd_limit, _grain_mp_worker_init
from crossformer.model.dream import DreamTIPS, DreamVGG
from crossformer.model.load import resolve_checkpoint_path
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.callbacks.synth_viz import composite_robot, fk_keypoints, rasterize_robot, solve_pnp
from crossformer.utils.imaug import *  # noqa: F403
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import create_optimizer, Timer
import wandb

KP_CONF_THRESHOLD = 0.03
KP_SMOOTH_SIGMA = 1.0
KP_SMOOTH_RADIUS = 2
ADD_THRESHOLDS_MM = np.linspace(0.0, 100.0, 100, dtype=np.float32)
SOURCE_SYNTH = np.uint8(0)
SOURCE_REAL = np.uint8(1)


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

    # LOADER
    bs: int = 1
    mix: Arec = default(Arec.from_name("xarm_dream_100k"), init=False)
    real_mix: Arec = default(Arec.from_name("xgym_sweep_single"), init=False)
    real_prob: float = 0.0
    min_visible_kp: int = 4
    irl_mix: Arec = default(Arec.from_name("xgym_sweep_single"), init=False)
    irl_image_keys: tuple[str, ...] = ("low", "side")
    mp: int = 16
    mp_buf: int = 4  # per worker buffer size
    n_preshard: int = 2  # prefetch sharded data

    coco_prob: float = 0.5
    coco_dir: Path = Path.home() / "bela/datasets/coco/train2014"

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


def _resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, round(src_w * scale))
    new_h = max(target_h, round(src_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def make_coco_dataset(cfg: Config):
    names = sorted(p.name for p in cfg.coco_dir.glob("*.jpg"))
    if not names:
        raise FileNotFoundError(f"no .jpg files under {cfg.coco_dir}")
    net_h, net_w = cfg.net_in_size

    def read_bg(name: str):
        img = Image.open(cfg.coco_dir / name).convert("RGB")
        img = _resize_cover(img, net_w, net_h)
        return {"bg": np.asarray(img, dtype=np.uint8)}

    return grain.MapDataset.source(names).seed(cfg.seed).shuffle().repeat().map(read_bg).to_iter_dataset()


def add_bg(x: dict, rng, prob=0.5):
    if rng.random() < prob:
        x["image"] = np.where(x["mask"][..., None] > 0, x["image"], x["bg"])
    x.pop("bg", None)
    return x


def mix_with_bg(robot_ds, coco_ds, cfg: Config):
    """mix robot dataset with coco background"""

    ds = grain.experimental.ZipIterDataset([robot_ds, coco_ds], strict=False)
    ds = ds.map(lambda pair: pair[0] | pair[1])
    ds = ds.seed(cfg.seed).random_map(partial(add_bg, prob=cfg.coco_prob))
    return ds


def make_dataset(cfg: Config):
    synth = (
        grain.MapDataset.source(cfg.mix.source)
        .seed(42)
        .shuffle()
        .repeat()
        .map(unpack_record)
        .map(lambda s: {**s, "_kind": "synth"})
    )

    if cfg.real_prob > 0.0:
        real_src = cfg.real_mix.source
        real = grain.MapDataset.source(real_src).seed(43).shuffle().repeat()
        if not isinstance(real_src, MultiArrayRecordSource):
            real = real.map(unpack_record)
        real = real.map(lambda s: {**s, "_kind": "real"})
        md = grain.MapDataset.mix([synth, real], weights=[1.0 - cfg.real_prob, cfg.real_prob])
    else:
        md = synth

    ds = md.to_iter_dataset(
        grain.ReadOptions(num_threads=32, prefetch_buffer_size=1024)
    )  # iter before batch so that procs do batching and doesnt impede read threads

    def _prepare(s):
        kind = s.pop("_kind")
        out = prepare_sample_np(cfg, s) if kind == "synth" else prepare_irl_sample_np(cfg, s)
        out["source"] = SOURCE_SYNTH if kind == "synth" else SOURCE_REAL
        return out

    ds = ds.map(_prepare)
    ds = ds.filter(
        lambda s: int(np.asarray(s["source"])) == int(SOURCE_REAL)
        or int(np.asarray(s["keypoints_visible"]).sum()) >= cfg.min_visible_kp
    )
    if cfg.coco_prob:
        coco = make_coco_dataset(cfg)
        ds = mix_with_bg(ds, coco, cfg)
    ds = ds.batch(cfg.bs, drop_remainder=True)

    if cfg.mp > 0:
        lim = _apply_fd_limit(512**2)
        # Workers spawn via multiprocessing and re-import JAX. Without these,
        # each worker claims a CUDA context on GPU:0 and OOMs the parent's model.
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_buf),
            worker_init_fn=_grain_mp_worker_init,
        )

    shard_fn = make_shard_fn()
    ds = ds.map(shard_fn)
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cfg.n_preshard)
    return ds


def make_irl_dataset(cfg: Config):
    ds = grain.MapDataset.source(cfg.irl_mix.source).seed(cfg.seed).repeat()
    ds = ds.map(partial(prepare_irl_sample_np, cfg))
    ds = ds.map(lambda s: {**s, "source": SOURCE_REAL})
    ds = ds.batch(cfg.bs, drop_remainder=True)
    ds = ds.map(make_shard_fn())
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cfg.n_preshard)
    return ds


def make_shard_fn():
    mesh = Mesh(jax.devices(), axis_names="batch")

    def shard_batch(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    return shard_batch


def _normalize_kp2d(kp2d: jax.Array, h: int, w: int) -> jax.Array:
    return kp2d / jnp.array([w, h], dtype=jnp.float32)


def _denormalize_kp2d(kp2d_norm: jax.Array, h: int, w: int) -> jax.Array:
    return kp2d_norm * jnp.array([w, h], dtype=jnp.float32)


def net_out_size(cfg: Config) -> tuple[int, int]:
    h, w = cfg.net_in_size
    if cfg.variant == "full":
        return h, w
    if cfg.variant == "half":
        return h // 2, w // 2
    if cfg.variant == "quarter":
        return h // 4, w // 4
    raise ValueError(f"unknown variant: {cfg.variant}")


def make_model(cfg: Config, num_keypoints: int):
    if cfg.encoder == "vgg":
        return DreamVGG(
            num_keypoints=num_keypoints,
            variant=cfg.variant,
            decoder=cfg.decoder,
            deconv_decoder=cfg.deconv_decoder,
            full_output=cfg.full_output,
            skip_connections=cfg.skip_connections,
            n_stages=cfg.n_stages,
            internalize_spatial_softmax=cfg.internalize_spatial_softmax,
            learned_beta=cfg.learned_beta,
            initial_beta=cfg.initial_beta,
        )
    if cfg.encoder == "tips":
        if cfg.n_stages != 1:
            raise ValueError("TIPS encoder currently supports n_stages=1")
        if cfg.internalize_spatial_softmax:
            raise ValueError("TIPS encoder does not support internalize_spatial_softmax")
        return DreamTIPS(
            num_keypoints=num_keypoints,
            variant=cfg.variant,
            decoder=cfg.decoder,
            tips_variant=cfg.tips_variant,
            freeze_encoder=not cfg.tips_trainable,
        )
    raise ValueError(f"unknown encoder: {cfg.encoder}")


def load_tips_params(cfg: Config, params):
    if cfg.encoder != "tips":
        return params
    ckpt_path = resolve_checkpoint_path(cfg.tips_variant, cfg.tips_checkpoint)
    p = unfreeze(params)
    p["tips"] = tips_checkpoint.load_checkpoint(ckpt_path, p["tips"])
    print(f"  tips_checkpoint: {ckpt_path}")
    return freeze(p)


def frozen_keys(cfg: Config) -> tuple[str, ...]:
    if cfg.encoder == "tips" and not cfg.tips_trainable:
        return ("tips.*",)
    return ()


def _normalize_kp2d_np(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    return kp2d / np.array([w, h], dtype=np.float32)


def _denormalize_kp2d_np(kp2d_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    return kp2d_norm * np.array([w, h], dtype=np.float32)


def _shrink_crop_resolution(raw_h: int, raw_w: int, net_h: int, net_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    scale_w = raw_w / net_w
    ref_h_from_w = int(scale_w * net_h)
    scale_h = raw_h / net_h
    ref_w_from_h = int(scale_h * net_w)

    if raw_w >= ref_w_from_h:
        crop_w, crop_h = ref_w_from_h, raw_h
    else:
        crop_w, crop_h = raw_w, ref_h_from_w

    left = (raw_w - crop_w) // 2
    top = (raw_h - crop_h) // 2
    return (crop_h, crop_w), (top, left)


def _shrink_crop_keypoints_np(kp2d: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    (crop_h, crop_w), (top, left) = _shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    out = np.asarray(kp2d, dtype=np.float32).copy()
    out[:, 0] = (out[:, 0] - left) / crop_w * net_w
    out[:, 1] = (out[:, 1] - top) / crop_h * net_h
    return out


def _shrink_crop_intrinsics_np(K: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    (crop_h, crop_w), (top, left) = _shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 2] -= left
    out[1, 2] -= top
    out[0] *= net_w / crop_w
    out[1] *= net_h / crop_h
    return out


def _default_intrinsics_np(h: int, w: int) -> np.ndarray:
    f = np.float32(600.0)
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]], dtype=np.float32)


def _opencv_w2c_np(w2c_raw: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c_raw, dtype=np.float32).T
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = flip @ w2c[:3, :3]
    out[:3, 3] = flip @ w2c[:3, 3]
    return out


def _shrink_crop_image_np(image: np.ndarray, net_h: int, net_w: int, resample: int) -> np.ndarray:
    raw_h, raw_w = image.shape[:2]
    (crop_h, crop_w), (top, left) = _shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    pil = Image.fromarray(image)
    pil = pil.crop((left, top, left + crop_w, top + crop_h))
    return np.asarray(pil.resize((net_w, net_h), resample))


def _image_to_float(image: jax.Array) -> jax.Array:
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def prepare_sample_np(cfg: Config, sample: dict) -> dict:
    raw_h, raw_w = cfg.raw_size
    net_h, net_w = cfg.net_in_size
    image = np.asarray(sample["image"])
    if tuple(image.shape[:2]) != (raw_h, raw_w):
        raise ValueError(f"expected raw_size={(raw_h, raw_w)} but got {tuple(image.shape[:2])}")
    image = _shrink_crop_image_np(image, net_h, net_w, Image.BILINEAR)
    mask = _shrink_crop_image_np(np.asarray(sample["mask"]), net_h, net_w, Image.NEAREST)
    joints = np.asarray(sample["state"]["joints"], dtype=np.float32)
    gripper = np.asarray([sample["state"]["gripper"]], dtype=np.float32)
    q = np.concatenate([joints, gripper], axis=-1)
    kp2d_raw = np.asarray(sample["state"]["kp2d"], dtype=np.float32)
    kp2d_netin = _shrink_crop_keypoints_np(kp2d_raw, raw_h, raw_w, net_h, net_w)
    kp2d_norm = _normalize_kp2d_np(kp2d_netin, net_h, net_w)
    K = np.asarray(sample["camera"]["intr"]["K"], dtype=np.float32)
    w2c = _opencv_w2c_np(sample["camera"]["extr"]["w2c"])
    return {
        "image": image,
        "mask": mask,
        "q": q,
        "keypoints_2d_norm": kp2d_norm,
        "keypoints_2d_netin": kp2d_netin,
        "keypoints_2d_raw": kp2d_raw,
        "keypoints_visible": np.asarray(sample["info"]["kp_visible"], dtype=bool),
        "K": _shrink_crop_intrinsics_np(K, raw_h, raw_w, net_h, net_w),
        "w2c": w2c,
    }


def prepare_irl_sample_np(cfg: Config, sample: dict) -> dict:
    raw_h, raw_w = cfg.raw_size
    net_h, net_w = cfg.net_in_size
    image_key = np.random.choice(cfg.irl_image_keys)
    image = np.asarray(sample["image"][image_key])
    if tuple(image.shape[:2]) != (raw_h, raw_w):
        raise ValueError(f"expected raw_size={(raw_h, raw_w)} but got {tuple(image.shape[:2])}")
    image = _shrink_crop_image_np(image, net_h, net_w, Image.BILINEAR)
    joints = np.asarray(sample["proprio"]["joints"][0], dtype=np.float32)
    gripper = np.asarray(sample["proprio"]["gripper"][0], dtype=np.float32).reshape(1)
    kp2d_norm = np.full((10, 2), 0.5, dtype=np.float32)
    K = _default_intrinsics_np(raw_h, raw_w)
    return {
        "image": image,
        "mask": np.ones((net_h, net_w), dtype=np.uint8),
        "q": np.concatenate([joints, gripper], axis=-1),
        "keypoints_2d_norm": kp2d_norm,
        "keypoints_2d_netin": _denormalize_kp2d_np(kp2d_norm, net_h, net_w),
        "keypoints_2d_raw": _denormalize_kp2d_np(kp2d_norm, raw_h, raw_w),
        "keypoints_visible": np.zeros((10,), dtype=bool),
        "K": _shrink_crop_intrinsics_np(K, raw_h, raw_w, net_h, net_w),
        "w2c": np.eye(4, dtype=np.float32),
    }


def _build_heatmaps_one(
    kp2d: jax.Array,
    visible: jax.Array,
    image_h: int,
    image_w: int,
    sigma: float = 2.0,
) -> jax.Array:
    ys = jnp.arange(image_h, dtype=jnp.float32)[:, None]
    xs = jnp.arange(image_w, dtype=jnp.float32)[None, :]
    u = kp2d[:, 0]
    v = kp2d[:, 1]
    pixel_u = u.astype(jnp.int32)
    pixel_v = v.astype(jnp.int32)
    dist2 = (xs - u[:, None, None]) ** 2 + (ys - v[:, None, None]) ** 2
    heatmaps = jnp.exp(-dist2 / (2.0 * sigma**2))
    radius = int(sigma * 2)
    in_window = (
        (xs[None, :, :] >= (pixel_u[:, None, None] - radius))
        & (xs[None, :, :] <= (pixel_u[:, None, None] + radius))
        & (ys[None, :, :] >= (pixel_v[:, None, None] - radius))
        & (ys[None, :, :] <= (pixel_v[:, None, None] + radius))
    )
    in_bounds = (
        (pixel_u - radius >= 0)
        & (pixel_u + radius + 1 < image_w)
        & (pixel_v - radius >= 0)
        & (pixel_v + radius + 1 < image_h)
    )
    # Match DREAM: avoid malformed clipped Gaussians near edges, at the cost of
    # throwing away supervision for those edge-near keypoints.
    mask = visible[:, None, None]
    mask = mask & in_bounds[:, None, None] & in_window
    return jnp.where(mask, heatmaps, jnp.zeros_like(heatmaps))


def belief_sigma(sigma_pct: float, out_h: int, out_w: int) -> float:
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid belief-map size: {(out_h, out_w)}")
    return sigma_pct * min(out_h, out_w) / 100.0


def build_heatmaps(
    kp2d: jax.Array,
    visible: jax.Array,
    image_h: int,
    image_w: int,
    sigma: float = 2.0,
) -> jax.Array:
    return jax.vmap(lambda uv, vis: _build_heatmaps_one(uv, vis, image_h=image_h, image_w=image_w, sigma=sigma))(
        kp2d, visible
    )


def keypoint_metrics(batch: dict, pred_heatmaps: jax.Array):
    pred_uv, pred_conf = extract_keypoints(pred_heatmaps)
    _, _, out_h, out_w = pred_heatmaps.shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *batch["image"].shape[1:3])
    gt_uv = batch["keypoints_2d_netin"]
    vis = batch["keypoints_visible"]
    err = jnp.linalg.norm(pred_uv - gt_uv, axis=-1)
    vis_f = vis.astype(jnp.float32)
    denom = jnp.maximum(vis_f.sum(), 1.0)
    mean_px = (err * vis_f).sum() / denom
    pck_5 = ((err < 5.0).astype(jnp.float32) * vis_f).sum() / denom
    pck_10 = ((err < 10.0).astype(jnp.float32) * vis_f).sum() / denom
    thresholds = jnp.linspace(0.0, 20.0, 100, dtype=jnp.float32)
    pck = ((err[..., None] < thresholds).astype(jnp.float32) * vis_f[..., None]).sum(axis=(0, 1)) / denom
    pck_auc_20 = (((pck[1:] + pck[:-1]) * 0.5).sum() * (thresholds[1] - thresholds[0])) / thresholds[-1]
    return {
        "mean_px": mean_px,
        "pck_5": pck_5,
        "pck_10": pck_10,
        "pck_auc_20": pck_auc_20,
        "conf_mean": pred_conf.mean(),
    }


def focal_heatmap_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet-style focal loss for Gaussian keypoint belief maps.

    See Objects as Points: https://arxiv.org/abs/1904.07850.
    The peak pixel is positive. Every non-peak pixel is negative, but
    (1 - target)^beta makes pixels near the Gaussian peak weak negatives
    instead of punishing them like background.
    """
    pred = jnp.clip(pred, eps, 1.0 - eps)

    peak = target.max(axis=(-2, -1), keepdims=True)
    pos = (target >= peak) & (peak > 0)
    neg = ~pos

    pos_loss = -((1.0 - pred) ** alpha) * jnp.log(pred) * pos
    neg_loss = -((1.0 - target) ** beta) * (pred**alpha) * jnp.log(1.0 - pred) * neg

    n_pos = jnp.maximum(pos.sum(), 1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def mask_loss(pred: jax.Array, target: jax.Array, eps: float = 1e-6) -> tuple[jax.Array, dict[str, jax.Array]]:
    target = target.astype(pred.dtype)
    bce = -(target * jnp.log(pred + eps) + (1.0 - target) * jnp.log1p(-pred + eps)).mean()
    inter = (pred * target).sum(axis=(-2, -1))
    denom = pred.sum(axis=(-2, -1)) + target.sum(axis=(-2, -1))
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()
    return bce + dice, {"mask_bce": bce, "mask_dice": dice}


def dream_loss_fn(batch: dict, out_dict: dict, sigma_pct: float = 1.0, mask_weight: float = 0.1):
    pred = out_dict["pred_heatmaps"]
    preds = pred if isinstance(pred, tuple) else (pred,)
    final_pred = preds[-1]
    _, _, out_h, out_w = final_pred.shape
    uv = _denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w)
    vis = batch["keypoints_visible"]
    target = build_heatmaps(
        uv,
        vis,
        image_h=out_h,
        image_w=out_w,
        sigma=belief_sigma(sigma_pct, out_h, out_w),
    )
    stage_losses = tuple(focal_heatmap_loss(stage_pred, target) for stage_pred in preds)
    stage_mses = tuple(jnp.mean((stage_pred - target) ** 2) for stage_pred in preds)
    heatmap_loss = jnp.mean(jnp.stack(stage_losses))
    loss = heatmap_loss
    metrics = {
        "loss": loss,
        "heatmap_loss": heatmap_loss,
        "mse": jnp.mean(jnp.stack(stage_mses)),
        "visible_kp": vis.sum(),
        **{f"stage_{i + 1}_mse": stage_mse for i, stage_mse in enumerate(stage_mses)},
        **{f"stage_{i + 1}_focal": stage_loss for i, stage_loss in enumerate(stage_losses)},
        **keypoint_metrics(batch, final_pred),
    }
    if mask_weight > 0.0 and "pred_mask" in out_dict:
        mask_term, mask_metrics = mask_loss(out_dict["pred_mask"], mask_target(batch["mask"], out_h, out_w))
        loss = heatmap_loss + mask_weight * mask_term
        metrics = {
            **metrics,
            "loss": loss,
            "mask_loss": mask_term,
            "mask_weighted_loss": mask_weight * mask_term,
            **mask_metrics,
        }
    return loss, metrics


def _gaussian_kernel2d(sigma: float = KP_SMOOTH_SIGMA, radius: int = KP_SMOOTH_RADIUS) -> jax.Array:
    xs = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(xs**2) / (2.0 * sigma**2))
    k = k / k.sum()
    return jnp.outer(k, k)


def _smooth_heatmap(hm: jax.Array) -> jax.Array:
    kernel = _gaussian_kernel2d()[:, :, None, None]
    return jax.lax.conv_general_dilated(
        hm[None, :, :, None],
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0, :, :, 0]


def _extract_keypoint_channel(hm: jax.Array) -> tuple[jax.Array, jax.Array]:
    hm = _smooth_heatmap(hm)
    h, w = hm.shape
    flat = hm.reshape(h * w)
    idx = jnp.argmax(flat)
    conf = flat[idx]
    ys = jnp.arange(h, dtype=jnp.float32)[:, None]
    xs = jnp.arange(w, dtype=jnp.float32)[None, :]
    weights = jnp.where(hm >= KP_CONF_THRESHOLD, jnp.maximum(hm, 0.0), 0.0)
    denom = weights.sum()
    denom_safe = jnp.maximum(denom, 1e-8)
    uv_sub = jnp.array([(weights * xs).sum() / denom_safe, (weights * ys).sum() / denom_safe], dtype=jnp.float32)
    uv_argmax = jnp.array([idx % w, idx // w], dtype=jnp.float32)
    return jnp.where(denom > 0, uv_sub, uv_argmax), conf


def _extract_keypoints_one(hm: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(_extract_keypoint_channel)(hm)


def extract_keypoints(pred_heatmaps: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(_extract_keypoints_one)(pred_heatmaps)


def _transform_points(w2c: np.ndarray, pts_3d: np.ndarray) -> np.ndarray:
    return (w2c[:3, :3] @ pts_3d.T).T + w2c[:3, 3]


def _project_points(w2c: np.ndarray, pts_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts_cam = _transform_points(w2c, pts_3d)
    pix = (K @ pts_cam.T).T
    return pix[:, :2] / np.maximum(pix[:, 2:3], 1e-8)


def _rot_err_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos = (np.trace(R_pred @ R_gt.T) - 1.0) * 0.5
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def _solve_pose_one(q, uv_px, conf, K) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    joints_rad = np.deg2rad(np.asarray(q[:7], dtype=np.float64))
    pts_3d = fk_keypoints(joints_rad)
    valid = np.isfinite(uv_px).all(axis=-1) & np.isfinite(conf) & (conf > KP_CONF_THRESHOLD)
    try:
        w2c = solve_pnp(pts_3d, uv_px, K, valid)
    except Exception:
        w2c = None
    return joints_rad, valid, w2c


def _pose_metrics_one(q, uv_px, conf, K, w2c_gt) -> dict:
    joints_rad, valid, w2c_pred = _solve_pose_one(q, uv_px, conf, K)
    pts_3d = fk_keypoints(joints_rad)
    out = {"valid_kp": float(valid.sum()), "success": 0.0}
    if w2c_pred is None:
        return out

    reproj = _project_points(w2c_pred, pts_3d, K)
    reproj_err = np.linalg.norm(reproj[valid] - uv_px[valid], axis=-1).mean()
    pred_cam = _transform_points(w2c_pred, pts_3d)
    gt_cam = _transform_points(w2c_gt, pts_3d)
    add_mm = np.linalg.norm(pred_cam - gt_cam, axis=-1).mean() * 1000.0
    return {
        **out,
        "success": 1.0,
        "reproj_px": float(reproj_err),
        "add_mm": float(add_mm),
        "rot_err_deg": _rot_err_deg(w2c_pred[:3, :3], w2c_gt[:3, :3]),
        "trans_err_mm": float(np.linalg.norm(w2c_pred[:3, 3] - w2c_gt[:3, 3]) * 1000.0),
    }


def pose_metrics(cfg: Config, batch: dict, out_dict: dict) -> dict:
    pred_uv, conf = extract_keypoints(out_dict["pred_heatmaps"])
    _, _, out_h, out_w = out_dict["pred_heatmaps"].shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    batch_np = jax.device_get(batch)
    uv_np = np.asarray(jax.device_get(pred_uv), dtype=np.float64)
    conf_np = np.asarray(jax.device_get(conf), dtype=np.float64)
    q_np = np.asarray(batch_np["q"], dtype=np.float64)
    K_np = np.asarray(batch_np["K"], dtype=np.float64)
    w2c_np = np.asarray(batch_np["w2c"], dtype=np.float64)

    rows = [_pose_metrics_one(q_np[i], uv_np[i], conf_np[i], K_np[i], w2c_np[i]) for i in range(q_np.shape[0])]
    vals = {}
    for key in ("valid_kp", "success", "reproj_px", "add_mm", "rot_err_deg", "trans_err_mm"):
        xs = np.asarray([r[key] for r in rows if key in r], dtype=np.float32)
        vals[key] = float(xs.mean()) if len(xs) else float("nan")
    adds = np.asarray([r["add_mm"] for r in rows if "add_mm" in r], dtype=np.float32)
    if len(adds):
        curve = (adds[:, None] < ADD_THRESHOLDS_MM[None]).mean(axis=0)
        vals["add_auc_100mm"] = float(
            (((curve[1:] + curve[:-1]) * 0.5).sum() * (ADD_THRESHOLDS_MM[1] - ADD_THRESHOLDS_MM[0]))
            / ADD_THRESHOLDS_MM[-1]
        )
    else:
        vals["add_auc_100mm"] = float("nan")
    return vals


def _render_overlay(batch: dict, pred_uv: np.ndarray, pred_conf: np.ndarray, pred_heatmaps: np.ndarray, idx: int = 0):
    import matplotlib.pyplot as plt

    image = np.asarray(batch["image"][idx])
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)
    uv_gt = np.asarray(batch["keypoints_2d"][idx])
    vis = np.asarray(batch["keypoints_visible"][idx])
    uv_pred = np.asarray(pred_uv[idx])
    conf = np.asarray(pred_conf[idx])
    hm = np.asarray(pred_heatmaps[idx])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image)
    axes[0].scatter(uv_gt[vis, 0], uv_gt[vis, 1], c="lime", s=20, label="gt")
    axes[0].scatter(uv_pred[:, 0], uv_pred[:, 1], c="red", s=20, label="pred")
    axes[0].set_title("image + keypoints")
    axes[0].legend()
    axes[0].axis("off")

    axes[1].imshow(hm.max(axis=0), cmap="magma")
    axes[1].set_title("max heatmap")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(hm.max(axis=0), cmap="magma", alpha=0.5, extent=(0, image.shape[1], image.shape[0], 0))
    axes[2].set_title(f"overlay conf={conf.mean():.3f}")
    axes[2].axis("off")

    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def _render_heatmap_overlay(image: np.ndarray, hm: np.ndarray):
    import matplotlib.pyplot as plt

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)

    hm = np.asarray(hm)
    hm = hm.max(axis=0) if hm.ndim == 3 else hm
    vmax = max(float(hm.max(initial=0.0)), 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(
        hm,
        cmap="magma",
        alpha=0.5,
        vmin=0.0,
        vmax=vmax,
        extent=(0, image.shape[1], image.shape[0], 0),
    )
    ax.set_title("max gt belief map")
    ax.axis("off")
    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def _render_mask(mask: np.ndarray, idx: int = 0):
    mask = _mask_image(mask, idx=idx)
    return wandb.Image(mask)


def _mask_image(mask: np.ndarray, idx: int = 0) -> np.ndarray:
    mask = np.asarray(mask[idx])
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255
    elif mask.max(initial=0) <= 1:
        mask = (mask.astype(np.float32) * 255).astype(np.uint8)
    else:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask


def _render_mask_overlay(image: np.ndarray, mask: np.ndarray, idx: int = 0, title: str = "mask overlay"):
    import matplotlib.pyplot as plt

    image = np.asarray(image[idx] if image.ndim == 4 else image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)
    mask = _mask_image(mask, idx=idx).astype(np.float32) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(mask, cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def _image_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def _render_pose_overlay(batch: dict, pred_uv: np.ndarray, pred_conf: np.ndarray, idx: int = 0):
    import matplotlib.pyplot as plt

    image = _image_u8(batch["image"][idx])
    uv = np.asarray(pred_uv[idx], dtype=np.float64)
    conf = np.asarray(pred_conf[idx], dtype=np.float64)
    q = np.asarray(batch["q"][idx], dtype=np.float64)
    K = np.asarray(batch["K"][idx], dtype=np.float64)
    joints_rad, valid, w2c = _solve_pose_one(q, uv, conf, K)

    panel = image
    title = f"PnP failed ({valid.sum()} kp)"
    if w2c is not None:
        try:
            mask = rasterize_robot(joints_rad, w2c, K, image.shape[1], image.shape[0])
            panel = composite_robot(image, mask)
            title = f"pose overlay ({valid.sum()} kp)"
        except Exception as exc:
            title = f"raster failed: {type(exc).__name__}"

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(panel)
    ax.scatter(uv[valid, 0], uv[valid, 1], c="lime", s=12)
    ax.scatter(uv[~valid, 0], uv[~valid, 1], c="red", s=12)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def maybe_log_viz(cfg: Config, batch: dict, out_dict: dict, step: int, prefix: str = "viz"):
    if not cfg.wandb.use or cfg.viz.every <= 0 or step % cfg.viz.every != 0:
        return
    pred_heatmaps = final_pred_heatmaps(out_dict["pred_heatmaps"])
    pred_uv, pred_conf = extract_keypoints(pred_heatmaps)
    _, _, out_h, out_w = pred_heatmaps.shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    gt_uv = _denormalize_kp2d(batch["keypoints_2d_norm"], *cfg.net_in_size)
    gt_heatmaps = build_heatmaps(
        _denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w),
        batch["keypoints_visible"],
        image_h=out_h,
        image_w=out_w,
        sigma=belief_sigma(cfg.sigma_pct, out_h, out_w),
    )
    log = {
        f"{prefix}/predictions": _render_overlay(
            {
                "image": jax.device_get(batch["image"]),
                "keypoints_2d": jax.device_get(gt_uv),
                "keypoints_visible": jax.device_get(batch["keypoints_visible"]),
            },
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
            jax.device_get(pred_heatmaps),
        ),
        f"{prefix}/gt": _render_heatmap_overlay(jax.device_get(batch["image"][0]), jax.device_get(gt_heatmaps[0])),
        f"{prefix}/mask": _render_mask_overlay(
            jax.device_get(batch["image"]),
            jax.device_get(batch["mask"]),
            title="gt mask",
        ),
        f"{prefix}/pose_overlay": _render_pose_overlay(
            jax.device_get(batch),
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
        ),
    }
    if "pred_mask" in out_dict:
        log[f"{prefix}/pred_mask"] = _render_mask_overlay(
            jax.device_get(batch["image"]),
            jax.device_get(out_dict["pred_mask"]),
            title="pred mask",
        )
    cfg.wandb.log(log, step=step)


def resize_pred_heatmaps(pred_heatmaps: jax.Array, out_h: int, out_w: int) -> jax.Array:
    if tuple(pred_heatmaps.shape[-2:]) == (out_h, out_w):
        return pred_heatmaps
    pred_heatmaps = jnp.transpose(pred_heatmaps, (0, 2, 3, 1))
    pred_heatmaps = jax.image.resize(
        pred_heatmaps,
        (pred_heatmaps.shape[0], out_h, out_w, pred_heatmaps.shape[-1]),
        method="bilinear",
    )
    return jnp.transpose(pred_heatmaps, (0, 3, 1, 2))


def resize_pred_mask(pred_mask: jax.Array, out_h: int, out_w: int) -> jax.Array:
    if tuple(pred_mask.shape[-2:]) == (out_h, out_w):
        return pred_mask
    pred_mask = jnp.transpose(pred_mask, (0, 2, 3, 1))
    pred_mask = jax.image.resize(
        pred_mask,
        (pred_mask.shape[0], out_h, out_w, pred_mask.shape[-1]),
        method="bilinear",
    )
    return jnp.transpose(pred_mask, (0, 3, 1, 2))


def mask_target(mask: jax.Array, out_h: int, out_w: int) -> jax.Array:
    mask = (mask > 0).astype(jnp.float32)
    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    if tuple(mask.shape[-2:]) == (out_h, out_w):
        return mask
    mask = jnp.transpose(mask, (0, 2, 3, 1))
    mask = jax.image.resize(mask, (mask.shape[0], out_h, out_w, mask.shape[-1]), method="nearest")
    return jnp.transpose(mask, (0, 3, 1, 2))


def _stage_belief_maps(model_out):
    if isinstance(model_out, dict):
        return (model_out["heatmaps"],)
    if hasattr(model_out, "shape"):
        return (model_out,)
    if not isinstance(model_out, tuple | list):
        raise TypeError(f"unexpected DREAM output type: {type(model_out)}")
    if not model_out:
        raise ValueError("DREAM returned no outputs")

    first = model_out[0]
    if isinstance(first, dict):
        return tuple(stage["heatmaps"] for stage in model_out)
    if hasattr(first, "shape"):
        return tuple(model_out)
    if isinstance(first, tuple | list):
        return tuple(stage[0] for stage in model_out)
    raise TypeError(f"unexpected DREAM stage output type: {type(first)}")


def _stage_masks(model_out):
    if isinstance(model_out, dict):
        return (model_out["mask"],) if "mask" in model_out else ()
    if not isinstance(model_out, tuple | list) or not model_out:
        return ()

    first = model_out[0]
    if isinstance(first, dict):
        return tuple(stage["mask"] for stage in model_out if "mask" in stage)
    return ()


def prepare_pred_heatmaps(model_out, out_h: int, out_w: int):
    preds = tuple(
        resize_pred_heatmaps(jnp.transpose(stage, (0, 3, 1, 2)), out_h, out_w)
        for stage in _stage_belief_maps(model_out)
    )
    return preds[0] if len(preds) == 1 else preds


def prepare_pred_mask(model_out, out_h: int, out_w: int):
    masks = tuple(
        resize_pred_mask(jnp.transpose(stage, (0, 3, 1, 2)), out_h, out_w) for stage in _stage_masks(model_out)
    )
    return masks[-1] if masks else None


def final_pred_heatmaps(pred_heatmaps):
    return pred_heatmaps[-1] if isinstance(pred_heatmaps, tuple) else pred_heatmaps


def predict_heatmap_out(model, params, batch: dict, out_h: int, out_w: int) -> dict[str, jax.Array]:
    model_out, _ = model.apply({"params": params}, _image_to_float(batch["image"]))
    out = {"pred_heatmaps": final_pred_heatmaps(prepare_pred_heatmaps(model_out, out_h, out_w))}
    pred_mask = prepare_pred_mask(model_out, out_h, out_w)
    if pred_mask is not None:
        out["pred_mask"] = pred_mask
    return out


def make_train_step_dream(model, loss_fn, out_h: int, out_w: int, lr_fn, param_norm_fn):
    @jax.jit
    def train_step(state, batch):
        image = _image_to_float(batch["image"])

        def _loss(params):
            model_out, _ = model.apply({"params": params}, image)
            pred_heatmaps = prepare_pred_heatmaps(model_out, out_h, out_w)
            out_dict = {"pred_heatmaps": pred_heatmaps}
            pred_mask = prepare_pred_mask(model_out, out_h, out_w)
            if pred_mask is not None:
                out_dict["pred_mask"] = pred_mask
            return loss_fn(batch, out_dict)

        (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
        update_info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": param_norm_fn(state.params),
            "learning_rate": lr_fn(state.step),
            **metrics,
        }
        state = state.replace(
            step=state.step + 1,
            params=optax.apply_updates(state.params, updates),
            opt_state=opt_state,
        )
        return state, update_info

    return train_step


def make_eval_step_dream(model, loss_fn, out_h: int, out_w: int):
    @jax.jit
    def eval_step(state, batch):
        image = _image_to_float(batch["image"])
        model_out, _ = model.apply({"params": state.params}, image)
        pred_heatmaps = prepare_pred_heatmaps(model_out, out_h, out_w)
        out_dict = {"pred_heatmaps": pred_heatmaps}
        pred_mask = prepare_pred_mask(model_out, out_h, out_w)
        if pred_mask is not None:
            out_dict["pred_mask"] = pred_mask
        loss, metrics = loss_fn(batch, out_dict)
        return {"loss": loss, **metrics}

    return eval_step


def _count_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def _count_trainable_params(params, frozen: tuple[str, ...]) -> int:
    flat = freeze(params).unfreeze() if hasattr(params, "unfreeze") else params
    flat = jax.tree_util.tree_flatten_with_path(flat)[0]
    n = 0
    for path, x in flat:
        key = ".".join(str(p.key) for p in path)
        if any(fnmatch(key, pattern) for pattern in frozen):
            continue
        n += x.size
    return n


def _print_shapes(shapes):
    table = Table("stage", "shape")
    for name, shape in shapes:
        table.add_row(name, str(shape))
    print(table)


def main(cfg: Config):
    timer = Timer()
    ndev = len(jax.devices())
    if cfg.bs % ndev != 0:
        raise ValueError(f"bs={cfg.bs} must be divisible by device_count={ndev}")
    ds = make_dataset(cfg)
    dsit = iter(ds)
    irl_dsit = iter(make_irl_dataset(cfg)) if cfg.wandb.use and cfg.viz.every > 0 else None
    batch = next(dsit)

    print(Rule("DREAM Prepared Sample", style="bold magenta"))
    pprint(spec(batch))
    run = cfg.wandb.initialize(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    init_rng = rng
    num_keypoints = cfg.num_keypoints or int(batch["keypoints_2d_norm"].shape[1])
    print(f"raw_size={cfg.raw_size} net_in_size={cfg.net_in_size} net_out_size={net_out_size(cfg)}")
    out_h, out_w = net_out_size(cfg)
    print(f"target_sigma={belief_sigma(cfg.sigma_pct, out_h, out_w):.3f} output px")
    model = make_model(cfg, num_keypoints)
    image = _image_to_float(batch["image"])
    params = model.init(init_rng, image)["params"]
    params = load_tips_params(cfg, params)
    frozen = frozen_keys(cfg)
    tx, lr_fn, param_norm_fn = cfg.optim.create(params, steps=cfg.steps, frozen_keys=frozen)
    print(Rule("optimizer", style="bold magenta"))
    print(f"  config: {cfg.optim.kwargs(cfg.steps, frozen_keys=frozen)}")
    print(f"  tx: {tx}")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    if cfg.save_dir is not None:
        save_dir = _save_path(cfg)
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir — checkpoints disabled[/]")

    loss_fn = lambda batch, out_dict: dream_loss_fn(
        batch, out_dict, sigma_pct=cfg.sigma_pct, mask_weight=cfg.mask_weight
    )
    train_step = make_train_step_dream(
        model, loss_fn, out_h=out_h, out_w=out_w, lr_fn=lr_fn, param_norm_fn=param_norm_fn
    )
    eval_step = make_eval_step_dream(model, loss_fn, out_h=out_h, out_w=out_w)

    model_out, shapes = model.apply({"params": state.params}, image)
    pred_heatmaps = prepare_pred_heatmaps(model_out, out_h, out_w)
    out_dict = {"pred_heatmaps": pred_heatmaps}
    pred_mask = prepare_pred_mask(model_out, out_h, out_w)
    if pred_mask is not None:
        out_dict["pred_mask"] = pred_mask
    _, init_metrics = loss_fn(batch, out_dict)

    print(Rule("DREAM Forward", style="bold magenta"))
    _print_shapes(shapes)
    print(f"params={_count_params(state.params):,}")
    print(f"trainable_params={_count_trainable_params(state.params, frozen):,}")
    final_pred = final_pred_heatmaps(out_dict["pred_heatmaps"])
    print(f"pred_heatmaps.shape={final_pred.shape}")
    if "pred_mask" in out_dict:
        print(f"pred_mask.shape={out_dict['pred_mask'].shape}")
    if tuple(final_pred.shape[-2:]) != (out_h, out_w):
        raise ValueError(
            f"expected net_out_size={(out_h, out_w)} from variant={cfg.variant}, got {tuple(final_pred.shape[-2:])}"
        )
    print(f"init_loss={float(init_metrics['loss']):.6f}")
    maybe_log_viz(cfg, batch, out_dict, step=0)
    if irl_dsit is not None:
        irl_batch = next(irl_dsit)
        irl_out = predict_heatmap_out(model, state.params, irl_batch, out_h, out_w)
        maybe_log_viz(cfg, irl_batch, irl_out, step=0, prefix="irl")

    print(Rule("DREAM Train Loop", style="bold magenta"))
    for step in tqdm(range(cfg.steps)):
        with timer("data"):
            batch = next(dsit)
        with timer("train_step"):
            state, metrics = train_step(state, batch)

        if step % cfg.log_every == 0:
            with timer("data"):
                eval_batch = next(dsit)
            with timer("eval_step"):
                eval_metrics = eval_step(state, eval_batch)
            with timer("pose"):
                model_out, _ = model.apply({"params": state.params}, _image_to_float(eval_batch["image"]))
                eval_out = {"pred_heatmaps": final_pred_heatmaps(prepare_pred_heatmaps(model_out, out_h, out_w))}
                pnp_metrics = pose_metrics(cfg, eval_batch, eval_out)
            times = {f"timer/{k}": v for k, v in timer.get_average_times().items()}
            cfg.wandb.log({"train": metrics, "eval": eval_metrics, "pose": pnp_metrics, **times}, step=step)
            print({**metrics, **eval_metrics, **pnp_metrics, **times})
        if cfg.viz.every > 0 and step % cfg.viz.every == 0:
            out_dict = predict_heatmap_out(model, state.params, batch, out_h, out_w)
            maybe_log_viz(cfg, batch, out_dict, step=step)
            if irl_dsit is not None:
                irl_batch = next(irl_dsit)
                irl_out = predict_heatmap_out(model, state.params, irl_batch, out_h, out_w)
                maybe_log_viz(cfg, irl_batch, irl_out, step=step, prefix="irl")
        if cfg.save_interval > 0 and (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback.save(_checkpoint_state(state), step + 1)

    if save_dir is not None:
        save_callback.save(_checkpoint_state(state), cfg.steps)
        save_callback.wait()
    if cfg.verbose:
        print(model.tabulate(init_rng, _image_to_float(batch["image"]), depth=2))
    cfg.wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
