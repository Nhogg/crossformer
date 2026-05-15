"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import partial
import os
from pathlib import Path

import augmax
import cv2
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
from crossformer.utils.callbacks.synth_viz import (
    composite_robot,
    fk_keypoints,
    rasterize_robot,
    solve_pnp,
)
from crossformer.utils.rig import K_for_size, load_w2c
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import create_optimizer, Timer
import wandb

KP_CONF_THRESHOLD = 0.03
KP_SMOOTH_SIGMA = 1.0
KP_SMOOTH_RADIUS = 2
KP_PEAK_THRESHOLD = 0.01
KP_PEAK_AMBIGUITY_GAP = 0.25
KP_MISSING_VALUE = -999.999
ADD_THRESHOLDS_MM = np.linspace(0.0, 100.0, 100, dtype=np.float32)


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
    irl_mix: Arec = default(Arec.from_name("xgym_lift_single"))
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


# Drop low cams at load time


# Image Augmentations


def _rotate_image_np(image: np.ndarray, angle_deg: float, resample: int, fill: int = 0) -> np.ndarray:
    out = Image.fromarray(image).rotate(angle_deg, resample=resample, expand=False, fillcolor=fill)
    return np.asarray(out)


def _rotate_keypoints_np(kp2d: np.ndarray, angle_deg: float, h: int, w: int) -> np.ndarray:
    a = np.deg2rad(np.float32(angle_deg))
    c, s = np.cos(a), np.sin(a)
    cx = np.float32(w) * 0.5
    cy = np.float32(h) * 0.5
    x = kp2d[..., 0] - cx
    y = kp2d[..., 1] - cy
    # PIL rotates CCW in screen / y-down coords: c*x + s*y, y' = -s*x + c*y
    return np.stack([c * x + s * y + cx, -s * x + c * y + cy], axis=-1).astype(np.float32)


_augmax_color_chain = augmax.Chain(
    augmax.ChannelShuffle(p=0.5),
    augmax.RandomGrayscale(p=0.5),
    augmax.ChannelDrop(),
    augmax.Blur(),
    augmax.RandomBrightness((-1.0, 1.0), p=0.5),
    augmax.RandomContrast(),
    augmax.RandomGamma(),
    augmax.RandomChannelGamma(),
    augmax.ColorJitter(),
    augmax.Solarization(),
)


def _apply_augmax_color(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply augmax color transforms to a single HWC uint8 image, returns uint8"""
    key = jax.random.key(rng.integers(2**32 - 1, dtype=np.uint32))
    img_f = jnp.asarray(image, dtype=jnp.float32) / 255.0
    img_f = _augmax_color_chain(key, img_f)
    return np.clip(np.asarray(img_f) * 255.0, 0, 255).astype(np.uint8)


@jax.jit
def _apply_augmax_color_batch(keys: jax.Array, images: jax.Array) -> jax.Array:
    """Apply augmax color transforms to a NHWC float32 batch, returns float32."""
    return jax.vmap(_augmax_color_chain)(keys, images)


def _translate_image_np(
    image: np.ndarray, tx: float, ty: float, resample: int = Image.BILINEAR, fill: int = 0
) -> np.ndarray:
    pil = Image.fromarray(image)
    pil = pil.transform(pil.size, Image.AFFINE, (1, 0, -tx, 0, 1, -ty), resample=resample, fillcolor=fill)
    return np.asarray(pil)


def _zoom_image_np(image: np.ndarray, scale: float, resample: int = Image.BILINEAR, fill: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    inv_s = 1.0 / scale
    pil = Image.fromarray(image)
    pil = pil.transform(
        pil.size,
        Image.AFFINE,
        (inv_s, 0, cx * (1 - inv_s), 0, inv_s, cy * (1 - inv_s)),
        resample=resample,
        fillcolor=fill,
    )
    return np.asarray(pil)


def _kp_in_bounds(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    return (kp2d[:, 0] >= 0.0) & (kp2d[:, 0] < np.float32(w)) & (kp2d[:, 1] >= 0.0) & (kp2d[:, 1] < np.float32(h))


def _kp_render_mask(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    kp2d = np.asarray(kp2d)
    return np.isfinite(kp2d).all(axis=-1) & _kp_in_bounds(kp2d, h, w)


def _plot_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
        return arr
    if arr.ndim == 3:
        return np.zeros((1, 1, arr.shape[2]), dtype=arr.dtype)
    return np.zeros((1, 1), dtype=arr.dtype)


def _maybe_apply_grain_imaug(ds, cfg: Config):
    if not (cfg.imaug or cfg.rotate):
        return ds

    def aug(batch: dict, rng):
        image = np.asarray(batch["image"]).copy()
        mask = np.asarray(batch["mask"]).copy()
        kp = np.asarray(batch["keypoints_2d_netin"], dtype=np.float32).copy()
        vis = np.asarray(batch["keypoints_visible"], dtype=bool).copy()
        K = np.asarray(batch["K"], dtype=np.float32).copy()
        w2c = np.asarray(batch["w2c"], dtype=np.float32).copy()
        h, w = image.shape[1:3]

        if cfg.imaug:
            base_key = jax.random.key(rng.integers(2**32 - 1, dtype=np.uint32))
            keys = jax.random.split(base_key, image.shape[0])
            imgs_f = jnp.asarray(image, dtype=jnp.float32) / 255.0
            image = np.clip(np.asarray(_apply_augmax_color_batch(keys, imgs_f)) * 255.0, 0, 255).astype(np.uint8)

        # Tentatively apply each transform and commit only if it leaves >= min_visible_kp
        # in-frame keypoints, so post-aug visibility respects the same floor as the
        # pre-batch dataset filter. Each commit also updates K / w2c so the stored
        # extrinsics stay self-consistent with the post-aug image (assumes K's
        # principal point is at image center, which holds after _shrink_crop_intrinsics_np).
        min_vis = cfg.min_visible_kp
        cx_img, cy_img = np.float32(w) * 0.5, np.float32(h) * 0.5
        for i in range(image.shape[0]):
            if cfg.rotate:
                if rng.random() < 0.3:
                    angle = float(rng.uniform(-15.0, 15.0))
                    kp_i = _rotate_keypoints_np(kp[i], angle, h=h, w=w)
                    vis_i = vis[i] & _kp_in_bounds(kp_i, h, w)
                    if int(vis_i.sum()) >= min_vis:
                        image[i] = _rotate_image_np(image[i], angle, Image.BILINEAR, fill=0)
                        mask[i] = _rotate_image_np(mask[i].astype(np.uint8), angle, Image.NEAREST, fill=0).astype(
                            mask.dtype
                        )
                        # Image rotation about its center = camera Rz rotation about the
                        # optical axis (because K's principal point sits at image center).
                        # PIL rotates pixels CCW for +angle; that corresponds to rotating
                        # the camera CW about +Z, i.e. Rz(-angle) on w2c from the left.
                        a = np.deg2rad(np.float32(-angle))
                        ca, sa = np.cos(a), np.sin(a)
                        Rz = np.eye(4, dtype=np.float32)
                        Rz[0, 0] = ca
                        Rz[0, 1] = -sa
                        Rz[1, 0] = sa
                        Rz[1, 1] = ca
                        w2c[i] = Rz @ w2c[i]
                        vis[i] = vis_i
                        kp[i] = kp_i

                if rng.random() < 0.5:
                    tx = float(rng.uniform(-0.1 * w, 0.1 * w))
                    ty = float(rng.uniform(-0.1 * h, 0.1 * h))
                    kp_i = kp[i].copy()
                    kp_i[:, 0] += tx
                    kp_i[:, 1] += ty
                    vis_i = vis[i] & _kp_in_bounds(kp_i, h, w)
                    if int(vis_i.sum()) >= min_vis:
                        image[i] = _translate_image_np(image[i], tx, ty, resample=Image.BILINEAR, fill=0)
                        mask[i] = _translate_image_np(
                            mask[i].astype(np.uint8), tx, ty, resample=Image.NEAREST, fill=0
                        ).astype(mask.dtype)
                        # Pixel translation = shift of K's principal point.
                        K[i, 0, 2] += np.float32(tx)
                        K[i, 1, 2] += np.float32(ty)
                        vis[i] = vis_i
                        kp[i] = kp_i

                if rng.random() < 0.5:
                    scale = float(rng.uniform(0.85, 1.15))
                    kp_i = kp[i].copy()
                    kp_i[:, 0] = (kp_i[:, 0] - cx_img) * scale + cx_img
                    kp_i[:, 1] = (kp_i[:, 1] - cy_img) * scale + cy_img
                    vis_i = vis[i] & _kp_in_bounds(kp_i, h, w)
                    if int(vis_i.sum()) >= min_vis:
                        image[i] = _zoom_image_np(image[i], scale, resample=Image.BILINEAR, fill=0)
                        mask[i] = _zoom_image_np(
                            mask[i].astype(np.uint8), scale, resample=Image.NEAREST, fill=0
                        ).astype(mask.dtype)
                        # Zoom-about-center: scale focal length; principal point unchanged
                        # because it already sits at image center.
                        s = np.float32(scale)
                        K[i, 0, 0] *= s
                        K[i, 1, 1] *= s
                        K[i, 0, 2] = (K[i, 0, 2] - cx_img) * s + cx_img
                        K[i, 1, 2] = (K[i, 1, 2] - cy_img) * s + cy_img
                        vis[i] = vis_i
                        kp[i] = kp_i

        return {
            **batch,
            "image": image,
            "mask": mask,
            "keypoints_2d_netin": kp,
            "keypoints_2d_norm": _normalize_kp2d_np(kp, h=h, w=w),
            "keypoints_visible": vis,
            "K": K,
            "w2c": w2c,
        }

    return ds.random_map(aug)


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


_DROPPED_CAMS = ("low",)


def _drop_low_cams(s):
    """Strip dropped cams from every per-cam dict in a sample"""
    if not isinstance(s, dict):
        return s
    for v in s.values():
        if isinstance(v, dict):
            for cam in _DROPPED_CAMS:
                v.pop(cam, None)
    return s


def prepare_irl_sample_np(cfg: Config, sample: dict) -> dict:
    """Prepare a real-data sample using pre-rendered mask + kp2d baked into the
    arec (xgym_sweep_single >= v0.6.0). No FK or rasterization at training time.
    """
    raw_h, raw_w = cfg.raw_size
    net_h, net_w = cfg.net_in_size
    image_key = str(np.random.choice(cfg.irl_image_keys))
    if image_key in _DROPPED_CAMS:
        raise ValueError(f"irl_image_keys={cfg.irl_image_keys} contains a dropped cam ({image_key}); see _DROPPED_CAMS")

    image_raw = np.asarray(sample["image"][image_key])
    if image_raw.ndim == 4:
        image_raw = image_raw[0]
    if tuple(image_raw.shape[:2]) != (raw_h, raw_w):
        raise ValueError(f"expected raw_size={(raw_h, raw_w)} but got {tuple(image_raw.shape[:2])}")

    mask_raw = np.asarray(sample["mask"][image_key])
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[0]

    kp2d_raw = np.asarray(sample["kp2d"][image_key], dtype=np.float32)  # (10, 3) u,v,vis
    if kp2d_raw.ndim == 3:
        kp2d_raw = kp2d_raw[0]
    visible = kp2d_raw[:, 2] > 0.5

    # xgym proprio stores joints in radians; the rest of the pipeline (model
    # proprio normalisation + every downstream `np.deg2rad(q[:7])` in metrics
    # and viz) is built around the synth convention of degrees. Convert here so
    # q[:7] is always degrees regardless of source.
    joints_arr = np.rad2deg(np.asarray(sample["proprio"]["joints"], dtype=np.float32).reshape(-1, 7))
    joints = joints_arr[0]
    gripper_arr = np.asarray(sample["proprio"]["gripper"], dtype=np.float32).reshape(-1)
    gripper = gripper_arr[:1]

    K_raw = K_for_size(raw_h, raw_w)
    w2c = load_w2c(image_key)

    image = _shrink_crop_image_np(image_raw, net_h, net_w, Image.BILINEAR)
    mask = _shrink_crop_image_np(mask_raw, net_h, net_w, Image.NEAREST)
    kp2d_netin = _shrink_crop_keypoints_np(kp2d_raw[:, :2], raw_h, raw_w, net_h, net_w)
    kp2d_norm = _normalize_kp2d_np(kp2d_netin, net_h, net_w)
    K = _shrink_crop_intrinsics_np(K_raw, raw_h, raw_w, net_h, net_w)

    # Visibility from prerender is raw-frame; intersect with the net-in crop so
    # the dataset filter and downstream metrics see in-frame-only visibility.
    in_crop = (
        (kp2d_netin[:, 0] >= 0.0)
        & (kp2d_netin[:, 0] < float(net_w))
        & (kp2d_netin[:, 1] >= 0.0)
        & (kp2d_netin[:, 1] < float(net_h))
    )
    visible = visible & in_crop

    return {
        "image": image,
        "mask": mask,
        "q": np.concatenate([joints, gripper], axis=-1),
        "keypoints_2d_norm": kp2d_norm,
        "keypoints_2d_netin": kp2d_netin,
        "keypoints_2d_raw": kp2d_raw[:, :2],
        "keypoints_visible": visible,
        "K": K,
        "w2c": w2c.astype(np.float32),
    }


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
        real = real.map(_drop_low_cams).map(lambda s: {**s, "_kind": "real"})
        md = grain.MapDataset.mix([synth, real], weights=[1.0 - cfg.real_prob, cfg.real_prob])
    else:
        md = synth

    ds = md.to_iter_dataset(grain.ReadOptions(num_threads=32, prefetch_buffer_size=1024))

    def _prepare(s):
        kind = s.pop("_kind")
        return prepare_sample_np(cfg, s) if kind == "synth" else prepare_irl_sample_np(cfg, s)

    ds = ds.map(_prepare)
    ds = ds.filter(lambda s: int(np.asarray(s["keypoints_visible"]).sum()) >= cfg.min_visible_kp)

    if cfg.coco_prob:
        coco = make_coco_dataset(cfg)
        ds = mix_with_bg(ds, coco, cfg)
    ds = ds.batch(cfg.bs, drop_remainder=True)
    ds = _maybe_apply_grain_imaug(ds, cfg)

    if cfg.mp > 0:
        lim = _apply_fd_limit(512**2)
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
    # Synth data is rendered in Mitsuba (camera convention: +X right, +Y up,
    # +Z forward, with projection x_img = cx - fx*x/z, y_img = cy - fy*y/z —
    # i.e. X and Y in camera-frame are flipped relative to OpenCV). Convert
    # Mitsuba camera-frame → OpenCV camera-frame: flip X and Y.
    w2c = np.asarray(w2c_raw, dtype=np.float32)
    flip = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
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
    # Intersect JSON visibility with crop-zone membership so that keypoints in the
    # left/right cropped-out columns are not treated as visible (they have negative
    # netin coords which cause scatter-plot dots to appear outside the image and
    # empty GT heatmap entries that destabilise the focal loss).
    in_crop = (
        (kp2d_netin[:, 0] >= 0.0)
        & (kp2d_netin[:, 0] < float(net_w))
        & (kp2d_netin[:, 1] >= 0.0)
        & (kp2d_netin[:, 1] < float(net_h))
    )
    kp_visible = np.asarray(sample["info"]["kp_visible"], dtype=bool) & in_crop
    return {
        "image": image,
        "mask": mask,
        "q": q,
        "keypoints_2d_norm": kp2d_norm,
        "keypoints_2d_netin": kp2d_netin,
        "keypoints_2d_raw": kp2d_raw,
        "keypoints_visible": kp_visible,
        "K": _shrink_crop_intrinsics_np(K, raw_h, raw_w, net_h, net_w),
        "w2c": w2c,
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
    in_bounds = (pixel_u >= 0) & (pixel_u < image_w) & (pixel_v >= 0) & (pixel_v < image_h)
    # Use the full Gaussian as the target so focal loss can suppress nearby negatives
    # via (1-target)^beta. in_bounds still drops edge keypoints with clipped Gaussians.
    mask = visible[:, None, None]
    mask = mask & in_bounds[:, None, None]
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
    local_max = jax.lax.reduce_window(
        hm,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(3, 3),
        window_strides=(1, 1),
        padding="SAME",
    )
    is_peak = (hm >= local_max) & (hm > KP_PEAK_THRESHOLD)
    peak_vals = jnp.where(is_peak.reshape(h * w), flat, -jnp.inf)
    best_idx = jnp.argmax(peak_vals)
    best = jnp.max(peak_vals)
    second_vals = peak_vals.at[best_idx].set(-jnp.inf)
    second = jnp.max(second_vals)
    has_peak = jnp.isfinite(best)
    unambiguous = has_peak & (~jnp.isfinite(second) | ((best - second) >= KP_PEAK_AMBIGUITY_GAP))
    uv = jnp.array([best_idx % w, best_idx // w], dtype=jnp.float32)
    missing = jnp.full((2,), KP_MISSING_VALUE, dtype=jnp.float32)
    return jnp.where(unambiguous, uv, missing), jnp.where(has_peak, best, 0.0)


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


PNP_MIN_VALID_KP = 5
PNP_REPROJ_THRESH = 30.0  # px — reject degenerate PnP solutions before rasterizing
PNP_MASK_IOU_THRESH = 0.45  # reject accepted PnP poses whose raster mask disagrees with available GT mask


def _solve_pnp_sqpnp_iter(pts_3d: np.ndarray, pts_2d_px: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    if pts_3d.shape[0] < 4:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts_3d.astype(np.float64),
            pts_2d_px.astype(np.float64),
            K.astype(np.float64),
            np.array([]),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if ok:
            ok, rvec, tvec = cv2.solvePnP(
                pts_3d.astype(np.float64),
                pts_2d_px.astype(np.float64),
                K.astype(np.float64),
                np.array([]),
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    return w2c


def _reprojection_errors(w2c: np.ndarray, pts_3d: np.ndarray, uv_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    reproj = _project_points(w2c, pts_3d, K)
    return np.linalg.norm(reproj - uv_px, axis=-1)


def _pnp_reproj_err(
    w2c: np.ndarray, joints_rad: np.ndarray, uv_px: np.ndarray, valid: np.ndarray, K: np.ndarray
) -> float:
    if not valid.any():
        return float("inf")
    pts_3d = fk_keypoints(joints_rad)
    reproj = _project_points(w2c, pts_3d, K)
    return float(np.linalg.norm(reproj[valid] - uv_px[valid], axis=-1).mean())


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary IoU between a float rasterized mask and a uint8 GT mask."""
    pred_bin = pred_mask > 0.5
    gt_bin = gt_mask > 0
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return float(inter / union) if union > 0 else float("nan")


def _solve_pose_one(q, uv_px, conf, K) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    joints_rad = np.deg2rad(np.asarray(q[:7], dtype=np.float64))
    pts_3d = fk_keypoints(joints_rad)
    valid = np.isfinite(uv_px).all(axis=-1) & (uv_px[:, 0] > -999.0) & np.isfinite(conf) & (conf > KP_CONF_THRESHOLD)
    if valid.sum() < PNP_MIN_VALID_KP:
        return joints_rad, valid, None
    w2c = _solve_pnp_sqpnp_iter(pts_3d[valid], uv_px[valid], K)
    if w2c is None:
        return joints_rad, valid, None

    errs = _reprojection_errors(w2c, pts_3d[valid], uv_px[valid], K)
    keep_local = errs <= PNP_REPROJ_THRESH
    if not keep_local.all():
        idx_valid = np.where(valid)[0]
        valid_refined = np.zeros_like(valid)
        valid_refined[idx_valid[keep_local]] = True
        if valid_refined.sum() < PNP_MIN_VALID_KP:
            return joints_rad, valid_refined, None
        w2c_refined = _solve_pnp_sqpnp_iter(pts_3d[valid_refined], uv_px[valid_refined], K)
        if w2c_refined is None:
            return joints_rad, valid_refined, None
        valid = valid_refined
        w2c = w2c_refined

    final_err = _pnp_reproj_err(w2c, joints_rad, uv_px, valid, K)
    if final_err > PNP_REPROJ_THRESH:
        w2c = None
    return joints_rad, valid, w2c


def _pose_metrics_one(q, uv_px, conf, K, kp2d_gt_px, kp_vis_gt, gt_mask: np.ndarray | None = None) -> dict:
    """Compare pred-PnP against GT-PnP (both self-consistent with FK frame).

    GT-PnP: solvePnP(FK 3D pts, stored kp2d_gt) → w2c that is FK-frame consistent.
    Pred-PnP: solvePnP(FK 3D pts, predicted 2D kp) → w2c to compare against.
    """
    joints_rad, valid, w2c_pred = _solve_pose_one(q, uv_px, conf, K)
    pts_3d = fk_keypoints(joints_rad)
    out = {"valid_kp": float(valid.sum()), "success": 0.0, "valid_mask": valid.astype(np.float32)}
    if w2c_pred is None:
        return out

    reproj = _project_points(w2c_pred, pts_3d, K)
    reproj_err = float(np.linalg.norm(reproj[valid] - uv_px[valid], axis=-1).mean())
    out = {**out, "success": 1.0, "reproj_px": reproj_err}

    # GT-PnP for reference (FK-frame consistent; Blender w2c in batch is NOT).
    vis_gt = np.asarray(kp_vis_gt, dtype=bool)
    w2c_gt = None
    if vis_gt.sum() >= 4:
        with contextlib.suppress(Exception):
            w2c_gt = solve_pnp(pts_3d[vis_gt], np.asarray(kp2d_gt_px, dtype=np.float64)[vis_gt], K)

    if w2c_gt is not None:
        pred_cam = _transform_points(w2c_pred, pts_3d)
        gt_cam = _transform_points(w2c_gt, pts_3d)
        add_mm = float(np.linalg.norm(pred_cam - gt_cam, axis=-1).mean() * 1000.0)
        out = {
            **out,
            "add_mm": add_mm,
            "rot_err_deg": _rot_err_deg(w2c_pred[:3, :3], w2c_gt[:3, :3]),
            "trans_err_mm": float(np.linalg.norm(w2c_pred[:3, 3] - w2c_gt[:3, 3]) * 1000.0),
        }

    if gt_mask is not None and reproj_err <= PNP_REPROJ_THRESH:
        h, w = gt_mask.shape[-2], gt_mask.shape[-1]
        try:
            gripper_rad = float(np.asarray(q)[7]) if np.asarray(q).shape[-1] > 7 else None
            rast_mask = rasterize_robot(joints_rad, w2c_pred, K, w, h, gripper_rad=gripper_rad)
            mask_iou = _mask_iou(rast_mask, gt_mask)
            out["mask_iou"] = mask_iou
            if np.isfinite(mask_iou) and mask_iou < PNP_MASK_IOU_THRESH:
                out["success"] = 0.0
                out["mask_iou_reject"] = 1.0
        except Exception:
            pass
    return out


def _pose_metrics_irl_one(q, uv_px, conf, K, gt_mask: np.ndarray) -> dict:
    """Like _pose_metrics_one but without GT w2c — for IRL where extrinsics are unknown.

    Mask IoU between the rasterized PnP estimate and the stored GT mask is the
    primary signal for whether the estimated extrinsics are correct.
    """
    joints_rad, valid, w2c_pred = _solve_pose_one(q, uv_px, conf, K)
    out = {"valid_kp": float(valid.sum()), "success": 0.0, "valid_mask": valid.astype(np.float32)}
    if w2c_pred is None:
        return out

    reproj_err = _pnp_reproj_err(w2c_pred, joints_rad, uv_px, valid, K)
    out = {**out, "success": 1.0, "reproj_px": reproj_err}
    if reproj_err <= PNP_REPROJ_THRESH:
        h, w = gt_mask.shape[-2], gt_mask.shape[-1]
        try:
            gripper_rad = float(np.asarray(q)[7]) if np.asarray(q).shape[-1] > 7 else None
            rast_mask = rasterize_robot(joints_rad, w2c_pred, K, w, h, gripper_rad=gripper_rad)
            mask_iou = _mask_iou(rast_mask, gt_mask)
            out["mask_iou"] = mask_iou
            if np.isfinite(mask_iou) and mask_iou < PNP_MASK_IOU_THRESH:
                out["success"] = 0.0
                out["mask_iou_reject"] = 1.0
        except Exception:
            pass
    return out


def pose_metrics(cfg: Config, batch: dict, out_dict: dict) -> dict:
    pred_uv, conf = extract_keypoints(out_dict["pred_heatmaps"])
    _, _, out_h, out_w = out_dict["pred_heatmaps"].shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    batch_np = jax.device_get(batch)
    uv_np = np.asarray(jax.device_get(pred_uv), dtype=np.float64)
    conf_np = np.asarray(jax.device_get(conf), dtype=np.float64)
    q_np = np.asarray(batch_np["q"], dtype=np.float64)
    K_np = np.asarray(batch_np["K"], dtype=np.float64)
    kp2d_gt_np = np.asarray(batch_np["keypoints_2d_netin"], dtype=np.float64)
    kp_vis_np = np.asarray(batch_np["keypoints_visible"])
    mask_np = np.asarray(batch_np["mask"])

    rows = [
        _pose_metrics_one(
            q_np[i],
            uv_np[i],
            conf_np[i],
            K_np[i],
            kp2d_gt_px=kp2d_gt_np[i],
            kp_vis_gt=kp_vis_np[i],
            gt_mask=mask_np[i],
        )
        for i in range(q_np.shape[0])
    ]
    vals = {}
    for key in (
        "valid_kp",
        "success",
        "reproj_px",
        "add_mm",
        "rot_err_deg",
        "trans_err_mm",
        "mask_iou",
        "mask_iou_reject",
    ):
        xs = np.asarray([r[key] for r in rows if key in r], dtype=np.float32)
        vals[key] = float(xs.mean()) if len(xs) else float("nan")
    valid_masks = np.asarray([r["valid_mask"] for r in rows if "valid_mask" in r], dtype=np.float32)
    if len(valid_masks):
        for i, rate in enumerate(valid_masks.mean(axis=0)):
            vals[f"kp{i}_kept"] = float(rate)
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


def pose_metrics_irl(cfg: Config, batch: dict, out_dict: dict) -> dict:
    """Extrinsics quality metrics for IRL batches (no GT w2c available).

    The primary signal is mask_iou: how well does the PnP-recovered rasterized
    robot silhouette match the GT pre-rendered mask stored in the batch.
    """
    pred_uv, conf = extract_keypoints(out_dict["pred_heatmaps"])
    _, _, out_h, out_w = out_dict["pred_heatmaps"].shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    batch_np = jax.device_get(batch)
    uv_np = np.asarray(jax.device_get(pred_uv), dtype=np.float64)
    conf_np = np.asarray(jax.device_get(conf), dtype=np.float64)
    q_np = np.asarray(batch_np["q"], dtype=np.float64)
    K_np = np.asarray(batch_np["K"], dtype=np.float64)
    mask_np = np.asarray(batch_np["mask"])

    rows = [
        _pose_metrics_irl_one(q_np[i], uv_np[i], conf_np[i], K_np[i], gt_mask=mask_np[i]) for i in range(q_np.shape[0])
    ]
    vals = {}
    for key in ("valid_kp", "success", "reproj_px", "mask_iou", "mask_iou_reject"):
        xs = np.asarray([r[key] for r in rows if key in r], dtype=np.float32)
        vals[key] = float(xs.mean()) if len(xs) else float("nan")
    valid_masks = np.asarray([r["valid_mask"] for r in rows if "valid_mask" in r], dtype=np.float32)
    if len(valid_masks):
        for i, rate in enumerate(valid_masks.mean(axis=0)):
            vals[f"kp{i}_kept"] = float(rate)
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
    h, w = image.shape[:2]
    gt_mask = vis & _kp_render_mask(uv_gt, h, w)
    pred_mask = _kp_render_mask(uv_pred, h, w)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(_plot_image(image))
    if gt_mask.any():
        axes[0].scatter(uv_gt[gt_mask, 0], uv_gt[gt_mask, 1], c="lime", s=20, label="gt")
    if pred_mask.any():
        axes[0].scatter(uv_pred[pred_mask, 0], uv_pred[pred_mask, 1], c="red", s=20, label="pred")
    axes[0].set_title("image + keypoints")
    if gt_mask.any() or pred_mask.any():
        axes[0].legend()
    axes[0].axis("off")

    axes[1].imshow(_plot_image(hm.max(axis=0)), cmap="magma")
    axes[1].set_title("max heatmap")
    axes[1].axis("off")

    axes[2].imshow(_plot_image(image))
    axes[2].imshow(_plot_image(hm.max(axis=0)), cmap="magma", alpha=0.5, extent=(0, image.shape[1], image.shape[0], 0))
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


def _render_gt_rast_overlay(batch: dict, idx: int = 0):
    """Render robot using the stored GT extrinsics (synth: Mitsuba w2c converted
    via _opencv_w2c_np; IRL: calibrated load_w2c). Both are FK-frame consistent,
    so we rasterize directly and avoid PnP's degeneracy with few visible kps.

    Panels:
    - GT-w2c rast composited on image (should overlay GT silhouette)
    - FK reprojection via GT-w2c vs stored kp2d (should match ~0 px)
    - GT mask for comparison
    """
    import matplotlib.pyplot as plt

    image = _image_u8(batch["image"][idx])
    q = np.asarray(batch["q"][idx], dtype=np.float64)
    K = np.asarray(batch["K"][idx], dtype=np.float64)
    gt_mask = np.asarray(batch["mask"][idx])
    kp2d_gt = np.asarray(batch["keypoints_2d_netin"][idx], dtype=np.float64)
    kp_vis = np.asarray(batch["keypoints_visible"][idx], dtype=bool)
    w2c_gt = np.asarray(batch["w2c"][idx], dtype=np.float64)
    joints_rad = np.deg2rad(q[:7])
    pts_3d = fk_keypoints(joints_rad)

    reproj_err = (
        float(np.linalg.norm(_project_points(w2c_gt, pts_3d, K) - kp2d_gt, axis=-1)[kp_vis].mean())
        if kp_vis.any()
        else float("nan")
    )

    panel = image.copy()
    rast_title = f"GT-w2c rast  reproj={reproj_err:.1f}px"
    try:
        gripper_rad = float(q[7]) if q.shape[-1] > 7 else None
        rast = rasterize_robot(joints_rad, w2c_gt, K, image.shape[1], image.shape[0], gripper_rad=gripper_rad)
        panel = composite_robot(image, rast)
        iou = _mask_iou(rast, gt_mask)
        rast_title = f"GT-w2c rast  IoU={iou:.2f}  reproj={reproj_err:.1f}px"
    except Exception as exc:
        rast_title = f"rast failed: {type(exc).__name__}  reproj={reproj_err:.1f}px"

    reproj_image = image.copy()
    try:
        reproj_px = _project_points(w2c_gt, pts_3d, K)
        reproj_title = f"GT-w2c reproj vs kp2d  err={reproj_err:.1f}px"
    except Exception:
        reproj_px = None
        reproj_title = "reproj failed"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_plot_image(panel))
    axes[0].set_title(rast_title)
    axes[0].axis("off")

    axes[1].imshow(_plot_image(reproj_image))
    if reproj_px is not None:
        gt_mask = kp_vis & _kp_render_mask(kp2d_gt, reproj_image.shape[0], reproj_image.shape[1])
        reproj_mask = kp_vis & _kp_render_mask(reproj_px, reproj_image.shape[0], reproj_image.shape[1])
        if gt_mask.any():
            axes[1].scatter(kp2d_gt[gt_mask, 0], kp2d_gt[gt_mask, 1], c="lime", s=20, label="stored kp2d")
        if reproj_mask.any():
            axes[1].scatter(
                reproj_px[reproj_mask, 0], reproj_px[reproj_mask, 1], c="red", s=20, marker="x", label="GT-PnP reproj"
            )
        if gt_mask.any() or reproj_mask.any():
            axes[1].legend(fontsize=6)
    axes[1].set_title(reproj_title)
    axes[1].axis("off")

    axes[2].imshow(_plot_image(gt_mask), cmap="gray")
    axes[2].set_title("GT mask")
    axes[2].axis("off")

    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def _render_extrinsics_comparison(batch: dict, pred_uv: np.ndarray, pred_conf: np.ndarray, max_samples: int = 16):
    """Error magnitude histograms: pred-PnP vs GT-PnP camera extrinsics.

    GT-PnP = solvePnP(FK 3D pts, stored kp2d) — FK-frame consistent.
    Pred-PnP = solvePnP(FK 3D pts, predicted 2D kp).
    Both histograms should shift left as training improves.
    Bird's-eye shows GT-PnP camera positions (FK world frame, robot base at origin).
    """
    import matplotlib.pyplot as plt

    B = min(len(batch["q"]), max_samples)
    trans_errs, rot_errs, gt_t = [], [], []
    for i in range(B):
        q = np.asarray(batch["q"][i], dtype=np.float64)
        uv = np.asarray(pred_uv[i], dtype=np.float64)
        conf = np.asarray(pred_conf[i], dtype=np.float64)
        K = np.asarray(batch["K"][i], dtype=np.float64)
        kp2d_gt = np.asarray(batch["keypoints_2d_netin"][i], dtype=np.float64)
        kp_vis = np.asarray(batch["keypoints_visible"][i], dtype=bool)

        joints_rad = np.deg2rad(np.asarray(q[:7], dtype=np.float64))
        pts_3d = fk_keypoints(joints_rad)

        # GT-PnP: self-consistent with FK frame.
        w2c_gt = None
        if kp_vis.sum() >= 4:
            with contextlib.suppress(Exception):
                w2c_gt = solve_pnp(pts_3d[kp_vis], kp2d_gt[kp_vis], K)
        if w2c_gt is not None:
            gt_t.append(w2c_gt[:3, 3])

        _, _, w2c_pred = _solve_pose_one(q, uv, conf, K)
        if w2c_pred is not None and w2c_gt is not None:
            trans_errs.append(np.linalg.norm(w2c_pred[:3, 3] - w2c_gt[:3, 3]) * 1000.0)
            rot_errs.append(_rot_err_deg(w2c_pred[:3, :3], w2c_gt[:3, :3]))

    gt_t_arr = np.array(gt_t) if gt_t else None
    trans_errs = np.array(trans_errs) if trans_errs else np.array([float("nan")])
    rot_errs = np.array(rot_errs) if rot_errs else np.array([float("nan")])
    n_success = len(trans_errs)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Translation error histogram (mm)
    valid_t = trans_errs[np.isfinite(trans_errs)]
    axes[0].hist(valid_t, bins=12, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Translation error (mm)")
    axes[0].set_ylabel("Count")
    mean_t = float(valid_t.mean()) if len(valid_t) else float("nan")
    axes[0].set_title(f"Trans err  mean={mean_t:.0f}mm  ({n_success}/{B} solved)")

    # Rotation error histogram (deg)
    valid_r = rot_errs[np.isfinite(rot_errs)]
    axes[1].hist(valid_r, bins=12, color="coral", edgecolor="white")
    axes[1].set_xlabel("Rotation error (deg)")
    axes[1].set_ylabel("Count")
    mean_r = float(valid_r.mean()) if len(valid_r) else float("nan")
    axes[1].set_title(f"Rot err  mean={mean_r:.1f}°")

    # Bird's-eye XY view of GT-PnP camera translations (shows dataset coverage).
    if gt_t_arr is not None:
        axes[2].scatter(gt_t_arr[:, 0], gt_t_arr[:, 1], c="green", s=30, alpha=0.8)
    else:
        axes[2].text(0.5, 0.5, "no GT-PnP solutions", ha="center", transform=axes[2].transAxes)
    axes[2].scatter([0], [0], c="black", s=60, marker="*", label="robot base")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    axes[2].set_title("GT camera positions (bird's-eye)")
    axes[2].legend(fontsize=7)
    axes[2].set_aspect("equal")

    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


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
        reproj_err = _pnp_reproj_err(w2c, joints_rad, uv, valid, K)
        if reproj_err > PNP_REPROJ_THRESH:
            title = f"PnP degenerate reproj={reproj_err:.0f}px ({valid.sum()} kp)"
        else:
            try:
                gripper_rad = float(q[7]) if q.shape[-1] > 7 else None
                mask = rasterize_robot(joints_rad, w2c, K, image.shape[1], image.shape[0], gripper_rad=gripper_rad)
                if "mask" in batch:
                    gt_mask = np.asarray(batch["mask"][idx])
                    mask_iou = _mask_iou(mask, gt_mask)
                    if np.isfinite(mask_iou) and mask_iou < PNP_MASK_IOU_THRESH:
                        title = f"PnP rejected IoU={mask_iou:.2f} reproj={reproj_err:.1f}px ({valid.sum()} kp)"
                    else:
                        panel = composite_robot(image, mask)
                        title = f"pose overlay IoU={mask_iou:.2f} reproj={reproj_err:.1f}px ({valid.sum()} kp)"
                else:
                    panel = composite_robot(image, mask)
                    title = f"pose overlay reproj={reproj_err:.1f}px ({valid.sum()} kp)"
            except Exception as exc:
                title = f"raster failed: {type(exc).__name__}"

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(_plot_image(panel))
    valid_mask = valid & _kp_render_mask(uv, image.shape[0], image.shape[1])
    invalid_mask = (~valid) & _kp_render_mask(uv, image.shape[0], image.shape[1])
    if valid_mask.any():
        ax.scatter(uv[valid_mask, 0], uv[valid_mask, 1], c="lime", s=12)
    if invalid_mask.any():
        ax.scatter(uv[invalid_mask, 0], uv[invalid_mask, 1], c="red", s=12)
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
    batch_np = jax.device_get(batch)
    log = {
        f"{prefix}/predictions": _render_overlay(
            {
                "image": batch_np["image"],
                "keypoints_2d": jax.device_get(gt_uv),
                "keypoints_visible": batch_np["keypoints_visible"],
            },
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
            jax.device_get(pred_heatmaps),
        ),
        f"{prefix}/gt": _render_heatmap_overlay(batch_np["image"][0], jax.device_get(gt_heatmaps[0])),
        f"{prefix}/mask": _render_mask_overlay(
            batch_np["image"],
            batch_np["mask"],
            title="gt mask",
        ),
        f"{prefix}/pose_overlay": _render_pose_overlay(
            batch_np,
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
        ),
    }
    # GT extrinsics panels: only meaningful for synth data (has w2c in batch).
    if "w2c" in batch_np:
        log[f"{prefix}/gt_rast"] = _render_gt_rast_overlay(batch_np)
        log[f"{prefix}/extr_compare"] = _render_extrinsics_comparison(
            batch_np, jax.device_get(pred_uv), jax.device_get(pred_conf)
        )
    if "pred_mask" in out_dict and prefix != "irl":
        log[f"{prefix}/pred_mask"] = _render_mask_overlay(
            batch_np["image"],
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


def make_irl_dataset(cfg: Config):
    ds = grain.MapDataset.source(cfg.irl_mix.source).seed(cfg.seed).repeat()
    ds = ds.map(_drop_low_cams)
    ds = ds.map(partial(prepare_irl_sample_np, cfg))
    ds = ds.batch(cfg.bs, drop_remainder=True)
    ds = ds.map(make_shard_fn())
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cfg.n_preshard)
    return ds


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
                with timer("irl_pose"):
                    irl_pose = pose_metrics_irl(cfg, irl_batch, irl_out)
                cfg.wandb.log({"irl_pose": irl_pose}, step=step)
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
