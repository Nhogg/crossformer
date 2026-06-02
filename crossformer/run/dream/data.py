from __future__ import annotations

from functools import partial
import os

import grain
from grain.experimental import ThreadPrefetchIterDataset
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
import numpy as np
from PIL import Image

from crossformer.data.geometry import (
    normalize_kp2d_np,
    opencv_w2c_np,
    shrink_crop_image_np,
    shrink_crop_intrinsics_np,
    shrink_crop_keypoints_np,
)
from crossformer.data.grain.datasets import MultiArrayRecordSource, unpack_record
from crossformer.data.grain.loader import _apply_fd_limit, _grain_mp_worker_init
from crossformer.utils.callbacks.synth_viz import fk_keypoints
from crossformer.utils.rig import K_for_size, load_w2c, render_robot_mask

from .augment import _maybe_apply_grain_imaug
from .config import Config
from .constants import SOURCE_REAL, SOURCE_SYNTH


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
    if not isinstance(s, dict):
        return s
    for v in s.values():
        if isinstance(v, dict):
            for cam in _DROPPED_CAMS:
                v.pop(cam, None)
    return s


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


def make_irl_dataset(cfg: Config):
    ds = grain.MapDataset.source(cfg.irl_mix.source).seed(cfg.seed).repeat()
    ds = ds.map(_drop_low_cams)
    ds = ds.map(partial(prepare_irl_sample_np, cfg))
    ds = ds.batch(cfg.bs, drop_remainder=True)
    ds = ds.map(make_shard_fn())
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cfg.n_preshard)
    return ds


def make_shard_fn():
    mesh = Mesh(jax.devices(), axis_names="batch")

    def shard_batch(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    return shard_batch


def _real_mask_raw_np(
    sample: dict, image_key: str, joints_rad: np.ndarray, w2c: np.ndarray, K: np.ndarray
) -> np.ndarray:
    if "mask" in sample and image_key in sample["mask"]:
        mask_raw = np.asarray(sample["mask"][image_key])
        return mask_raw[0] if mask_raw.ndim == 3 else mask_raw

    gripper_val = float(np.asarray(sample["proprio"]["gripper"], dtype=np.float32).reshape(-1)[0])
    gripper_drive = (1.0 - gripper_val) * 0.85
    raw_h, raw_w = sample["image"][image_key].shape[-3:-1]
    return render_robot_mask(joints_rad, w2c, K, raw_h, raw_w, gripper_rad=gripper_drive)


def _project_points_np(w2c: np.ndarray, pts_3d: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = (w2c[:3, :3].astype(np.float64) @ pts_3d.T).T + w2c[:3, 3].astype(np.float64)
    pix = (K.astype(np.float64) @ pts_cam.T).T
    uv = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-8)
    return uv.astype(np.float32), pts_cam[:, 2].astype(np.float32)


def _real_kp2d_raw_np(
    sample: dict,
    image_key: str,
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    raw_h: int,
    raw_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "kp2d" in sample and image_key in sample["kp2d"]:
        kp2d_raw = np.asarray(sample["kp2d"][image_key], dtype=np.float32)
        if kp2d_raw.ndim == 3:
            kp2d_raw = kp2d_raw[0]
        visible = kp2d_raw[:, 2] > 0.5 if kp2d_raw.shape[-1] > 2 else np.isfinite(kp2d_raw).all(axis=-1)
        return kp2d_raw[:, :2], visible

    kp2d_raw, z_cam = _project_points_np(w2c, fk_keypoints(joints_rad.astype(np.float64)), K)
    visible = (
        (kp2d_raw[:, 0] >= 0.0)
        & (kp2d_raw[:, 0] < float(raw_w))
        & (kp2d_raw[:, 1] >= 0.0)
        & (kp2d_raw[:, 1] < float(raw_h))
        & (z_cam > 0.0)
    )
    return kp2d_raw, visible


def prepare_sample_np(cfg: Config, sample: dict) -> dict:
    raw_h, raw_w = cfg.raw_size
    net_h, net_w = cfg.net_in_size
    image = np.asarray(sample["image"])
    if tuple(image.shape[:2]) != (raw_h, raw_w):
        raise ValueError(f"expected raw_size={(raw_h, raw_w)} but got {tuple(image.shape[:2])}")
    image = shrink_crop_image_np(image, net_h, net_w, Image.BILINEAR)
    mask = shrink_crop_image_np(np.asarray(sample["mask"]), net_h, net_w, Image.NEAREST)
    joints = np.asarray(sample["state"]["joints"], dtype=np.float32)
    gripper = np.asarray([sample["state"]["gripper"]], dtype=np.float32)
    q = np.concatenate([joints, gripper], axis=-1)
    kp2d_raw = np.asarray(sample["state"]["kp2d"], dtype=np.float32)
    kp2d_netin = shrink_crop_keypoints_np(kp2d_raw, raw_h, raw_w, net_h, net_w)
    kp2d_norm = normalize_kp2d_np(kp2d_netin, net_h, net_w)
    K = np.asarray(sample["camera"]["intr"]["K"], dtype=np.float32)
    w2c = opencv_w2c_np(sample["camera"]["extr"]["w2c"])
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
        "K": shrink_crop_intrinsics_np(K, raw_h, raw_w, net_h, net_w),
        "w2c": w2c,
        "source": SOURCE_SYNTH,
    }


def prepare_irl_sample_np(cfg: Config, sample: dict) -> dict:
    raw_h, raw_w = cfg.raw_size
    net_h, net_w = cfg.net_in_size
    image_key = str(np.random.choice(cfg.irl_image_keys))
    if image_key in _DROPPED_CAMS:
        raise ValueError(f"irl_image_keys={cfg.irl_image_keys} contains dropped cam {image_key!r}")

    image_raw = np.asarray(sample["image"][image_key])
    if image_raw.ndim == 4:
        image_raw = image_raw[0]
    if tuple(image_raw.shape[:2]) != (raw_h, raw_w):
        raise ValueError(f"expected raw_size={(raw_h, raw_w)} but got {tuple(image_raw.shape[:2])}")

    joints_rad = np.asarray(sample["proprio"]["joints"], dtype=np.float32).reshape(-1, 7)[0]
    joints = np.rad2deg(joints_rad)
    gripper_arr = np.asarray(sample["proprio"]["gripper"], dtype=np.float32).reshape(-1)
    gripper = gripper_arr[:1]

    K_raw = K_for_size(raw_h, raw_w)
    w2c = load_w2c(image_key)
    mask_raw = _real_mask_raw_np(sample, image_key, joints_rad, w2c, K_raw)
    kp2d_raw, visible = _real_kp2d_raw_np(sample, image_key, joints_rad, w2c, K_raw, raw_h, raw_w)

    image = shrink_crop_image_np(image_raw, net_h, net_w, Image.BILINEAR)
    mask = shrink_crop_image_np(mask_raw, net_h, net_w, Image.NEAREST)
    kp2d_netin = shrink_crop_keypoints_np(kp2d_raw, raw_h, raw_w, net_h, net_w)
    kp2d_norm = normalize_kp2d_np(kp2d_netin, net_h, net_w)
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
        "keypoints_2d_raw": kp2d_raw,
        "keypoints_visible": visible,
        "K": shrink_crop_intrinsics_np(K_raw, raw_h, raw_w, net_h, net_w),
        "w2c": w2c.astype(np.float32),
        "source": SOURCE_REAL,
    }
