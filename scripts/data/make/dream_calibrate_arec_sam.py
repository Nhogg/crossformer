"""Calibrate camera extrinsics from an ArrayRecord dataset.

Loads DREAM locally, calls SAM3D server for masks, runs per-frame PnP,
and writes per-camera calibration artifacts + optional debug outputs.

Usage:
    python scripts/data/make/dream_calibrate_arec_sam.py \
        --name xgym_lift_single \
        --version 0.5.11 \
        --cams side \
        --checkpoint /path/to/dream/checkpoint \
        --sam-host 127.0.0.1 --sam-port 8080 \
        --debug-dir ~/debug_calib
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from rich import print
import tyro
import webpolicy.client as webpolicy_client
from webpolicy.client import Client

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record
from crossformer.data.geometry import shrink_crop_resolution
from crossformer.data.grain.datasets import MultiArrayRecordSource
from crossformer.utils.rig import K_for_size
from scripts.serve.dream import (
    _extrinsics_from_keypoints,
    _shrink_crop_mask_batch,
    DreamPolicy,
)
from scripts.serve.dream import Config as DreamServeConfig


@dataclass
class Config:
    """Calibrate camera extrinsics from an arec with DREAM + SAM3D."""

    name: str
    version: str
    branch: str = "main"
    root: Path = Path("~/.cache/arrayrecords")
    checkpoint: Path | None = None
    step: int | None = None
    out_dir: Path = Path("data/dream_calib")
    debug_dir: Path | None = None

    cams: tuple[str, ...] = ("side",)
    n_frames: int = 8
    n_candidates: int = 128
    start: int = 0
    stop: int | None = None
    chunk: int = 1
    focal_px: float = 515.0
    kp_conf_threshold: float = 0.001
    save_images: bool = False
    rasterize_all: bool = False  # rasterize with solved w2c on ALL frames in arec
    rasterize_max: int | None = None  # cap on how many frames to rasterize (None = all)
    export_npz: bool = False  # export per-frame npz files for dream_dr.py

    sam_host: str = "127.0.0.1"
    sam_port: int = 8080
    sam_prompt: str = "xArm robot arm with gripper"
    sam_confidence: float = 0.5
    sam_raw_webpolicy: bool = True

    dream: DreamServeConfig = field(default_factory=lambda: DreamServeConfig(warmup=False))


# ---------------------------------------------------------------------------
# arec helpers
# ---------------------------------------------------------------------------


def _open_source(cfg: Config):
    builder = ArrayRecordBuilder(
        name=cfg.name,
        version=cfg.version,
        branch=cfg.branch,
        root=str(cfg.root),
    )
    writers = builder.meta.get("writers", {})
    if writers:
        builder.writers = builder._normalize_writers(writers)
        builder.default_writer = "data" if "data" in builder.writers else next(iter(builder.writers))
    if {"image", "proprio"}.issubset(builder.writers):
        return MultiArrayRecordSource(
            builder.get_source("image"),
            builder.get_source("proprio"),
            chunk=cfg.chunk,
        )
    return builder.source


def _read(src, idx: int) -> dict:
    x = src[idx]
    return unpack_record(x) if isinstance(x, bytes) else x


def _cam_names(sample: dict) -> list[str]:
    keys = sample.get("info", {}).get("image_keys", None)
    if keys is None:
        image = sample.get("image")
        if isinstance(image, dict):
            return list(image.keys())
        return []
    flat = np.asarray(keys).reshape(-1)
    return [str(k).strip("\x00") for k in flat]


def _resolve_cam(cam: str, names: list[str]) -> int:
    """Find camera index by exact match or substring match."""
    for i, name in enumerate(names):
        if name == cam:
            return i
    for i, name in enumerate(names):
        if cam in name:
            return i
    raise KeyError(f"Camera {cam!r} not found. Available: {names}")


def _image(sample: dict, cam: str) -> np.ndarray:
    image = sample["image"]
    if isinstance(image, dict):
        return np.asarray(image[cam])
    img = np.asarray(image)
    if img.ndim == 4:
        names = _cam_names(sample)
        idx = _resolve_cam(cam, names)
        return img[idx]
    return img


def _mask(sample: dict, cam: str) -> np.ndarray | None:
    if "mask" not in sample:
        return None
    masks = sample["mask"]
    if isinstance(masks, dict):
        if cam not in masks:
            return None
        masks = masks[cam]
    mask = np.asarray(masks)
    return mask[0] if mask.ndim == 3 else mask


def _q(sample: dict) -> np.ndarray:
    joints = np.rad2deg(np.asarray(sample["proprio"]["joints"], dtype=np.float32).reshape(-1, 7)[0])
    grip = np.asarray(sample["proprio"].get("gripper", [0.0]), dtype=np.float32).reshape(-1)[:1]
    return np.concatenate([joints, grip], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# frame selection
# ---------------------------------------------------------------------------


def _candidate_idxs(n: int, cfg: Config) -> np.ndarray:
    stop = n if cfg.stop is None else min(cfg.stop, n)
    start = min(max(cfg.start, 0), stop - 1)
    n_cand = min(cfg.n_candidates, stop - start)
    return np.linspace(start, stop - 1, n_cand, dtype=np.int64)


def _select_diverse(src, cfg: Config) -> np.ndarray:
    idxs = _candidate_idxs(len(src), cfg)
    qs = np.stack([_q(_read(src, int(i)))[:7] for i in idxs], axis=0)
    keep = [0]
    while len(keep) < min(cfg.n_frames, len(idxs)):
        chosen = qs[keep]
        dist = np.linalg.norm(qs[:, None] - chosen[None], axis=-1).min(axis=1)
        dist[keep] = -1.0
        keep.append(int(np.argmax(dist)))
    return idxs[np.array(keep, dtype=np.int64)]


# ---------------------------------------------------------------------------
# SAM3D masks via webpolicy server
# ---------------------------------------------------------------------------


class _RawWebPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.uri = f"ws://{host}:{port}"
        self.ws = webpolicy_client.websockets.sync.client.connect(self.uri, compression=None, max_size=None)
        self.packer = webpolicy_client.msgpack_numpy.Packer()
        self.metadata = webpolicy_client.msgpack_numpy.unpackb(self.ws.recv())

    def step(self, obs: dict) -> dict:
        self.ws.send(self.packer.pack(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in SAM server:\n{response}")
        unpacked = webpolicy_client.msgpack_numpy.unpackb(response)
        if isinstance(unpacked, dict) and "action" in unpacked:
            return unpacked["action"]
        return unpacked


def _sam_mask_from_response(out: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = out.get("masks")
    if masks is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(masks)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        return np.zeros(shape, dtype=np.uint8)
    combined = np.any([(a > (0.5 if a.max() <= 1.0 else 0.0)) for a in arr], axis=0)
    mask = np.asarray(combined, dtype=np.uint8) * 255
    mask = np.ascontiguousarray(mask)
    if mask.shape != shape:
        mask = cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return mask


def _collect_sam_masks(cfg: Config, images: np.ndarray) -> np.ndarray:
    """images: (T, H, W, C) -> masks: (T, H, W) uint8."""
    client = (
        _RawWebPolicyClient(cfg.sam_host, cfg.sam_port) if cfg.sam_raw_webpolicy else Client(cfg.sam_host, cfg.sam_port)
    )
    masks = []
    for t in range(images.shape[0]):
        logging.info("SAM frame %d/%d", t + 1, images.shape[0])
        out = client.step(
            {
                "type": "image",
                "image": images[t],
                "text": cfg.sam_prompt,
                "confidence": cfg.sam_confidence,
            }
        )
        masks.append(_sam_mask_from_response(out, images[t].shape[:2]))
    return np.stack(masks).astype(np.uint8)


# ---------------------------------------------------------------------------
# DREAM per-frame PnP
# ---------------------------------------------------------------------------


def _payload(samples: list[dict], cam: str, masks: np.ndarray, cfg: Config) -> dict:
    images = np.stack([_image(s, cam) for s in samples], axis=0)
    q = np.stack([_q(s) for s in samples], axis=0)
    raw_h, raw_w = images.shape[1:3]
    K = np.repeat(K_for_size(raw_h, raw_w, f=cfg.focal_px)[None], len(samples), axis=0)
    return {
        "image": images,
        "q": q,
        "K": K.astype(np.float32),
        "mask": masks,
    }


def _uncrop_keypoints(kp_net: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    (crop_h, crop_w), (top, left) = shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    out = np.asarray(kp_net, dtype=np.float32).copy()
    out[..., 0] = out[..., 0] / net_w * crop_w + left
    out[..., 1] = out[..., 1] / net_h * crop_h + top
    return out


def _argmax_keypoints(heatmaps: np.ndarray, net_h: int, net_w: int) -> tuple[np.ndarray, np.ndarray]:
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, h * w)
    idx = np.argmax(flat, axis=-1)
    conf = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
    uv = np.stack([idx % w, idx // w], axis=-1).astype(np.float32)
    uv *= np.array([net_w / w, net_h / h], dtype=np.float32)
    return uv, conf


def _postprocess_dream(payload: dict, out: dict, cfg: Config) -> dict:
    if "heatmaps" not in out:
        return out

    net_h, net_w = cfg.dream.dream.net_in_size
    keypoints, confidence = _argmax_keypoints(out["heatmaps"], net_h, net_w)
    out = dict(out)
    out["keypoints"] = keypoints
    out["keypoints_norm"] = keypoints / np.array(cfg.dream.dream.net_in_size[::-1], dtype=np.float32)
    out["confidence"] = confidence

    payload_net = dict(payload)
    payload_net["K"] = out["K"]
    payload_net["mask"] = _shrink_crop_mask_batch(payload["mask"], net_h, net_w)

    out.update(
        _extrinsics_from_keypoints(
            payload_net,
            out,
            use_reject=cfg.dream.ret.use_reject,
            mask_iou_thresh=cfg.dream.ret.mask_iou_thresh,
            dr=cfg.dream.dr,
        )
    )

    return out


# ---------------------------------------------------------------------------
# save artifacts
# ---------------------------------------------------------------------------


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _save_artifact(
    path: Path, payload: dict, out: dict, sam_masks: np.ndarray, idxs: np.ndarray, cam: str, cfg: Config
) -> dict:
    images = np.asarray(payload["image"])
    raw_h, raw_w = images.shape[1:3]
    net_h, net_w = cfg.dream.dream.net_in_size
    keypoints_netin = np.asarray(out["keypoints"], dtype=np.float32)
    keypoints_raw = _uncrop_keypoints(keypoints_netin, raw_h, raw_w, net_h, net_w)

    pnp_success = np.asarray(out["pnp_success"], dtype=bool)
    data = {
        "idx": idxs.astype(np.int64),
        "cam": np.array(cam),
        "q": np.asarray(payload["q"], dtype=np.float32),
        "K_raw": np.asarray(payload["K"], dtype=np.float32),
        "K_netin": np.asarray(out["K"], dtype=np.float32),
        "keypoints_netin": keypoints_netin,
        "keypoints_raw": keypoints_raw,
        "confidence": np.asarray(out["confidence"], dtype=np.float32),
        "pnp_w2c": np.asarray(out["w2c"], dtype=np.float32),
        "pnp_success": pnp_success,
        "pnp_valid": np.asarray(out["pnp_valid"], dtype=bool),
        "pnp_reproj_px": np.asarray(out["pnp_reproj_px"], dtype=np.float32),
        "mask_iou": np.asarray(out["mask_iou"], dtype=np.float32),
        "sam_masks": sam_masks,
    }
    if "mask" in out:
        data["pred_mask"] = np.asarray(out["mask"], dtype=np.float32)
    if cfg.save_images:
        data["image"] = images

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)

    n_success = int(pnp_success.sum())
    return {
        "cam": cam,
        "path": str(path),
        "pnp_success_rate": float(pnp_success.astype(np.float32).mean()),
        "n_pnp_success": n_success,
    }


# ---------------------------------------------------------------------------
# debug outputs
# ---------------------------------------------------------------------------


def _uncrop_mask(mask_net: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    """Reverse the shrink-crop: resize network mask to crop size, place in raw canvas."""
    (crop_h, crop_w), (top, left) = shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    resized = cv2.resize(mask_net, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((raw_h, raw_w), dtype=mask_net.dtype)
    canvas[top : top + crop_h, left : left + crop_w] = resized
    return canvas


def _rasterize_w2c(payload: dict, out: dict, w2c: np.ndarray, cfg: Config, per_frame: bool = False) -> list[np.ndarray]:
    """Rasterize with a single w2c (4,4) or per-frame w2c (T,4,4).

    The w2c was solved against out["K"] (network-input intrinsic), so we
    rasterize at network-input resolution then uncrop to raw resolution.
    """
    from crossformer.utils.callbacks.synth_viz import rasterize_robot

    images = np.asarray(payload["image"])
    q = np.asarray(payload["q"], dtype=np.float64)
    raw_h, raw_w = images.shape[1:3]
    net_h, net_w = cfg.dream.dream.net_in_size
    K_net = np.asarray(out["K"][0] if np.asarray(out["K"]).ndim == 3 else out["K"], dtype=np.float64)
    rasters = []
    for t in range(len(q)):
        w2c_t = np.asarray(w2c[t] if per_frame else w2c, dtype=np.float64)
        joints_rad = np.deg2rad(q[t, :7])
        try:
            rast = rasterize_robot(joints_rad, w2c_t, K_net, net_w, net_h)
            rast = (np.asarray(rast) > 0.5).astype(np.uint8) * 255
            rast = _uncrop_mask(rast, raw_h, raw_w, net_h, net_w)
        except Exception:
            rast = np.zeros((raw_h, raw_w), dtype=np.uint8)
        rasters.append(rast)
    return rasters


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / max(union, 1)


def _write_debug(cfg: Config, cam: str, payload: dict, out: dict, sam_masks: np.ndarray) -> dict:
    d = cfg.debug_dir / cam
    images = np.asarray(payload["image"])
    T = images.shape[0]
    net_h, net_w = cfg.dream.dream.net_in_size
    raw_h, raw_w = images.shape[1:3]
    keypoints_netin = np.asarray(out.get("keypoints", []), dtype=np.float32)
    keypoints_raw = (
        _uncrop_keypoints(keypoints_netin, raw_h, raw_w, net_h, net_w) if keypoints_netin.size else keypoints_netin
    )
    confidence = np.asarray(out.get("confidence", []), dtype=np.float32)
    n_kp = keypoints_raw.shape[1] if keypoints_raw.ndim >= 2 else 0

    for t in range(T):
        img = images[t].copy()

        # SAM masks
        mask_dir = d / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_dir / f"{t:04d}_mask.png"), sam_masks[t])
        overlay = img.copy()
        region = sam_masks[t] > 0
        if np.any(region):
            layer = np.zeros_like(overlay)
            layer[:] = (0, 255, 0)
            overlay[region] = cv2.addWeighted(overlay[region], 0.45, layer[region], 0.55, 0.0)
        cv2.imwrite(str(mask_dir / f"{t:04d}_overlay.png"), overlay)

        # DREAM predicted mask
        if "mask" in out:
            pred_mask = np.asarray(out["mask"], dtype=np.float32)
            if t < pred_mask.shape[0]:
                pm = pred_mask[t]
                if pm.ndim > 2:
                    pm = pm.squeeze()
                pm = (pm > 0.5).astype(np.uint8) * 255
                pm = np.ascontiguousarray(pm)
                if pm.shape[:2] != (raw_h, raw_w):
                    pm = cv2.resize(pm, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
                pred_dir = d / "pred_masks"
                pred_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pred_dir / f"{t:04d}_pred_mask.png"), pm)

        # keypoints
        kp_dir = d / "keypoints"
        kp_dir.mkdir(parents=True, exist_ok=True)
        kp_img = img.copy()
        for j in range(n_kp):
            c = confidence[t, j] if confidence.ndim >= 2 else 0.0
            if c < cfg.kp_conf_threshold:
                continue
            x, y = round(keypoints_raw[t, j, 0]), round(keypoints_raw[t, j, 1])
            color = (0, 255, 255) if c >= 0.1 else (0, 128, 128)
            cv2.circle(kp_img, (x, y), 4, color, -1)
            cv2.putText(kp_img, f"{j}:{c:.2f}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.imwrite(str(kp_dir / f"{t:04d}.png"), kp_img)

    pnp_w2c = np.asarray(out.get("w2c"), dtype=np.float64) if "w2c" in out else None
    pnp_success = np.asarray(out.get("pnp_success", []), dtype=bool)

    if pnp_w2c is not None:
        rasters_dream = _rasterize_w2c(payload, out, pnp_w2c, cfg, per_frame=True)
        rast_dir = d / "raster_dream"
        rast_dir.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            if not (pnp_success.ndim and t < len(pnp_success) and pnp_success[t]):
                continue
            cv2.imwrite(str(rast_dir / f"{t:04d}_rast.png"), rasters_dream[t])
            overlay = images[t].copy()
            for mask_arr, color in [(sam_masks[t], (0, 255, 0)), (rasters_dream[t], (255, 0, 0))]:
                region = mask_arr > 0
                if np.any(region):
                    layer = np.zeros_like(overlay)
                    layer[:] = color
                    overlay[region] = cv2.addWeighted(overlay[region], 0.45, layer[region], 0.55, 0.0)
            cv2.imwrite(str(rast_dir / f"{t:04d}_overlay.png"), overlay)

    return _compute_iou_comparison(cfg, payload, out, sam_masks, cam)


# ---------------------------------------------------------------------------
# rasterize all frames
# ---------------------------------------------------------------------------


def _compute_iou_comparison(cfg: Config, payload: dict, out: dict, sam_masks: np.ndarray, cam: str) -> dict:
    """Compute IoU of SAM masks vs DREAM per-frame PnP rasterizations."""
    T = np.asarray(payload["image"]).shape[0]
    pnp_w2c = np.asarray(out["w2c"], dtype=np.float64) if "w2c" in out else None
    pnp_success = np.asarray(out.get("pnp_success", []), dtype=bool)

    stats = {"cam": cam}
    if pnp_w2c is not None:
        rasters_dream = _rasterize_w2c(payload, out, pnp_w2c, cfg, per_frame=True)
        dream_ious = [
            _mask_iou(sam_masks[t], rasters_dream[t])
            for t in range(T)
            if pnp_success.ndim and t < len(pnp_success) and pnp_success[t]
        ]
        stats["dream_iou"] = float(np.mean(dream_ious)) if dream_ious else float("nan")
        stats["n_valid"] = len(dream_ious)
    return stats


def _rasterize_all_frames(cfg: Config, src, cam: str, w2c: np.ndarray, K_netin: np.ndarray, root: Path) -> None:
    from crossformer.utils.callbacks.synth_viz import rasterize_robot

    n = len(src)
    cap = n if cfg.rasterize_max is None else min(cfg.rasterize_max, n)
    rast_dir = root / f"{cam}_rasterizations"
    rast_dir.mkdir(parents=True, exist_ok=True)
    net_h, net_w = cfg.dream.dream.net_in_size
    K_net = np.asarray(K_netin, dtype=np.float64)

    logging.info("Rasterizing %d frames for camera %s", cap, cam)
    for i in range(cap):
        sample = _read(src, i)
        img = _image(sample, cam).astype(np.uint8)
        q_deg = _q(sample)
        joints_rad = np.deg2rad(q_deg[:7].astype(np.float64))
        raw_h, raw_w = img.shape[:2]

        try:
            rast = rasterize_robot(joints_rad, w2c, K_net, net_w, net_h)
            rast_mask = (np.asarray(rast) > 0.5).astype(np.uint8) * 255
            rast_mask = _uncrop_mask(rast_mask, raw_h, raw_w, net_h, net_w)
        except Exception:
            logging.warning("Rasterization failed for frame %d", i)
            rast_mask = np.zeros((raw_h, raw_w), dtype=np.uint8)

        cv2.imwrite(str(rast_dir / f"{i:06d}_rast.png"), rast_mask)

        overlay = img.copy()
        region = rast_mask > 0
        if np.any(region):
            layer = np.zeros_like(overlay)
            layer[:] = (255, 0, 0)
            overlay[region] = cv2.addWeighted(overlay[region], 0.45, layer[region], 0.55, 0.0)
        cv2.imwrite(str(rast_dir / f"{i:06d}_overlay.png"), overlay)

        if (i + 1) % 500 == 0 or i == cap - 1:
            logging.info("Rasterized %d/%d frames", i + 1, cap)


# ---------------------------------------------------------------------------
# export npz for dream_dr.py
# ---------------------------------------------------------------------------


def _export_npz(cfg: Config, cam: str, payload: dict, out: dict, sam_masks: np.ndarray, idxs: np.ndarray) -> Path:
    """Export per-frame .npz files compatible with xclients/scripts/dream_dr.py."""
    export_dir = cfg.out_dir.expanduser() / cfg.name / cfg.version / cfg.branch / f"{cam}_npz"
    export_dir.mkdir(parents=True, exist_ok=True)

    images = np.asarray(payload["image"])
    q = np.asarray(payload["q"], dtype=np.float32)
    K_raw = np.asarray(payload["K"], dtype=np.float32)
    pnp_w2c = np.asarray(out.get("w2c"), dtype=np.float32) if "w2c" in out else None
    pnp_success = np.asarray(out.get("pnp_success", []), dtype=bool)

    for t in range(len(idxs)):
        data = {
            "image": images[t],
            "joints": np.deg2rad(q[t, :7]),
            "K": K_raw[t] if K_raw.ndim == 3 else K_raw,
        }
        if pnp_w2c is not None and pnp_success.ndim and t < len(pnp_success) and pnp_success[t]:
            data["w2c"] = pnp_w2c[t]
        np.savez(export_dir / f"{int(idxs[t]):06d}.npz", **data)

    mask_dir = export_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    for t in range(len(idxs)):
        cv2.imwrite(str(mask_dir / f"{int(idxs[t]):06d}_mask.png"), sam_masks[t])

    logging.info("Exported %d npz files to %s", len(idxs), export_dir)
    return export_dir


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    if cfg.checkpoint is not None:
        cfg.dream.path = cfg.checkpoint
    if cfg.step is not None:
        cfg.dream.step = cfg.step
    cfg.dream.warmup = False
    cfg.dream.ret.calibration = True
    cfg.dream.ret.mask = True
    cfg.dream.ret.heatmaps = True
    if cfg.debug_dir is not None:
        cfg.dream.ret.raster = True

    src = _open_source(cfg)
    n = len(src)
    logging.info("Opened arec %s/%s/%s: %d records", cfg.name, cfg.version, cfg.branch, n)

    sample0 = _read(src, 0)
    available_cams = _cam_names(sample0)
    logging.info("Available cameras: %s", available_cams)
    for cam in cfg.cams:
        if available_cams:
            _resolve_cam(cam, available_cams)

    idxs = _select_diverse(src, cfg)
    samples = [_read(src, int(i)) for i in idxs]
    T = len(samples)
    logging.info("Selected %d diverse frames from %d candidates", T, len(_candidate_idxs(n, cfg)))

    policy = DreamPolicy(cfg.dream)

    root = cfg.out_dir.expanduser() / cfg.name / cfg.version / cfg.branch
    summary = {
        "name": cfg.name,
        "version": cfg.version,
        "branch": cfg.branch,
        "idx": idxs.tolist(),
        "cameras": [],
    }

    cam_data = {}
    for cam in cfg.cams:
        logging.info("--- Camera: %s ---", cam)

        images = np.stack([_image(s, cam) for s in samples], axis=0).astype(np.uint8)

        # SAM3D masks
        record_masks = [_mask(s, cam) for s in samples]
        if all(m is not None for m in record_masks):
            sam_masks = np.stack(record_masks, axis=0).astype(np.uint8)
            logging.info("Using existing masks from arec")
        else:
            logging.info("Collecting SAM3D masks")
            sam_masks = _collect_sam_masks(cfg, images)

        payload = _payload(samples, cam, sam_masks, cfg)
        out = _postprocess_dream(payload, policy.step(payload), cfg)
        cam_data[cam] = (payload, out, sam_masks)

        path = root / f"{cam}.npz"
        row = _save_artifact(path, payload, out, sam_masks, idxs, cam, cfg)
        summary["cameras"].append(row)
        print(row)

        if cfg.debug_dir is not None:
            _write_debug(cfg, cam, payload, out, sam_masks)

        if cfg.export_npz:
            export_dir = _export_npz(cfg, cam, payload, out, sam_masks, idxs)
            print(f"  dream_dr.py input: --data-dir {export_dir}")

    if cfg.rasterize_all:
        for cam_row in summary["cameras"]:
            cam = cam_row["cam"]
            if cam_row["n_pnp_success"] == 0:
                logging.warning("Skipping rasterize_all for %s: no successful PnP frames", cam)
                continue
            npz = np.load(cam_row["path"], allow_pickle=False)
            pnp_w2c = npz["pnp_w2c"].astype(np.float64)
            pnp_success = npz["pnp_success"].astype(bool)
            best_t = int(np.where(pnp_success)[0][0])
            K_netin = npz["K_netin"][0] if npz["K_netin"].ndim == 3 else npz["K_netin"]
            _rasterize_all_frames(cfg, src, cam, pnp_w2c[best_t], K_netin, root)

    print("\n=== IoU (DREAM PnP vs SAM masks) ===")
    for cam in cfg.cams:
        payload, out, sam_masks = cam_data[cam]
        iou_stats = _compute_iou_comparison(cfg, payload, out, sam_masks, cam)
        dream_iou = iou_stats.get("dream_iou", float("nan"))
        n_valid = iou_stats.get("n_valid", 0)
        print(f"  {cam}: IoU = {dream_iou:.4f}  ({n_valid} valid frames)")

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    logging.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main(tyro.cli(Config))
