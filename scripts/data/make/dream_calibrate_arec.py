from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from rich import print
import tyro

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record
from crossformer.data.geometry import shrink_crop_resolution
from crossformer.data.grain.datasets import MultiArrayRecordSource
from crossformer.utils.rig import K_for_size
from crossformer.utils.spatial.calibration import solve_stacked_pnp
from crossformer.utils.spatial.kp import fk_keypoints
from scripts.serve.dream import (
    _extrinsics_from_keypoints,
    _shrink_crop_mask_batch,
    DreamPolicy,
)
from scripts.serve.dream import (
    Config as DreamServeConfig,
)


@dataclass
class Config:
    """Run DREAM on an Arec and write camera calibration artifacts."""

    name: str
    version: str
    branch: str = "main"
    root: Path = Path("~/.cache/arrayrecords")
    checkpoint: Path | None = None
    step: int | None = None
    out_dir: Path = Path("data/dream_calib")
    cams: tuple[str, ...] = ("side",)
    n_frames: int = 8
    n_candidates: int = 128
    start: int = 0
    stop: int | None = None
    chunk: int = 1
    focal_px: float = 515.0
    kp_conf_threshold: float = 0.01
    save_images: bool = False
    dream: DreamServeConfig = field(default_factory=lambda: DreamServeConfig(warmup=False))


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


def _image(sample: dict, cam: str) -> np.ndarray:
    image = sample["image"]
    img = np.asarray(image[cam] if isinstance(image, dict) else image)
    return img[0] if img.ndim == 4 else img


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


def _uncrop_keypoints(kp_net: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    (crop_h, crop_w), (top, left) = shrink_crop_resolution(raw_h, raw_w, net_h, net_w)
    out = np.asarray(kp_net, dtype=np.float32).copy()
    out[..., 0] = out[..., 0] / net_w * crop_w + left
    out[..., 1] = out[..., 1] / net_h * crop_h + top
    return out


def _payload(samples: list[dict], cam: str, cfg: Config) -> dict:
    images = np.stack([_image(s, cam) for s in samples], axis=0)
    q = np.stack([_q(s) for s in samples], axis=0)
    raw_h, raw_w = images.shape[1:3]
    K = np.repeat(K_for_size(raw_h, raw_w, f=cfg.focal_px)[None], len(samples), axis=0)
    payload = {"image": images, "q": q, "K": K.astype(np.float32), "calibrate": True}
    masks = [_mask(s, cam) for s in samples]
    if all(m is not None for m in masks):
        payload["mask"] = np.stack(masks, axis=0)
    return payload


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _save_artifact(path: Path, payload: dict, out: dict, idxs: np.ndarray, cam: str, cfg: Config) -> dict:
    images = np.asarray(payload["image"])
    raw_h, raw_w = images.shape[1:3]
    net_h, net_w = cfg.dream.dream.net_in_size
    keypoints_netin = np.asarray(out["keypoints"], dtype=np.float32)
    keypoints_raw = _uncrop_keypoints(keypoints_netin, raw_h, raw_w, net_h, net_w)

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
        "pnp_success": np.asarray(out["pnp_success"], dtype=bool),
        "pnp_valid": np.asarray(out["pnp_valid"], dtype=bool),
        "pnp_reproj_px": np.asarray(out["pnp_reproj_px"], dtype=np.float32),
        "mask_iou": np.asarray(out["mask_iou"], dtype=np.float32),
        "calib_w2c": np.asarray(out["calib_w2c"], dtype=np.float32),
        "calib_success": np.asarray(out["calib_success"], dtype=bool),
        "calib_valid": np.asarray(out["calib_valid"], dtype=bool),
        "calib_reproj_px": np.asarray(out["calib_reproj_px"], dtype=np.float32),
        "calib_n_points": np.asarray(out["calib_n_points"], dtype=np.int32),
    }
    if "mask" in out:
        data["pred_mask"] = np.asarray(out["mask"], dtype=np.float32)
    if cfg.save_images:
        data["image"] = images

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return {
        "cam": cam,
        "path": str(path),
        "success": bool(out["calib_success"]),
        "n_points": int(out["calib_n_points"]),
        "reproj_px": float(out["calib_reproj_px"]),
        "pnp_success_rate": float(np.asarray(out["pnp_success"], dtype=np.float32).mean()),
    }


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
    if "mask" in payload:
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

    q = np.asarray(payload_net["q"], dtype=np.float64)
    pts_3d = np.stack([fk_keypoints(np.deg2rad(q_i[:7])) for q_i in q], axis=0)
    valid = np.isfinite(confidence) & (confidence > cfg.kp_conf_threshold)
    calib = solve_stacked_pnp(pts_3d, keypoints, np.asarray(out["K"])[0], valid)
    out["calib_w2c"] = (
        calib.w2c.astype(np.float32) if calib.w2c is not None else np.full((4, 4), np.nan, dtype=np.float32)
    )
    out["calib_success"] = calib.success
    out["calib_valid"] = calib.valid
    out["calib_reproj_px"] = np.float32(calib.reproj_px)
    out["calib_n_points"] = np.int32(calib.n_points)
    return out


def main(cfg: Config) -> None:
    if cfg.checkpoint is not None:
        cfg.dream.path = cfg.checkpoint
    if cfg.step is not None:
        cfg.dream.step = cfg.step
    cfg.dream.warmup = False
    cfg.dream.ret.calibration = True
    cfg.dream.ret.mask = True
    cfg.dream.ret.heatmaps = True

    src = _open_source(cfg)
    idxs = _select_diverse(src, cfg)
    samples = [_read(src, int(i)) for i in idxs]
    policy = DreamPolicy(cfg.dream)

    root = cfg.out_dir.expanduser() / cfg.name / cfg.version / cfg.branch
    summary = {
        "name": cfg.name,
        "version": cfg.version,
        "branch": cfg.branch,
        "idx": idxs.tolist(),
        "cameras": [],
    }
    for cam in cfg.cams:
        payload = _payload(samples, cam, cfg)
        out = _postprocess_dream(payload, policy.step(payload), cfg)
        path = root / f"{cam}.npz"
        row = _save_artifact(path, payload, out, idxs, cam, cfg)
        summary["cameras"].append(row)
        print(row)

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
