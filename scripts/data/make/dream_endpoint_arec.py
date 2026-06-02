from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from rich import print
from tqdm import tqdm
import tyro
from webpolicy.client import Client

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record
from crossformer.data.grain.datasets import MultiArrayRecordSource
from crossformer.utils.rig import K_for_size


@dataclass
class Config:
    """Send an Arec through a running DREAM endpoint and save responses."""

    name: str
    version: str
    branch: str = "main"
    root: Path = Path("~/.cache/arrayrecords")
    host: str = "127.0.0.1"
    port: int = 8002
    out_dir: Path = Path("data/dream_endpoint")
    cams: tuple[str, ...] = ("side",)
    batch_size: int = 32
    start: int = 0
    stop: int | None = None
    chunk: int = 1
    focal_px: float = 515.0
    include_mask: bool = True
    save_images: bool = False
    calibrate_batches: bool = False


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


def _payload(samples: list[dict], cam: str, cfg: Config) -> dict:
    images = np.stack([_image(s, cam) for s in samples], axis=0)
    raw_h, raw_w = images.shape[1:3]
    K = np.repeat(K_for_size(raw_h, raw_w, f=cfg.focal_px)[None], len(samples), axis=0)
    payload = {
        "image": images,
        "q": np.stack([_q(s) for s in samples], axis=0),
        "K": K.astype(np.float32),
    }
    if cfg.calibrate_batches:
        payload["calibrate"] = True
    if cfg.include_mask:
        masks = [_mask(s, cam) for s in samples]
        if all(m is not None for m in masks):
            payload["mask"] = np.stack(masks, axis=0)
    return payload


def _save_chunk(path: Path, idx: np.ndarray, payload: dict, out: dict, cfg: Config) -> None:
    data = {
        "idx": idx.astype(np.int64),
        "q": np.asarray(payload["q"], dtype=np.float32),
        "K_raw": np.asarray(payload["K"], dtype=np.float32),
    }
    for key, val in out.items():
        if isinstance(val, np.ndarray | np.generic | bool | int | float):
            data[key] = np.asarray(val)
    if cfg.save_images:
        data["image"] = np.asarray(payload["image"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _batches(start: int, stop: int, bs: int):
    for lo in range(start, stop, bs):
        hi = min(lo + bs, stop)
        yield np.arange(lo, hi, dtype=np.int64)


def main(cfg: Config) -> None:
    src = _open_source(cfg)
    start = max(cfg.start, 0)
    stop = len(src) if cfg.stop is None else min(cfg.stop, len(src))
    if start >= stop:
        raise ValueError(f"empty range: start={start} stop={stop}")

    client = Client(host=cfg.host, port=cfg.port)
    root = cfg.out_dir.expanduser() / cfg.name / cfg.version / cfg.branch
    summary = {
        "name": cfg.name,
        "version": cfg.version,
        "branch": cfg.branch,
        "host": cfg.host,
        "port": cfg.port,
        "start": start,
        "stop": stop,
        "batch_size": cfg.batch_size,
        "cameras": [],
    }

    for cam in cfg.cams:
        cam_dir = root / cam
        rows = []
        for bi, idx in enumerate(tqdm(list(_batches(start, stop, cfg.batch_size)), desc=f"DREAM {cam}")):
            samples = [_read(src, int(i)) for i in idx]
            payload = _payload(samples, cam, cfg)
            out = client.step(payload)
            path = cam_dir / f"chunk_{bi:06d}.npz"
            _save_chunk(path, idx, payload, out, cfg)
            rows.append(
                {
                    "chunk": bi,
                    "path": str(path),
                    "start": int(idx[0]),
                    "stop": int(idx[-1]) + 1,
                }
            )
        summary["cameras"].append({"cam": cam, "chunks": rows})

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
