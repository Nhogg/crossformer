from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterator, Literal

import numpy as np
from PIL import Image
from rich import print
from tqdm import tqdm
import tyro

from crossformer.data.arec.arec import ArrayRecordBuilder, WriterSpec
from crossformer.utils.spec import spec

_JOINT_KEYS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
_KEEP_KP = {"base", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "eef", "tcp"}
_STEM_RE = re.compile(r"view_(\d+)")


@dataclass
class Config:
    path: Path
    mode: Literal["preview", "build"] = "preview"
    preview: int = 1

    shard_size: int = 1000
    name: str = "xarm_dream_100k"
    version: str = "0.0.1"
    branch: str = "main"
    builder: ArrayRecordBuilder = field(init=False)
    writers: WriterSpec = field(init=False)

    def __post_init__(self) -> None:
        self.writers = {"data": (["*"], {"options": "group_size:1"})}
        print(self)

    def build(self, fn) -> None:
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            branch=self.branch,
            shard_size=self.shard_size,
            writers=self.writers,
        )
        print(self.builder.root)
        self.builder.prepare(fn)


def _frame_index(stem: str) -> int:
    m = _STEM_RE.fullmatch(stem)
    if not m:
        raise ValueError(f"unexpected filename stem: {stem}")
    return int(m.group(1))


class RobotVgaLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        jsons = sorted(self.path.glob("view_*.json"), key=lambda p: _frame_index(p.stem))
        if not jsons:
            raise ValueError(f"no view_*.json under {self.path}")
        self._items: list[tuple[int, Path, Path, Path]] = []
        for jp in jsons:
            ip = jp.with_suffix(".png")
            if not ip.exists():
                continue
            self._items.append((_frame_index(jp.stem), jp, ip, mp))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame_idx, jp, ip = self._items[idx]
        meta = json.loads(jp.read_text())
        img = np.asarray(Image.open(ip).convert("RGB"))
        mask = np.asarray(Image.open(mp).convert("L"))
        return standardize(meta, img, mask, frame_idx=frame_idx, global_idx=idx)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]


def standardize(meta: dict[str, Any], img: np.ndarray, *, frame_idx: int, global_idx: int) -> dict[str, Any]:
    joints = np.asarray([meta["joints"][k] for k in _JOINT_KEYS], dtype=np.float32)
    gripper = np.float32(meta["joints"]["gripper_angle"])

    kps = [k for k in meta["keypoints"] if k["name"] in _KEEP_KP]
    kp2d = np.asarray([k["pixel_xy"] for k in kps], dtype=np.float32)
    kp3d_world = np.asarray([k["world_xyz"] for k in kps], dtype=np.float32)
    kp3d_camera = np.asarray([k["camera_xyz"] for k in kps], dtype=np.float32)
    kp_visible = np.asarray([k["visible"] for k in kps], dtype=np.bool_)

    cam = meta["camera"]
    intr = cam["intrinsics"]
    extr = cam["extrinsics"]

    return {
        "image": img,
        "mask": mask,
        "state": {
            "joints": joints,
            "gripper": gripper,
            "kp2d": kp2d,
            "kp3d_world": kp3d_world,
            "kp3d_camera": kp3d_camera,
        },
        "camera": {
            "intr": {
                "K": np.asarray(intr["K"], dtype=np.float32),
                "fx": np.float32(intr["fx"]),
                "fy": np.float32(intr["fy"]),
                "cx": np.float32(intr["cx"]),
                "cy": np.float32(intr["cy"]),
            },
            "extr": {
                "c2w": np.asarray(extr["camera_to_world"], dtype=np.float32),
                "w2c": np.asarray(extr["world_to_camera"], dtype=np.float32),
            },
            "target_world": np.asarray(meta["camera_target_world"], dtype=np.float32),
            "rpy_deg": np.asarray(meta["camera_rpy_deg"], dtype=np.float32),
            "focal_length": np.float32(meta["camera_focal_length"]),
        },
        "info": {
            "id": {
                "episode": np.asarray(frame_idx, dtype=np.int32),
                "step": np.asarray(0, dtype=np.int32),
                "global": np.asarray(global_idx, dtype=np.int32),
            },
            "len": np.asarray(1, dtype=np.int32),
            "kp_visible": kp_visible,
        },
    }


def main(cfg: Config) -> None:
    loader = RobotVgaLoader(cfg.path)
    print(f"Root: {loader.path}")
    print(f"n_frames={len(loader)}")

    if cfg.mode == "preview":
        n = min(cfg.preview, len(loader))
        for i in tqdm(range(n)):
            print(spec(loader[i]))
        return

    def build_fn() -> Iterator[dict[str, Any]]:
        yield from tqdm(loader, total=len(loader), desc="Building dataset")

    cfg.build(build_fn)


if __name__ == "__main__":
    main(tyro.cli(Config))
