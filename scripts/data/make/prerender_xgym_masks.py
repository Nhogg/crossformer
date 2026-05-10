"""Pre-render robot masks + 2D keypoints for xgym_sweep_single (or any
xgym dataset on the same rig).

Input: existing arec at ``<root>/<name>/<src_version>/<branch>/`` with image+proprio.
Output: new arec at ``<root>/<name>/<dst_version>/<branch>/`` with extra
``mask`` and ``kp2d`` per-camera fields baked in.

The point: render_robot_mask is the bottleneck during training (~1.5s/sample).
By pre-computing these once, ``prepare_irl_sample_np`` becomes synth-fast.

Usage:
    # preview a few records to a tmp arec
    uv run scripts/data/make/prerender_xgym_masks.py \\
        --name xgym_sweep_single --src-version 0.5.6 --dst-version 0.6.0 \\
        --cams low side --n-preview 16

    # full build (this is the slow step — uses --workers parallelism)
    uv run scripts/data/make/prerender_xgym_masks.py \\
        --name xgym_sweep_single --src-version 0.5.6 --dst-version 0.6.0 \\
        --cams low side --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parents[3]))

import numpy as np
from tqdm import tqdm

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record

# ---------------------------------------------------------------------------
# Worker: each process loads pyroki + URDF mesh once, then renders masks.

_WORKER_CTX: dict | None = None


def _worker_init(extr_dir: str, cams: tuple[str, ...], raw_h: int, raw_w: int) -> None:
    """Load pyroki mesh + per-camera (K, w2c) once per worker."""
    # Force CPU JAX in workers - GPUs are for training, not pre-render.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    # Imports happen inside worker so the parent process doesn't pay them.
    from crossformer.utils.callbacks.synth_viz import _get_robot_mesh, fk_keypoints
    from crossformer.utils.rig import K_for_size, load_w2c

    robot = _get_robot_mesh()  # warms pyroki singleton
    K_raw = K_for_size(raw_h, raw_w)
    w2c_per_cam = {cam: load_w2c(cam, Path(extr_dir)) for cam in cams}

    # warm pyroki FK + projection JIT compiles with a dummy joint vector
    dummy_joints = np.zeros(7, dtype=np.float64)
    _ = fk_keypoints(dummy_joints)
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    _ = robot.posed_verts(q)

    global _WORKER_CTX
    _WORKER_CTX = {
        "robot": robot,
        "fk_keypoints": fk_keypoints,
        "K_raw": K_raw,
        "w2c": w2c_per_cam,
        "raw_h": raw_h,
        "raw_w": raw_w,
        "cams": cams,
    }


def _project_kp(pts_3d: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_h = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=pts_3d.dtype)], axis=-1)
    cam = (w2c.astype(np.float64) @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pix = (K.astype(np.float64) @ cam.T).T
    uv = pix[:, :2] / z_safe[:, None]
    return uv.astype(np.float32), z.astype(np.float32)


def _render_one(joints: np.ndarray, gripper_drive: float, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mask uint8 HxW, kp2d (10,3) float32 = u,v,vis)."""
    from crossformer.utils.mask_renderer import Intrinsics, rasterize_mesh

    ctx = _WORKER_CTX
    assert ctx is not None, "worker not initialized"
    robot = ctx["robot"]
    K_raw = ctx["K_raw"]
    h, w = ctx["raw_h"], ctx["raw_w"]

    pts_3d = ctx["fk_keypoints"](joints.astype(np.float64))
    uv, z = _project_kp(pts_3d, w2c, K_raw)
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    vis = (in_frame & (z > 0)).astype(np.float32)
    kp2d = np.concatenate([uv, vis[:, None]], axis=-1)  # (10, 3)

    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints
    q[0, 7] = float(gripper_drive)
    verts_world = robot.posed_verts(q)[0]

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c_gl = flip @ w2c.astype(np.float64)
    verts_cam = (w2c_gl @ verts_world.astype(np.float64).T).T[:, :3]

    intr = Intrinsics(
        fx=float(K_raw[0, 0]),
        fy=float(K_raw[1, 1]),
        cx=float(K_raw[0, 2]),
        cy=float(K_raw[1, 2]),
        width=int(w),
        height=int(h),
    )
    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    inst_buf = np.zeros((h, w), dtype=np.uint8)
    rasterize_mesh(verts_cam, robot.faces, 1, intr, depth_buf, inst_buf)
    mask = (inst_buf > 0).astype(np.uint8) * 255
    return mask, kp2d


def _process_record(args: tuple[int, np.ndarray, float]) -> tuple[int, dict]:
    """Render masks + kp2d for all configured cameras.

    Input: (idx, joints (7,), gripper_drive float)
    Output: (idx, {"mask": {cam: (H,W) uint8}, "kp2d": {cam: (10,3) float32}})
    """
    idx, joints, gripper_drive = args
    ctx = _WORKER_CTX
    assert ctx is not None
    masks: dict[str, np.ndarray] = {}
    kps: dict[str, np.ndarray] = {}
    for cam in ctx["cams"]:
        mask, kp2d = _render_one(joints, gripper_drive, ctx["w2c"][cam])
        masks[cam] = mask
        kps[cam] = kp2d
    return idx, {"mask": masks, "kp2d": kps}


# ---------------------------------------------------------------------------
# Main pipeline


def _open_src(name: str, version: str, branch: str, root: Path):
    """Open the existing image+proprio writers as plain ArrayRecordDataSources.

    Note: we read raw shards with the standard builder, NOT MultiArrayRecordSource,
    so that we get one record per step (not the chunked join).
    """
    builder = ArrayRecordBuilder(name=name, version=version, branch=branch, root=str(root))
    # populate writers from on-disk meta
    meta_path = builder.root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta: {meta_path}")
    import json

    meta = json.loads(meta_path.read_text())
    builder.writers = builder._normalize_writers(meta["writers"])
    builder.default_writer = "data" if "data" in builder.writers else next(iter(builder.writers))

    return builder, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="xgym_sweep_single")
    p.add_argument("--src-version", default="0.5.6")
    p.add_argument("--dst-version", default="0.6.0")
    p.add_argument("--branch", default="main")
    p.add_argument("--root", type=Path, default=Path.home() / ".cache/arrayrecords")
    p.add_argument("--cams", nargs="+", default=["low", "side"])
    p.add_argument("--extr-dir", type=Path, default=Path.home() / "data/extr/cam")
    p.add_argument("--raw-h", type=int, default=480)
    p.add_argument("--raw-w", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-preview", type=int, default=0, help="if > 0, only process this many records to a tmp branch")
    p.add_argument("--shard-size", type=int, default=1000)
    args = p.parse_args()

    root = args.root.expanduser()
    cams = tuple(args.cams)
    print(f"reading {args.name} v{args.src_version} from {root}", flush=True)
    src_builder, src_meta = _open_src(args.name, args.src_version, args.branch, root)

    if not {"image", "proprio"}.issubset(src_builder.writers):
        raise ValueError(f"expected image+proprio writers, got {list(src_builder.writers)}")

    img_src = src_builder.get_source("image")
    pro_src = src_builder.get_source("proprio")
    n = len(img_src)
    assert n == len(pro_src), f"image/proprio length mismatch: {len(img_src)} vs {len(pro_src)}"
    print(f"  records: {n}", flush=True)

    if args.n_preview > 0:
        n = min(n, args.n_preview)
        dst_branch = f"{args.branch}_preview"
        print(f"  preview mode: writing first {n} records to branch={dst_branch}", flush=True)
    else:
        dst_branch = args.branch

    # Destination builder: same writer split, plus mask/kp2d on each side.
    # Mask is per-step (same as image), kp2d is small (goes with proprio side).
    dst_writers = {
        "image": (["image", "mask"], {"options": "group_size:1"}),
        "proprio": (["proprio", "info", "kp2d"], {"options": "group_size:1"}),
    }
    dst_builder = ArrayRecordBuilder(
        name=args.name,
        version=args.dst_version,
        branch=dst_branch,
        root=str(root),
        shard_size=args.shard_size,
        writers=dst_writers,
    )
    print(f"  writing to {dst_builder.root}", flush=True)

    # Generator that pairs (image, proprio) records, dispatches to workers,
    # collects results, yields combined samples in order.
    def gen():
        ctx_args = (str(args.extr_dir), cams, args.raw_h, args.raw_w)
        with mp.get_context("spawn").Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=ctx_args,
        ) as pool:
            print(f"  pool of {args.workers} workers spawned + warmed", flush=True)

            def task_iter():
                for i in range(n):
                    pro = unpack_record(pro_src[i])
                    joints = np.asarray(pro["proprio"]["joints"], dtype=np.float32).reshape(7)
                    gripper = float(np.asarray(pro["proprio"]["gripper"]).reshape(-1)[0])
                    gripper_drive = gripper * 0.85  # 0=open → 1=closed
                    yield (i, joints, gripper_drive)

            t0 = time.perf_counter()
            n_done = 0
            for idx, render_out in tqdm(
                pool.imap(_process_record, task_iter(), chunksize=4),
                total=n,
                desc="rendering",
            ):
                # re-read at consumption time so we have image+proprio for this idx
                img_rec = unpack_record(img_src[idx])
                pro_rec = unpack_record(pro_src[idx])
                sample = {
                    "image": img_rec["image"],  # dict[cam, img]
                    "mask": render_out["mask"],  # dict[cam, mask]
                    "proprio": pro_rec["proprio"],
                    "info": pro_rec.get("info", {}),
                    "kp2d": render_out["kp2d"],  # dict[cam, (10,3)]
                }
                yield sample
                n_done += 1
                if n_done % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = n_done / elapsed
                    eta = (n - n_done) / rate if rate > 0 else float("inf")
                    print(f"  [{n_done}/{n}] rate={rate:.1f}/s eta={eta / 60:.1f}min", flush=True)

    print("starting build...", flush=True)
    t = time.perf_counter()
    dst_builder.prepare(gen)
    print(f"done in {(time.perf_counter() - t) / 60:.1f} min → {dst_builder.root}", flush=True)


if __name__ == "__main__":
    main()
