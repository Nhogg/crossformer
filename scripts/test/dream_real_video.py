from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from rich import print
from rich.table import Table
from tqdm import tqdm

from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.callbacks.synth_viz import composite_robot, rasterize_robot
from scripts.train import dream as train_dream

DEFAULT_DATA_DIR = Path("/home/bela/datasets/dream-real-testdata/IRL-Test-Data")
DEFAULT_CHECKPOINT = Path("/home/bela/weights/nvdream-v1")
DEFAULT_JOINTS_DEG = (0.0, -45.0, 0.0, 35.0, 0.0, 65.0, 90.0)


@dataclass
class PredRow:
    frame: int
    video_time_s: float
    success: bool
    valid_kp: int
    reproj_px: float
    mean_conf: float
    w2c: np.ndarray | None


def _checkpoint_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "params":
        return path.parent
    if (path / "params").exists():
        return path
    raise FileNotFoundError(f"expected checkpoint root with params/ or params dir, got {path}")


def _read_K(path: Path) -> np.ndarray:
    K = np.loadtxt(path.expanduser(), delimiter=",", dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"expected 3x3 camera matrix in {path}, got {K.shape}")
    return K


def _make_cfg(args: argparse.Namespace) -> train_dream.Config:
    cfg = train_dream.Config()
    cfg.name = "dream-real-video-test"
    cfg.seed = args.seed
    cfg.encoder = args.encoder
    cfg.decoder = args.decoder
    cfg.variant = args.variant
    cfg.net_in_size = tuple(args.net_in_size)
    cfg.tips_variant = args.tips_variant
    cfg.tips_checkpoint = args.tips_checkpoint
    cfg.n_stages = args.n_stages
    cfg.skip_connections = args.skip_connections
    cfg.internalize_spatial_softmax = False
    cfg.wandb.use = False
    cfg.bs = args.batch_size
    return cfg


def _load_model(cfg: train_dream.Config, ckpt: Path, *, old_checkpoint_api: bool):
    net_h, net_w = cfg.net_in_size
    out_h, out_w = train_dream.net_out_size(cfg)
    model = train_dream.make_model(cfg, num_keypoints=10)
    dummy = jnp.zeros((1, net_h, net_w, cfg.image_c), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(cfg.seed), dummy)["params"]
    params = train_dream.load_tips_params(cfg, params)
    params = jax.tree.map(jnp.asarray, params)
    params = SaveCallback(_checkpoint_root(ckpt), new_api=not old_checkpoint_api).load_params(params, step=None)
    return model, params, out_h, out_w


def _frame_indices(total: int, stride: int, max_frames: int | None) -> set[int]:
    idx = list(range(0, total, stride))
    if max_frames is not None:
        idx = idx[:max_frames]
    return set(idx)


def _read_video_batches(
    video_path: Path,
    raw_K: np.ndarray,
    net_size: tuple[int, int],
    batch_size: int,
    stride: int,
    max_frames: int | None,
):
    cap = cv2.VideoCapture(str(video_path.expanduser()))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected = _frame_indices(total, stride, max_frames)
    net_h, net_w = net_size
    frames: list[np.ndarray] = []
    Ks: list[np.ndarray] = []
    meta: list[tuple[int, float]] = []

    pbar = tqdm(total=len(selected), desc="read frames")
    frame_i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_i not in selected:
            frame_i += 1
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_h, raw_w = rgb.shape[:2]
        img = train_dream._shrink_crop_image_np(rgb, net_h, net_w, Image.BILINEAR)
        K = train_dream._shrink_crop_intrinsics_np(raw_K, raw_h, raw_w, net_h, net_w)
        frames.append(img)
        Ks.append(K)
        meta.append((frame_i, frame_i / fps if fps > 0 else float("nan")))
        pbar.update(1)

        if len(frames) == batch_size:
            yield np.stack(frames), np.stack(Ks), meta
            frames, Ks, meta = [], [], []
        frame_i += 1

    pbar.close()
    cap.release()
    if frames:
        yield np.stack(frames), np.stack(Ks), meta


def _predict_batch(model, params, images: np.ndarray, out_h: int, out_w: int):
    batch = {"image": jnp.asarray(images)}
    out = train_dream.predict_heatmap_out(model, params, batch, out_h, out_w)
    heatmaps = train_dream.final_pred_heatmaps(out["pred_heatmaps"])
    pred_uv_out, conf = train_dream.extract_keypoints(heatmaps)
    pred_uv = train_dream._denormalize_kp2d(
        pred_uv_out / jnp.array([out_w, out_h], dtype=jnp.float32),
        images.shape[1],
        images.shape[2],
    )
    pred_mask = out.get("pred_mask")
    if pred_mask is not None:
        pred_mask = jnp.squeeze(pred_mask, axis=1)
    return jax.device_get(pred_uv), jax.device_get(conf), jax.device_get(pred_mask) if pred_mask is not None else None


def _solve_and_raster(q_deg: np.ndarray, uv: np.ndarray, conf: np.ndarray, K: np.ndarray, h: int, w: int):
    joints_rad, valid, w2c = train_dream._solve_pose_one(q_deg, uv, conf, K)
    reproj_px = float("nan")
    rast = None
    if w2c is not None:
        reproj_px = train_dream._pnp_reproj_err(w2c, joints_rad, uv, valid, K)
        rast = rasterize_robot(joints_rad, w2c, K, w, h)
    return valid, w2c, reproj_px, rast


def _draw_kp(img: np.ndarray, uv: np.ndarray, conf: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, (pt, c) in enumerate(zip(uv, conf)):
        if not np.isfinite(pt).all() or pt[0] <= -999.0:
            continue
        color = (0, 255, 0) if valid[i] else (0, 0, 255)
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(out, str(i), (int(pt[0]) + 5, int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(out, f"{c:.2f}", (int(pt[0]) + 5, int(pt[1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _panel(image: np.ndarray, uv: np.ndarray, conf: np.ndarray, valid: np.ndarray, pred_mask, rast_mask) -> np.ndarray:
    kp_img = _draw_kp(image, uv, conf, valid)
    mask_img = np.zeros_like(image)
    if pred_mask is not None:
        m = np.clip(pred_mask, 0.0, 1.0)
        mask_img[..., 1] = (m * 255).astype(np.uint8)
    rast_img = composite_robot(image, rast_mask) if rast_mask is not None else image.copy()
    return np.concatenate([kp_img, mask_img, rast_img], axis=1)


def _write_outputs(out_dir: Path, rows: list[PredRow]) -> None:
    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "video_time_s", "success", "valid_kp", "reproj_px", "mean_conf"])
        for row in rows:
            writer.writerow([row.frame, row.video_time_s, int(row.success), row.valid_kp, row.reproj_px, row.mean_conf])

    w2c_dir = out_dir / "w2c"
    w2c_dir.mkdir(exist_ok=True)
    for row in rows:
        if row.w2c is not None:
            np.save(w2c_dir / f"frame_{row.frame:06d}.npy", row.w2c)


def _print_summary(rows: list[PredRow]) -> None:
    success = np.asarray([r.success for r in rows], dtype=bool)
    reproj = np.asarray([r.reproj_px for r in rows], dtype=np.float32)
    valid = np.asarray([r.valid_kp for r in rows], dtype=np.float32)
    conf = np.asarray([r.mean_conf for r in rows], dtype=np.float32)
    table = Table("frames", "pnp success", "valid kp mean", "conf mean", "reproj px mean/median")
    table.add_row(
        str(len(rows)),
        f"{float(success.mean()) if len(rows) else 0.0:.3f}",
        f"{float(valid.mean()) if len(rows) else 0.0:.2f}",
        f"{float(conf.mean()) if len(rows) else 0.0:.3f}",
        f"{float(np.nanmean(reproj)):.2f} / {float(np.nanmedian(reproj)):.2f}"
        if np.isfinite(reproj).any()
        else "nan / nan",
    )
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DREAM checkpoint inference on a real MP4.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--camera-matrix", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/dream_real_video_test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder", default="tips")
    parser.add_argument("--decoder", default="dpt")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--net-in-size", type=int, nargs=2, default=(200, 200), metavar=("H", "W"))
    parser.add_argument("--tips-variant", default="tips_v2_b14")
    parser.add_argument("--tips-checkpoint", type=Path, default=None)
    parser.add_argument("--n-stages", type=int, default=1)
    parser.add_argument("--skip-connections", action="store_true")
    parser.add_argument("--old-checkpoint-api", action="store_true")
    parser.add_argument(
        "--joints-deg",
        type=float,
        nargs=7,
        default=DEFAULT_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    args = parser.parse_args()

    video = args.video or args.data_dir / "rgb.MP4"
    camera_matrix = args.camera_matrix or args.data_dir / "camera_matrix.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "panels").mkdir(exist_ok=True)
    (args.out_dir / "pred_masks").mkdir(exist_ok=True)
    (args.out_dir / "rast_masks").mkdir(exist_ok=True)

    raw_K = _read_K(camera_matrix)
    cfg = _make_cfg(args)
    model, params, out_h, out_w = _load_model(cfg, args.checkpoint, old_checkpoint_api=args.old_checkpoint_api)
    q_deg = np.asarray(args.joints_deg, dtype=np.float64)

    rows: list[PredRow] = []
    for images, Ks, meta in _read_video_batches(
        video, raw_K, cfg.net_in_size, args.batch_size, args.stride, args.max_frames
    ):
        uv, conf, pred_masks = _predict_batch(model, params, images, out_h, out_w)
        for i, (frame_i, t_s) in enumerate(meta):
            valid, w2c, reproj_px, rast = _solve_and_raster(
                q_deg, uv[i], conf[i], Ks[i], images.shape[1], images.shape[2]
            )
            rows.append(
                PredRow(
                    frame=frame_i,
                    video_time_s=t_s,
                    success=w2c is not None,
                    valid_kp=int(valid.sum()),
                    reproj_px=reproj_px,
                    mean_conf=float(np.nanmean(conf[i])),
                    w2c=w2c,
                )
            )

            pred_mask_i = pred_masks[i] if pred_masks is not None else None
            if pred_mask_i is not None:
                cv2.imwrite(
                    str(args.out_dir / "pred_masks" / f"frame_{frame_i:06d}.png"),
                    (np.clip(pred_mask_i, 0, 1) * 255).astype(np.uint8),
                )
            if rast is not None:
                cv2.imwrite(
                    str(args.out_dir / "rast_masks" / f"frame_{frame_i:06d}.png"),
                    (np.clip(rast, 0, 1) * 255).astype(np.uint8),
                )
            if len(rows) == 1 or (len(rows) - 1) % args.save_every == 0:
                panel = _panel(images[i], uv[i], conf[i], valid, pred_mask_i, rast)
                cv2.imwrite(
                    str(args.out_dir / "panels" / f"frame_{frame_i:06d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                )

    _write_outputs(args.out_dir, rows)
    _print_summary(rows)
    print(f"[green]wrote outputs to {args.out_dir}[/]")


if __name__ == "__main__":
    main()
