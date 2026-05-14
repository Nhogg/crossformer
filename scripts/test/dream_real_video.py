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

from crossformer.utils.callbacks.rast import _RobotMesh
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
    refined: bool = False
    refine_loss0: float = float("nan")
    refine_loss1: float = float("nan")
    refine_iou0: float = float("nan")
    refine_iou1: float = float("nan")
    w2c_refined: np.ndarray | None = None


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


def _mask_iou_float(a: np.ndarray | None, b: np.ndarray | None, thresh: float = 0.5) -> float:
    if a is None or b is None:
        return float("nan")
    aa = np.asarray(a) > thresh
    bb = np.asarray(b) > thresh
    union = aa | bb
    return float((aa & bb).sum() / union.sum()) if union.any() else float("nan")


def _axis_angle_R(vec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(vec, dtype=np.float64) / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _apply_w2c_delta(w2c: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = np.asarray(w2c, dtype=np.float64).copy()
    R = _axis_angle_R(delta[:3])
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = R @ out[:3, 3] + delta[3:]
    return out


def _hard_iou_pose_search(
    w2c0: np.ndarray,
    joints_rad: np.ndarray,
    K: np.ndarray,
    target_mask: np.ndarray,
    width: int,
    height: int,
    *,
    max_rot_deg: float,
    max_trans_m: float,
    passes: int,
):
    """Coordinate search over tiny camera-frame pose deltas using true hard-raster IoU."""
    best_w2c = np.asarray(w2c0, dtype=np.float64)
    best_rast = rasterize_robot(joints_rad, best_w2c, K, width, height)
    best_iou = _mask_iou_float(best_rast, target_mask)
    rot0 = np.deg2rad(max_rot_deg)
    trans0 = max_trans_m

    for p in range(max(passes, 0)):
        scale = 0.5**p
        moves = []
        for ax in range(3):
            for sign in (-1.0, 1.0):
                d = np.zeros(6, dtype=np.float64)
                d[ax] = sign * rot0 * scale
                moves.append(d)
        for ax in range(3, 6):
            for sign in (-1.0, 1.0):
                d = np.zeros(6, dtype=np.float64)
                d[ax] = sign * trans0 * scale
                moves.append(d)

        improved = False
        for d in moves:
            cand_w2c = _apply_w2c_delta(best_w2c, d)
            cand_rast = rasterize_robot(joints_rad, cand_w2c, K, width, height)
            cand_iou = _mask_iou_float(cand_rast, target_mask)
            if np.isfinite(cand_iou) and cand_iou > best_iou:
                best_w2c, best_rast, best_iou = cand_w2c, cand_rast, cand_iou
                improved = True
        if not improved:
            continue
    return best_w2c, best_rast, best_iou


def _proj_from_K_torch(torch, K, width: int, height: int, device):
    P = torch.zeros((4, 4), dtype=torch.float32, device=device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = -(10.0 + 0.01) / (10.0 - 0.01)
    P[2, 3] = -2.0 * 10.0 * 0.01 / (10.0 - 0.01)
    P[3, 2] = -1.0
    return P


def _euler_to_R_torch(torch, a):
    sx, sy, sz = torch.sin(a)
    cx, cy, cz = torch.cos(a)
    one = torch.ones((), dtype=a.dtype, device=a.device)
    zero = torch.zeros((), dtype=a.dtype, device=a.device)
    Rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cx, -sx]),
            torch.stack([zero, sx, cx]),
        ]
    )
    Ry = torch.stack(
        [
            torch.stack([cy, zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cz, -sz, zero]),
            torch.stack([sz, cz, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    return Rz @ Ry @ Rx


class MaskPoseRefiner:
    """Differentiable silhouette refinement of camera extrinsics from a PnP init."""

    def __init__(
        self,
        q_deg: np.ndarray,
        *,
        width: int,
        height: int,
        lr: float,
        steps: int,
        reg: float,
        kp_weight: float,
        bce_weight: float,
        dice_weight: float,
        mask_thresh: float,
        max_rot_deg: float,
        max_trans_m: float,
    ):
        import nvdiffrast.torch as dr
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("mask refinement requires CUDA for nvdiffrast")
        self.dr = dr
        self.torch = torch
        self.device = torch.device("cuda")
        self.width = width
        self.height = height
        self.lr = lr
        self.steps = steps
        self.reg = reg
        self.kp_weight = kp_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.mask_thresh = mask_thresh
        self.max_rot = np.deg2rad(max_rot_deg)
        self.max_trans = max_trans_m
        self.robot = _RobotMesh(Path("xarm7_standalone.urdf"), Path("assets"))
        q = np.zeros((1, self.robot.actuated), dtype=np.float32)
        q[0, :7] = np.deg2rad(np.asarray(q_deg, dtype=np.float32))
        verts = np.asarray(self.robot.posed_verts(q)[0], dtype=np.float32)
        keypoints = train_dream.fk_keypoints(np.deg2rad(np.asarray(q_deg, dtype=np.float64)), self.robot)
        self.verts = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
        self.keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=self.device)
        self.faces = torch.as_tensor(self.robot.faces.astype(np.int32), dtype=torch.int32, device=self.device)
        self.flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=self.device))
        self.ctx = dr.RasterizeCudaContext(device=self.device)

    def _render(self, w2c, K):
        torch = self.torch
        P = _proj_from_K_torch(torch, K, self.width, self.height, self.device)
        view = self.flip @ w2c
        mvp = P @ view
        clip = self.verts @ mvp.T
        rast, _ = self.dr.rasterize(self.ctx, clip[None], self.faces, resolution=[self.height, self.width])
        alpha = torch.clamp(rast[..., 3:4], 0.0, 1.0)
        alpha = self.dr.antialias(alpha, rast, clip[None], self.faces)
        return alpha[0, ..., 0]

    def refine(
        self,
        w2c_np: np.ndarray,
        K_np: np.ndarray,
        target_np: np.ndarray,
        uv_np: np.ndarray,
        valid_np: np.ndarray,
    ) -> dict:
        torch = self.torch
        w2c0 = torch.as_tensor(w2c_np, dtype=torch.float32, device=self.device)
        K_scaled = np.asarray(K_np, dtype=np.float32).copy()
        src_h, src_w = target_np.shape[:2]
        K_scaled[0] *= self.width / float(src_w)
        K_scaled[1] *= self.height / float(src_h)
        K = torch.as_tensor(K_scaled, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(np.clip(target_np, 0.0, 1.0), dtype=torch.float32, device=self.device)
        if tuple(target.shape) != (self.height, self.width):
            target = torch.nn.functional.interpolate(
                target[None, None],
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        target_bin = (target >= self.mask_thresh).float()

        uv = np.asarray(uv_np, dtype=np.float32).copy()
        uv[:, 0] *= self.width / float(src_w)
        uv[:, 1] *= self.height / float(src_h)
        uv = torch.as_tensor(uv, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(np.asarray(valid_np, dtype=bool), dtype=torch.bool, device=self.device)

        raw_delta = torch.zeros(6, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([raw_delta], lr=self.lr)

        def bounded_delta():
            delta = torch.cat(
                [
                    torch.tanh(raw_delta[:3]) * self.max_rot,
                    torch.tanh(raw_delta[3:]) * self.max_trans,
                ]
            )
            return delta

        def compose(delta=None):
            delta = bounded_delta() if delta is None else delta
            R_delta = _euler_to_R_torch(torch, delta[:3])
            t_delta = delta[3:]
            out = w2c0.clone()
            out[:3, :3] = R_delta @ w2c0[:3, :3]
            out[:3, 3] = R_delta @ w2c0[:3, 3] + t_delta
            return out

        def reproj_loss(w2c):
            if int(valid.sum().detach().cpu()) == 0:
                return torch.zeros((), dtype=torch.float32, device=self.device)
            pts = self.keypoints[valid]
            pts_cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
            pix_h = pts_cam @ K.T
            pix = pix_h[:, :2] / torch.clamp(pix_h[:, 2:3], min=1e-6)
            scale = torch.tensor([self.width, self.height], dtype=torch.float32, device=self.device)
            return torch.mean(((pix - uv[valid]) / scale) ** 2)

        def soft_iou(mask):
            inter = torch.sum(mask * target_bin)
            union = torch.sum(mask) + torch.sum(target_bin) - inter
            return inter / torch.clamp(union, min=1e-6)

        def mask_loss(mask):
            eps = 1e-5
            mse = torch.mean((mask - target) ** 2)
            bce = torch.nn.functional.binary_cross_entropy(torch.clamp(mask, eps, 1.0 - eps), target_bin)
            dice = 1.0 - (2.0 * torch.sum(mask * target_bin) + eps) / (torch.sum(mask) + torch.sum(target_bin) + eps)
            return mse + self.bce_weight * bce + self.dice_weight * dice

        with torch.no_grad():
            mask0 = self._render(w2c0, K)
            loss0 = mask_loss(mask0)
            best_score = soft_iou(mask0)
            best_delta = bounded_delta().detach().clone()
            best_loss = loss0.detach().clone()

        loss = loss0
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            w2c = compose()
            mask = self._render(w2c, K)
            data_loss = mask_loss(mask)
            pose_loss = self.reg * torch.sum(raw_delta * raw_delta)
            loss = data_loss + self.kp_weight * reproj_loss(w2c) + pose_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                post_delta = bounded_delta().detach().clone()
                post_mask = self._render(compose(post_delta), K)
                post_score = soft_iou(post_mask)
                if post_score > best_score:
                    best_score = post_score.detach().clone()
                    best_delta = post_delta
                    best_loss = mask_loss(post_mask).detach().clone()

        with torch.no_grad():
            w2c = compose(best_delta)
            mask1 = self._render(w2c, K)
            target_out = target.detach().cpu().numpy()
        return {
            "w2c": w2c.detach().cpu().numpy().astype(np.float64),
            "soft_mask": mask1.detach().cpu().numpy(),
            "loss0": float(loss0.detach().cpu()),
            "loss1": float(best_loss.detach().cpu()),
            "soft_iou0": _mask_iou_float(mask0.detach().cpu().numpy(), target_out),
            "soft_iou1": _mask_iou_float(mask1.detach().cpu().numpy(), target_out),
        }


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


def _mask_overlay(
    base: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[float, float, float],
    alpha: float = 0.45,
) -> np.ndarray:
    out = base.astype(np.float32) / 255.0
    if mask is None:
        return base.copy()
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)[..., None]
    c = np.asarray(color, dtype=np.float32)[None, None, :]
    out = out * (1.0 - alpha * m) + c * (alpha * m)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _mask_diff_overlay(image: np.ndarray, pred_mask: np.ndarray | None, rast_mask: np.ndarray | None) -> np.ndarray:
    base = image.copy()
    if rast_mask is not None:
        base = _mask_overlay(base, rast_mask, color=(0.15, 0.45, 1.0), alpha=0.45)
    if pred_mask is not None:
        base = _mask_overlay(base, pred_mask, color=(1.0, 0.85, 0.05), alpha=0.50)
    return base


def _panel(
    image: np.ndarray,
    uv: np.ndarray,
    conf: np.ndarray,
    valid: np.ndarray,
    pred_mask,
    rast_mask,
    refined_rast=None,
    candidate_rast=None,
) -> np.ndarray:
    kp_img = _draw_kp(image, uv, conf, valid)
    mask_img = np.zeros_like(image)
    if pred_mask is not None:
        m = np.clip(pred_mask, 0.0, 1.0)
        mask_img[..., 1] = (m * 255).astype(np.uint8)
    rast_img = composite_robot(image, rast_mask) if rast_mask is not None else image.copy()
    pred_on_image = _mask_overlay(image, pred_mask, color=(0.0, 1.0, 0.25), alpha=0.45)
    pred_vs_rast = _mask_diff_overlay(image, pred_mask, rast_mask)
    panels = [kp_img, mask_img, rast_img, pred_on_image, pred_vs_rast]
    if refined_rast is not None:
        panels.extend(
            [
                composite_robot(image, refined_rast),
                _mask_diff_overlay(image, pred_mask, refined_rast),
            ]
        )
    elif candidate_rast is not None:
        panels.extend(
            [
                composite_robot(image, candidate_rast),
                _mask_diff_overlay(image, pred_mask, candidate_rast),
            ]
        )
    return np.concatenate(panels, axis=1)


def _write_outputs(out_dir: Path, rows: list[PredRow]) -> None:
    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "video_time_s",
                "success",
                "valid_kp",
                "reproj_px",
                "mean_conf",
                "refined",
                "refine_loss0",
                "refine_loss1",
                "refine_iou0",
                "refine_iou1",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.frame,
                    row.video_time_s,
                    int(row.success),
                    row.valid_kp,
                    row.reproj_px,
                    row.mean_conf,
                    int(row.refined),
                    row.refine_loss0,
                    row.refine_loss1,
                    row.refine_iou0,
                    row.refine_iou1,
                ]
            )

    w2c_dir = out_dir / "w2c"
    w2c_dir.mkdir(exist_ok=True)
    for row in rows:
        if row.w2c is not None:
            np.save(w2c_dir / f"frame_{row.frame:06d}.npy", row.w2c)
    w2c_refined_dir = out_dir / "w2c_refined"
    w2c_refined_dir.mkdir(exist_ok=True)
    for row in rows:
        if row.w2c_refined is not None:
            np.save(w2c_refined_dir / f"frame_{row.frame:06d}.npy", row.w2c_refined)


def _print_summary(rows: list[PredRow]) -> None:
    success = np.asarray([r.success for r in rows], dtype=bool)
    reproj = np.asarray([r.reproj_px for r in rows], dtype=np.float32)
    valid = np.asarray([r.valid_kp for r in rows], dtype=np.float32)
    conf = np.asarray([r.mean_conf for r in rows], dtype=np.float32)
    refined = np.asarray([r.refined for r in rows], dtype=bool)
    iou0 = np.asarray([r.refine_iou0 for r in rows], dtype=np.float32)
    iou1 = np.asarray([r.refine_iou1 for r in rows], dtype=np.float32)
    table = Table("frames", "pnp success", "valid kp mean", "conf mean", "reproj px mean/median", "refined IoU")
    table.add_row(
        str(len(rows)),
        f"{float(success.mean()) if len(rows) else 0.0:.3f}",
        f"{float(valid.mean()) if len(rows) else 0.0:.2f}",
        f"{float(conf.mean()) if len(rows) else 0.0:.3f}",
        f"{float(np.nanmean(reproj)):.2f} / {float(np.nanmedian(reproj)):.2f}"
        if np.isfinite(reproj).any()
        else "nan / nan",
        f"{float(np.nanmean(iou0[refined])):.3f} -> {float(np.nanmean(iou1[refined])):.3f}" if refined.any() else "off",
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
    parser.add_argument("--refine", action="store_true", help="Refine PnP extrinsics against DREAM predicted mask.")
    parser.add_argument("--refine-steps", type=int, default=40)
    parser.add_argument("--refine-lr", type=float, default=3e-2)
    parser.add_argument("--refine-reg", type=float, default=1e-3)
    parser.add_argument("--refine-kp-weight", type=float, default=10.0)
    parser.add_argument("--refine-bce-weight", type=float, default=0.05)
    parser.add_argument("--refine-dice-weight", type=float, default=0.25)
    parser.add_argument("--refine-mask-thresh", type=float, default=0.5)
    parser.add_argument("--refine-max-rot-deg", type=float, default=2.0)
    parser.add_argument("--refine-max-trans-m", type=float, default=0.02)
    parser.add_argument("--refine-min-iou-gain", type=float, default=0.0)
    parser.add_argument("--refine-hard-search-passes", type=int, default=2)
    parser.add_argument(
        "--refine-size", type=int, default=200, help="Differentiable render size. Use <= net input size."
    )
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
    (args.out_dir / "rast_masks_refined").mkdir(exist_ok=True)
    (args.out_dir / "rast_masks_candidates").mkdir(exist_ok=True)
    (args.out_dir / "w2c_candidates").mkdir(exist_ok=True)

    raw_K = _read_K(camera_matrix)
    cfg = _make_cfg(args)
    model, params, out_h, out_w = _load_model(cfg, args.checkpoint, old_checkpoint_api=args.old_checkpoint_api)
    q_deg = np.asarray(args.joints_deg, dtype=np.float64)
    joints_rad = np.deg2rad(q_deg)

    refiner = None
    if args.refine:
        ref_h = min(int(args.refine_size), int(cfg.net_in_size[0]))
        ref_w = min(int(args.refine_size), int(cfg.net_in_size[1]))
        refiner = MaskPoseRefiner(
            q_deg,
            width=ref_w,
            height=ref_h,
            lr=args.refine_lr,
            steps=args.refine_steps,
            reg=args.refine_reg,
            kp_weight=args.refine_kp_weight,
            bce_weight=args.refine_bce_weight,
            dice_weight=args.refine_dice_weight,
            mask_thresh=args.refine_mask_thresh,
            max_rot_deg=args.refine_max_rot_deg,
            max_trans_m=args.refine_max_trans_m,
        )

    rows: list[PredRow] = []
    for images, Ks, meta in _read_video_batches(
        video, raw_K, cfg.net_in_size, args.batch_size, args.stride, args.max_frames
    ):
        uv, conf, pred_masks = _predict_batch(model, params, images, out_h, out_w)
        for i, (frame_i, t_s) in enumerate(meta):
            valid, w2c, reproj_px, rast = _solve_and_raster(
                q_deg, uv[i], conf[i], Ks[i], images.shape[1], images.shape[2]
            )
            pred_mask_i = pred_masks[i] if pred_masks is not None else None
            w2c_refined = None
            refined_rast = None
            candidate_w2c = None
            candidate_rast = None
            refine_loss0 = float("nan")
            refine_loss1 = float("nan")
            refine_iou0 = float("nan")
            refine_iou1 = float("nan")
            refined = False
            if refiner is not None and w2c is not None and pred_mask_i is not None:
                refine_out = refiner.refine(w2c, Ks[i], pred_mask_i, uv[i], valid)
                refine_loss0 = refine_out["loss0"]
                refine_loss1 = refine_out["loss1"]
                candidate_w2c = refine_out["w2c"]
                candidate_rast = rasterize_robot(joints_rad, candidate_w2c, Ks[i], images.shape[2], images.shape[1])
                refine_iou0 = _mask_iou_float(rast, pred_mask_i)
                refine_iou1 = _mask_iou_float(candidate_rast, pred_mask_i)
                search_w2c, search_rast, search_iou = _hard_iou_pose_search(
                    w2c,
                    joints_rad,
                    Ks[i],
                    pred_mask_i,
                    images.shape[2],
                    images.shape[1],
                    max_rot_deg=args.refine_max_rot_deg,
                    max_trans_m=args.refine_max_trans_m,
                    passes=args.refine_hard_search_passes,
                )
                if np.isfinite(search_iou) and search_iou > refine_iou1:
                    candidate_w2c = search_w2c
                    candidate_rast = search_rast
                    refine_iou1 = search_iou
                improved = np.isfinite(refine_iou1) and refine_iou1 >= refine_iou0 + args.refine_min_iou_gain
                if improved:
                    w2c_refined = candidate_w2c
                    refined_rast = candidate_rast
                    refined = True

            rows.append(
                PredRow(
                    frame=frame_i,
                    video_time_s=t_s,
                    success=w2c is not None,
                    valid_kp=int(valid.sum()),
                    reproj_px=reproj_px,
                    mean_conf=float(np.nanmean(conf[i])),
                    w2c=w2c,
                    refined=refined,
                    refine_loss0=refine_loss0,
                    refine_loss1=refine_loss1,
                    refine_iou0=refine_iou0,
                    refine_iou1=refine_iou1,
                    w2c_refined=w2c_refined,
                )
            )

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
            if refined_rast is not None:
                cv2.imwrite(
                    str(args.out_dir / "rast_masks_refined" / f"frame_{frame_i:06d}.png"),
                    (np.clip(refined_rast, 0, 1) * 255).astype(np.uint8),
                )
            if candidate_rast is not None:
                cv2.imwrite(
                    str(args.out_dir / "rast_masks_candidates" / f"frame_{frame_i:06d}.png"),
                    (np.clip(candidate_rast, 0, 1) * 255).astype(np.uint8),
                )
            if candidate_w2c is not None:
                np.save(args.out_dir / "w2c_candidates" / f"frame_{frame_i:06d}.npy", candidate_w2c)
            if len(rows) == 1 or (len(rows) - 1) % args.save_every == 0:
                panel = _panel(images[i], uv[i], conf[i], valid, pred_mask_i, rast, refined_rast, candidate_rast)
                cv2.imwrite(
                    str(args.out_dir / "panels" / f"frame_{frame_i:06d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                )

    _write_outputs(args.out_dir, rows)
    _print_summary(rows)
    print(f"[green]wrote outputs to {args.out_dir}[/]")


if __name__ == "__main__":
    main()
