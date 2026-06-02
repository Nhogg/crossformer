from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from crossformer.data.geometry import denormalize_kp2d
from crossformer.utils.callbacks.synth_viz import fk_keypoints, rasterize_robot, solve_pnp

from .constants import (
    ADD_THRESHOLDS_MM,
    KP_CONF_THRESHOLD,
    KP_MISSING_VALUE,
    KP_PEAK_AMBIGUITY_GAP,
    KP_PEAK_THRESHOLD,
    KP_SMOOTH_RADIUS,
    KP_SMOOTH_SIGMA,
)

if TYPE_CHECKING:
    from .config import Config


def keypoint_metrics(batch: dict, pred_heatmaps: jax.Array):
    pred_uv, pred_conf = extract_keypoints(pred_heatmaps)
    _, _, out_h, out_w = pred_heatmaps.shape
    pred_uv = denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *batch["image"].shape[1:3])
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
PNP_REPROJ_THRESH = 30.0
PNP_MASK_IOU_THRESH = 0.45


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

    if _pnp_reproj_err(w2c, joints_rad, uv_px, valid, K) > PNP_REPROJ_THRESH:
        w2c = None
    return joints_rad, valid, w2c


def _pose_metrics_one(q, uv_px, conf, K, kp2d_gt_px, kp_vis_gt, gt_mask: np.ndarray | None = None) -> dict:
    joints_rad, valid, w2c_pred = _solve_pose_one(q, uv_px, conf, K)
    pts_3d = fk_keypoints(joints_rad)
    out = {"valid_kp": float(valid.sum()), "success": 0.0, "valid_mask": valid.astype(np.float32)}
    if w2c_pred is None:
        return out

    reproj = _project_points(w2c_pred, pts_3d, K)
    reproj_err = float(np.linalg.norm(reproj[valid] - uv_px[valid], axis=-1).mean())
    out = {**out, "success": 1.0, "reproj_px": reproj_err}

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
    pred_uv = denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    batch_np = jax.device_get(batch)
    uv_np = np.asarray(jax.device_get(pred_uv), dtype=np.float64)
    conf_np = np.asarray(jax.device_get(conf), dtype=np.float64)
    q_np = np.asarray(batch_np["q"], dtype=np.float64)
    K_np = np.asarray(batch_np["K"], dtype=np.float64)
    kp2d_gt_np = np.asarray(batch_np["keypoints_2d_netin"], dtype=np.float64)
    kp_vis_np = np.asarray(batch_np["keypoints_visible"])
    mask_np = np.asarray(batch_np["mask"])

    rows = [
        _pose_metrics_one(q_np[i], uv_np[i], conf_np[i], K_np[i], kp2d_gt_np[i], kp_vis_np[i], gt_mask=mask_np[i])
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
    pred_uv, conf = extract_keypoints(out_dict["pred_heatmaps"])
    _, _, out_h, out_w = out_dict["pred_heatmaps"].shape
    pred_uv = denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
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
