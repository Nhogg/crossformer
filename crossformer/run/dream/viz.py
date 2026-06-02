from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crossformer.data.geometry import denormalize_kp2d
from crossformer.utils.callbacks.synth_viz import composite_robot, rasterize_robot
from crossformer.utils.imaug import kp_render_mask, plot_image
import wandb

from .config import Config
from .losses import belief_sigma, build_heatmaps
from .metrics import (
    _mask_iou,
    _pnp_reproj_err,
    _solve_pose_one,
    extract_keypoints,
    PNP_MASK_IOU_THRESH,
)
from .train_steps import final_pred_heatmaps


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
    gt_mask = vis & kp_render_mask(uv_gt, h, w)
    pred_mask = kp_render_mask(uv_pred, h, w) & (uv_pred[:, 0] > -999.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(plot_image(image))
    if gt_mask.any():
        axes[0].scatter(uv_gt[gt_mask, 0], uv_gt[gt_mask, 1], c="lime", s=20, label="gt")
    if pred_mask.any():
        axes[0].scatter(uv_pred[pred_mask, 0], uv_pred[pred_mask, 1], c="red", s=20, label="pred")
    axes[0].set_title("image + keypoints")
    if gt_mask.any() or pred_mask.any():
        axes[0].legend()
    axes[0].axis("off")

    axes[1].imshow(plot_image(hm.max(axis=0)), cmap="magma")
    axes[1].set_title("max heatmap")
    axes[1].axis("off")

    axes[2].imshow(plot_image(image))
    axes[2].imshow(plot_image(hm.max(axis=0)), cmap="magma", alpha=0.5, extent=(0, image.shape[1], image.shape[0], 0))
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
        try:
            gripper_rad = float(q[7]) if q.shape[-1] > 7 else None
            mask = rasterize_robot(joints_rad, w2c, K, image.shape[1], image.shape[0], gripper_rad=gripper_rad)
            if "mask" in batch:
                mask_iou = _mask_iou(mask, np.asarray(batch["mask"][idx]))
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
    ax.imshow(plot_image(panel))
    valid_mask = valid & kp_render_mask(uv, image.shape[0], image.shape[1])
    invalid_mask = (~valid) & kp_render_mask(uv, image.shape[0], image.shape[1]) & (uv[:, 0] > -999.0)
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
    pred_uv = denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    gt_uv = denormalize_kp2d(batch["keypoints_2d_norm"], *cfg.net_in_size)
    gt_heatmaps = build_heatmaps(
        denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w),
        batch["keypoints_visible"],
        image_h=out_h,
        image_w=out_w,
        sigma=belief_sigma(cfg.sigma_pct, out_h, out_w),
    )
    log = {
        f"{prefix}/predictions": _render_overlay(
            {
                "image": jax.device_get(batch["image"]),
                "keypoints_2d": jax.device_get(gt_uv),
                "keypoints_visible": jax.device_get(batch["keypoints_visible"]),
            },
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
            jax.device_get(pred_heatmaps),
        ),
        f"{prefix}/gt": _render_heatmap_overlay(jax.device_get(batch["image"][0]), jax.device_get(gt_heatmaps[0])),
        f"{prefix}/mask": _render_mask_overlay(
            jax.device_get(batch["image"]),
            jax.device_get(batch["mask"]),
            title="gt mask",
        ),
        f"{prefix}/pose_overlay": _render_pose_overlay(
            jax.device_get(batch),
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
        ),
    }
    if "pred_mask" in out_dict:
        log[f"{prefix}/pred_mask"] = _render_mask_overlay(
            jax.device_get(batch["image"]),
            jax.device_get(out_dict["pred_mask"]),
            title="pred mask",
        )
    cfg.wandb.log(log, step=step)
