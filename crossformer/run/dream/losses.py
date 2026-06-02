from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.data.geometry import denormalize_kp2d

from .metrics import keypoint_metrics


def _build_heatmaps_one(
    kp2d: jax.Array,
    visible: jax.Array,
    image_h: int,
    image_w: int,
    sigma: float = 2.0,
) -> jax.Array:
    ys = jnp.arange(image_h, dtype=jnp.float32)[:, None]
    xs = jnp.arange(image_w, dtype=jnp.float32)[None, :]
    u = kp2d[:, 0]
    v = kp2d[:, 1]
    pixel_u = u.astype(jnp.int32)
    pixel_v = v.astype(jnp.int32)
    dist2 = (xs - u[:, None, None]) ** 2 + (ys - v[:, None, None]) ** 2
    heatmaps = jnp.exp(-dist2 / (2.0 * sigma**2))
    in_bounds = (pixel_u >= 0) & (pixel_u < image_w) & (pixel_v >= 0) & (pixel_v < image_h)
    mask = visible[:, None, None]
    mask = mask & in_bounds[:, None, None]
    return jnp.where(mask, heatmaps, jnp.zeros_like(heatmaps))


def belief_sigma(sigma_pct: float, out_h: int, out_w: int) -> float:
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid belief-map size: {(out_h, out_w)}")
    return sigma_pct * min(out_h, out_w) / 100.0


def build_heatmaps(
    kp2d: jax.Array,
    visible: jax.Array,
    image_h: int,
    image_w: int,
    sigma: float = 2.0,
) -> jax.Array:
    return jax.vmap(lambda uv, vis: _build_heatmaps_one(uv, vis, image_h=image_h, image_w=image_w, sigma=sigma))(
        kp2d, visible
    )


def focal_heatmap_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet-style focal loss for Gaussian keypoint belief maps.

    See Objects as Points: https://arxiv.org/abs/1904.07850.
    The peak pixel is positive. Every non-peak pixel is negative, but
    (1 - target)^beta makes pixels near the Gaussian peak weak negatives
    instead of punishing them like background.
    """
    pred = jnp.clip(pred, eps, 1.0 - eps)

    peak = target.max(axis=(-2, -1), keepdims=True)
    pos = (target >= peak) & (peak > 0)
    neg = ~pos

    pos_loss = -((1.0 - pred) ** alpha) * jnp.log(pred) * pos
    neg_loss = -((1.0 - target) ** beta) * (pred**alpha) * jnp.log(1.0 - pred) * neg

    n_pos = jnp.maximum(pos.sum(), 1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def mask_loss(pred: jax.Array, target: jax.Array, eps: float = 1e-6) -> tuple[jax.Array, dict[str, jax.Array]]:
    target = target.astype(pred.dtype)
    bce = -(target * jnp.log(pred + eps) + (1.0 - target) * jnp.log1p(-pred + eps)).mean()
    inter = (pred * target).sum(axis=(-2, -1))
    denom = pred.sum(axis=(-2, -1)) + target.sum(axis=(-2, -1))
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()
    return bce + dice, {"mask_bce": bce, "mask_dice": dice}


def dream_loss_fn(batch: dict, out_dict: dict, sigma_pct: float = 1.0, mask_weight: float = 0.1):
    pred = out_dict["pred_heatmaps"]
    preds = pred if isinstance(pred, tuple) else (pred,)
    final_pred = preds[-1]
    _, _, out_h, out_w = final_pred.shape
    uv = denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w)
    vis = batch["keypoints_visible"]
    target = build_heatmaps(
        uv,
        vis,
        image_h=out_h,
        image_w=out_w,
        sigma=belief_sigma(sigma_pct, out_h, out_w),
    )
    stage_losses = tuple(focal_heatmap_loss(stage_pred, target) for stage_pred in preds)
    stage_mses = tuple(jnp.mean((stage_pred - target) ** 2) for stage_pred in preds)
    heatmap_loss = jnp.mean(jnp.stack(stage_losses))
    loss = heatmap_loss
    metrics = {
        "loss": loss,
        "heatmap_loss": heatmap_loss,
        "mse": jnp.mean(jnp.stack(stage_mses)),
        "visible_kp": vis.sum(),
        **{f"stage_{i + 1}_mse": stage_mse for i, stage_mse in enumerate(stage_mses)},
        **{f"stage_{i + 1}_focal": stage_loss for i, stage_loss in enumerate(stage_losses)},
        **keypoint_metrics(batch, final_pred),
    }
    if mask_weight > 0.0 and "pred_mask" in out_dict:
        mask_term, mask_metrics = mask_loss(out_dict["pred_mask"], mask_target(batch["mask"], out_h, out_w))
        loss = heatmap_loss + mask_weight * mask_term
        metrics = {
            **metrics,
            "loss": loss,
            "mask_loss": mask_term,
            "mask_weighted_loss": mask_weight * mask_term,
            **mask_metrics,
        }
    return loss, metrics


def mask_target(mask: jax.Array, out_h: int, out_w: int) -> jax.Array:
    mask = (mask > 0).astype(jnp.float32)
    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    if tuple(mask.shape[-2:]) == (out_h, out_w):
        return mask
    mask = jnp.transpose(mask, (0, 2, 3, 1))
    mask = jax.image.resize(mask, (mask.shape[0], out_h, out_w, mask.shape[-1]), method="nearest")
    return jnp.transpose(mask, (0, 3, 1, 2))
