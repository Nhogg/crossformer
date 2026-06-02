from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from crossformer.data.geometry import normalize_kp2d_np
from crossformer.utils.imaug import (
    apply_augmax_color_batch,
    kp_in_bounds,
    rotate_image_np,
    rotate_keypoints_np,
    translate_image_np,
    zoom_image_np,
)

from .config import Config


def _maybe_apply_grain_imaug(ds, cfg: Config):  # noqa: C901
    if not (cfg.imaug or cfg.rotate):
        return ds

    def aug(batch: dict, rng):
        image = np.asarray(batch["image"]).copy()
        mask = np.asarray(batch["mask"]).copy()
        kp = np.asarray(batch["keypoints_2d_netin"], dtype=np.float32).copy()
        vis = np.asarray(batch["keypoints_visible"], dtype=bool).copy()
        K = np.asarray(batch["K"], dtype=np.float32).copy()
        w2c = np.asarray(batch["w2c"], dtype=np.float32).copy()
        h, w = image.shape[1:3]

        if cfg.imaug:
            base_key = jax.random.key(rng.integers(2**32 - 1, dtype=np.uint32))
            keys = jax.random.split(base_key, image.shape[0])
            imgs_f = jnp.asarray(image, dtype=jnp.float32) / 255.0
            image = np.clip(np.asarray(apply_augmax_color_batch(keys, imgs_f)) * 255.0, 0, 255).astype(np.uint8)

        min_vis = cfg.min_visible_kp
        cx_img, cy_img = np.float32(w) * 0.5, np.float32(h) * 0.5
        for i in range(image.shape[0]):
            if not cfg.rotate:
                continue
            if rng.random() < 0.3:
                angle = float(rng.uniform(-15.0, 15.0))
                kp_i = rotate_keypoints_np(kp[i], angle, h=h, w=w)
                vis_i = vis[i] & kp_in_bounds(kp_i, h, w)
                if int(vis_i.sum()) >= min_vis:
                    image[i] = rotate_image_np(image[i], angle, Image.BILINEAR, fill=0)
                    mask[i] = rotate_image_np(mask[i].astype(np.uint8), angle, Image.NEAREST, fill=0).astype(mask.dtype)
                    a = np.deg2rad(np.float32(-angle))
                    ca, sa = np.cos(a), np.sin(a)
                    Rz = np.eye(4, dtype=np.float32)
                    Rz[0, 0] = ca
                    Rz[0, 1] = -sa
                    Rz[1, 0] = sa
                    Rz[1, 1] = ca
                    w2c[i] = Rz @ w2c[i]
                    vis[i] = vis_i
                    kp[i] = kp_i

            if rng.random() < 0.5:
                tx = float(rng.uniform(-0.1 * w, 0.1 * w))
                ty = float(rng.uniform(-0.1 * h, 0.1 * h))
                kp_i = kp[i].copy()
                kp_i[:, 0] += tx
                kp_i[:, 1] += ty
                vis_i = vis[i] & kp_in_bounds(kp_i, h, w)
                if int(vis_i.sum()) >= min_vis:
                    image[i] = translate_image_np(image[i], tx, ty, resample=Image.BILINEAR, fill=0)
                    mask[i] = translate_image_np(
                        mask[i].astype(np.uint8), tx, ty, resample=Image.NEAREST, fill=0
                    ).astype(mask.dtype)
                    K[i, 0, 2] += np.float32(tx)
                    K[i, 1, 2] += np.float32(ty)
                    vis[i] = vis_i
                    kp[i] = kp_i

            if rng.random() < 0.5:
                scale = float(rng.uniform(0.85, 1.15))
                kp_i = kp[i].copy()
                kp_i[:, 0] = (kp_i[:, 0] - cx_img) * scale + cx_img
                kp_i[:, 1] = (kp_i[:, 1] - cy_img) * scale + cy_img
                vis_i = vis[i] & kp_in_bounds(kp_i, h, w)
                if int(vis_i.sum()) >= min_vis:
                    image[i] = zoom_image_np(image[i], scale, resample=Image.BILINEAR, fill=0)
                    mask[i] = zoom_image_np(mask[i].astype(np.uint8), scale, resample=Image.NEAREST, fill=0).astype(
                        mask.dtype
                    )
                    s = np.float32(scale)
                    K[i, 0, 0] *= s
                    K[i, 1, 1] *= s
                    K[i, 0, 2] = (K[i, 0, 2] - cx_img) * s + cx_img
                    K[i, 1, 2] = (K[i, 1, 2] - cy_img) * s + cy_img
                    vis[i] = vis_i
                    kp[i] = kp_i

        return {
            **batch,
            "image": image,
            "mask": mask,
            "keypoints_2d_netin": kp,
            "keypoints_2d_norm": normalize_kp2d_np(kp, h=h, w=w),
            "keypoints_visible": vis,
            "K": K,
            "w2c": w2c,
        }

    return ds.random_map(aug)
