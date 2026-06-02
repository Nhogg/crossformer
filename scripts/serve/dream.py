from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

from einops import rearrange
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import optax
import orbax.checkpoint as ocp
from PIL import Image
from rich import print
import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

from crossformer.data.geometry import _denormalize_kp2d, _shrink_crop_image_np, _shrink_crop_intrinsics_np
from crossformer.embody import KP2D_NAMES
from crossformer.run.dream.metrics import (
    _mask_iou,
    _pnp_reproj_err,
    _solve_pnp_sqpnp_iter,
    extract_keypoints,
    PNP_MASK_IOU_THRESH,
)
from crossformer.run.dream.modeling import _count_params, _image_to_float, make_model, net_out_size
from crossformer.run.dream.train_steps import final_pred_heatmaps, prepare_pred_heatmaps, prepare_pred_mask
from crossformer.utils.callbacks.synth_viz import _get_robot_mesh, composite_robot, fk_keypoints, rasterize_robot
from crossformer.utils.softras import silhouette
from crossformer.utils.spatial.calibration import solve_stacked_pnp
from crossformer.utils.spatial.solve import levenberg_marquardt_se3_pnp
from crossformer.utils.spec import spec


@dataclass
class DreamModelConfig:
    """DREAM model shape/config fields used for random init."""

    seed: int = 0
    net_in_size: tuple[int, int] = (200, 200)
    image_c: int = 3
    num_keypoints: int = 0  # 0 = xArm DREAM landmarks
    encoder: Literal["vgg", "tips"] = "tips"
    variant: Literal["quarter", "half", "full"] = "full"
    decoder: Literal["auto", "upsample", "deconv", "dpt"] = "dpt"
    tips_variant: str = "tips_v2_b14"
    tips_checkpoint: Path | None = None
    tips_trainable: bool = False
    deconv_decoder: bool | None = None
    full_output: bool | None = None
    skip_connections: bool = False
    n_stages: int = 1
    internalize_spatial_softmax: bool = False
    learned_beta: bool = True
    initial_beta: float = 1.0


@dataclass
class ReturnConfig:
    heatmaps: bool = False  # heatmaps are large; keep off for interactive serving
    mask: bool = True  # return mask when the decoder produces one
    raster: bool = False  # return resized image composited with accepted PnP robot raster
    use_reject: bool = True  # reject accepted poses by mask IoU when a mask is available
    mask_iou_thresh: float = PNP_MASK_IOU_THRESH
    calibration: bool = False  # run stacked PnP across all frames for camera calibration


@dataclass
class DRConfig:
    """Differentiable raster pose/intrinsics refinement."""

    use: bool = False
    steps: int = 32
    lr: float = 1e-3
    final_lr: float = 1e-4
    size: int = 96
    sigma: float = 2e-2
    chunk: int = 256
    cxy_limit: float = 8.0
    kp_weight: float = 0.01
    pose_prior: float = 1e-3
    cxy_prior: float = 1e-2


@dataclass
class Config:
    """Serve a DREAM model."""

    host: str = "0.0.0.0"
    port: int = 8002
    path: Path | None = None  # params checkpoint dir; None keeps random init
    step: int | None = None  # checkpoint step; None loads latest
    warmup: bool = True
    ret: ReturnConfig = field(default_factory=ReturnConfig)
    dr: DRConfig = field(default_factory=DRConfig)
    dream: DreamModelConfig = field(default_factory=DreamModelConfig)


def _image_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.uint8, copy=False)
    scale = 255.0 if np.nanmax(image) <= 1.5 else 1.0
    return np.clip(image * scale, 0, 255).astype(np.uint8)


def _as_batch(x, name: str, ndim: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == ndim - 1:
        x = x[None]
    if x.ndim != ndim:
        raise ValueError(f"expected {name} ndim {ndim - 1} or {ndim}, got {x.shape}")
    return x


def _as_mask_batch(mask) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"expected mask shape (H,W), (B,H,W), (B,1,H,W), or (B,H,W,1), got {mask.shape}")
    if np.issubdtype(mask.dtype, np.floating):
        mask = mask > 0.5
    return mask


def _match_batch(x: np.ndarray, b: int, name: str) -> np.ndarray:
    if x.shape[0] == b:
        return x
    if x.shape[0] == 1:
        return np.repeat(x, b, axis=0)
    raise ValueError(f"{name} batch {x.shape[0]} does not match image batch {b}")


def _shrink_crop_image_batch(image: np.ndarray, net_h: int, net_w: int, resample: int) -> np.ndarray:
    image = _image_u8(image)
    return np.stack([_shrink_crop_image_np(x, net_h, net_w, resample) for x in image], axis=0)


def _shrink_crop_mask_batch(mask: np.ndarray, net_h: int, net_w: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    return np.stack([_shrink_crop_image_np(x, net_h, net_w, Image.NEAREST) > 0 for x in mask], axis=0)


def _shrink_crop_intrinsics_batch(K: np.ndarray, raw_h: int, raw_w: int, net_h: int, net_w: int) -> np.ndarray:
    return np.stack([_shrink_crop_intrinsics_np(k, raw_h, raw_w, net_h, net_w) for k in K], axis=0)


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if tuple(mask.shape[-2:]) == (h, w):
        return mask
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return np.asarray(pil.resize((w, h), Image.NEAREST)) > 0


def _params_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "params":
        return path
    if (path / "params").exists():
        return path / "params"
    return path


def _load_params(path: Path, target_params, step: int | None):
    path = _params_path(path)
    mngr = ocp.CheckpointManager(path)
    step = step if step is not None else mngr.latest_step()
    if step is None:
        raise ValueError(f"no checkpoints found under {path}")
    abstract = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
        target_params,
    )
    print(f"loading DREAM params: path={path} step={step}")
    return mngr.restore(step, args=ocp.args.StandardRestore(abstract))


def _pnp_weights(conf: np.ndarray) -> np.ndarray:
    weights = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    wmax = weights.max(initial=0.0)
    if wmax <= 0.0:
        return np.ones_like(weights, dtype=np.float64)
    return weights / wmax


@jax.jit
def _refine_pose_lm_jax(
    w2c: jax.Array, pts_3d: jax.Array, uv: jax.Array, K: jax.Array, weights: jax.Array
) -> jax.Array:
    T0 = jaxlie.SE3.from_matrix(w2c)
    T, _ = levenberg_marquardt_se3_pnp(T0, K, pts_3d, uv, weights=weights)
    return T.as_matrix()


def _refine_pose_lm(
    w2c: np.ndarray, pts_3d: np.ndarray, uv: np.ndarray, K: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    w2c = _refine_pose_lm_jax(
        jnp.asarray(w2c, dtype=jnp.float32),
        jnp.asarray(pts_3d, dtype=jnp.float32),
        jnp.asarray(uv, dtype=jnp.float32),
        jnp.asarray(K, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
    )
    return np.asarray(jax.device_get(w2c), dtype=np.float64)


def _solve_pose_weighted(q: np.ndarray, uv: np.ndarray, conf: np.ndarray, K: np.ndarray):
    joints_rad = np.deg2rad(np.asarray(q[:7], dtype=np.float64))
    pts_3d = fk_keypoints(joints_rad)
    valid = np.isfinite(uv).all(axis=-1) & (uv[:, 0] > -999.0) & np.isfinite(conf)
    if valid.sum() < 4:
        return joints_rad, valid, None

    w2c = _solve_pnp_sqpnp_iter(pts_3d[valid], uv[valid], K)
    if w2c is not None:
        w2c = _refine_pose_lm(w2c, pts_3d[valid], uv[valid], K, _pnp_weights(conf[valid]))
    return joints_rad, valid, w2c


def _dr_size(h: int, w: int, max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        return h, w
    scale = min(1.0, max_size / max(h, w))
    return max(4, round(h * scale)), max(4, round(w * scale))


def _posed_robot_verts(joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    robot = _get_robot_mesh()
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    return robot.posed_verts(q)[0].astype(np.float32), robot.faces.astype(np.int32)


def _K_with_cxy_delta(K0: jax.Array, raw_cxy: jax.Array, cxy_limit: float) -> tuple[jax.Array, jax.Array]:
    f = 0.5 * (K0[0, 0] + K0[1, 1])
    dcxy = cxy_limit * jnp.tanh(raw_cxy)
    K = jnp.eye(3, dtype=K0.dtype)
    K = K.at[0, 0].set(f)
    K = K.at[1, 1].set(f)
    K = K.at[0, 2].set(K0[0, 2] + dcxy[0])
    K = K.at[1, 2].set(K0[1, 2] + dcxy[1])
    return K, dcxy


def _scale_K(K: jax.Array, h0: int, w0: int, h: int, w: int) -> jax.Array:
    K = K.copy()
    K = K.at[0].multiply(w / w0)
    K = K.at[1].multiply(h / h0)
    return K


def _opencv_projection(K: jax.Array, w: int, h: int) -> jax.Array:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    znear, zfar = 0.01, 10.0
    P = jnp.zeros((4, 4), dtype=K.dtype)
    P = P.at[0, 0].set(2.0 * fx / w)
    P = P.at[1, 1].set(2.0 * fy / h)
    P = P.at[0, 2].set(1.0 - 2.0 * cx / w)
    P = P.at[1, 2].set(2.0 * cy / h - 1.0)
    P = P.at[2, 2].set(-(zfar + znear) / (zfar - znear))
    P = P.at[2, 3].set(-2.0 * zfar * znear / (zfar - znear))
    P = P.at[3, 2].set(-1.0)
    return P


def _project_points_jax(T: jaxlie.SE3, K: jax.Array, pts_3d: jax.Array) -> jax.Array:
    x_cam = jax.vmap(T.apply)(pts_3d)
    z = jnp.clip(x_cam[:, 2], 1e-6)
    u = K[0, 0] * x_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * x_cam[:, 1] / z + K[1, 2]
    return jnp.stack([u, v], axis=-1)


def _robust_kp_loss(pred: jax.Array, obs: jax.Array, weights: jax.Array) -> jax.Array:
    err = jnp.linalg.norm(pred - obs, axis=-1)
    loss = jnp.sqrt(err * err + 1.0) - 1.0
    denom = jnp.maximum(weights.sum(), 1e-6)
    return (loss * weights).sum() / denom


def _mask_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    eps = 1e-6
    pred = jnp.clip(pred, eps, 1.0 - eps)
    bce = -(target * jnp.log(pred) + (1.0 - target) * jnp.log1p(-pred)).mean()
    inter = (pred * target).sum()
    dice = 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)
    return bce + dice


@partial(
    jax.jit,
    static_argnames=(
        "h0",
        "w0",
        "h",
        "w",
        "steps",
        "lr",
        "final_lr",
        "sigma",
        "chunk",
        "cxy_limit",
        "kp_weight",
        "pose_prior",
        "cxy_prior",
    ),
)
def _dr_refine_jax(
    w2c0: jax.Array,
    K0: jax.Array,
    verts: jax.Array,
    faces: jax.Array,
    target: jax.Array,
    pts_3d: jax.Array,
    uv: jax.Array,
    weights: jax.Array,
    *,
    h0: int,
    w0: int,
    h: int,
    w: int,
    steps: int,
    lr: float,
    final_lr: float,
    sigma: float,
    chunk: int,
    cxy_limit: float,
    kp_weight: float,
    pose_prior: float,
    cxy_prior: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    T0 = jaxlie.SE3.from_matrix(w2c0)
    opt = jnp.zeros((8,), dtype=w2c0.dtype)
    alpha = jnp.clip(final_lr / lr, 0.0, 1.0)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(lr, decay_steps=steps, alpha=alpha)),
    )
    opt_state = tx.init(opt)

    flip = jnp.diag(jnp.array([1.0, -1.0, -1.0, 1.0], dtype=w2c0.dtype))

    def render(params):
        T = jaxlie.manifold.rplus(T0, params[:6])
        K_full, dcxy = _K_with_cxy_delta(K0, params[6:], cxy_limit)
        K = _scale_K(K_full, h0, w0, h, w)
        mvp = _opencv_projection(K, w, h) @ (flip @ T.as_matrix())
        clip = jnp.einsum("vi,ji->vj", verts, mvp)
        mask = silhouette(clip, faces, h, w, sigma=sigma, chunk=chunk)
        return T, K_full, K, dcxy, mask

    def loss_fn(params):
        T, _K_full, K, dcxy, pred = render(params)
        loss = _mask_loss(pred, target)
        if kp_weight > 0.0:
            uv_dr = uv * jnp.array([w / w0, h / h0], dtype=uv.dtype)
            loss = loss + kp_weight * _robust_kp_loss(_project_points_jax(T, K, pts_3d), uv_dr, weights)
        loss = loss + pose_prior * jnp.mean(params[:6] ** 2)
        loss = loss + cxy_prior * jnp.mean((dcxy / cxy_limit) ** 2)
        return loss

    def step_fn(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), grads)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        params = jnp.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        return (params, state), loss

    init_loss = loss_fn(opt)
    (opt, _opt_state), losses = jax.lax.scan(step_fn, (opt, opt_state), None, length=steps)
    T, K_full, _K, dcxy, mask = render(opt)
    final_loss = loss_fn(opt)
    return T.as_matrix(), K_full, mask, dcxy, jnp.concatenate([jnp.array([init_loss, final_loss]), losses])


def _dr_refine_pose(
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    pts_3d: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    cfg: DRConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h0, w0 = mask.shape[-2:]
    h, w = _dr_size(h0, w0, cfg.size)
    target = _resize_mask(mask, h, w).astype(np.float32)
    verts, faces = _posed_robot_verts(joints_rad)
    w2c, K, _pred, dcxy, losses = _dr_refine_jax(
        jnp.asarray(w2c, dtype=jnp.float32),
        jnp.asarray(K, dtype=jnp.float32),
        jnp.asarray(verts, dtype=jnp.float32),
        jnp.asarray(faces, dtype=jnp.int32),
        jnp.asarray(target, dtype=jnp.float32),
        jnp.asarray(pts_3d, dtype=jnp.float32),
        jnp.asarray(uv, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
        h0=h0,
        w0=w0,
        h=h,
        w=w,
        steps=cfg.steps,
        lr=cfg.lr,
        final_lr=cfg.final_lr,
        sigma=cfg.sigma,
        chunk=cfg.chunk,
        cxy_limit=cfg.cxy_limit,
        kp_weight=cfg.kp_weight,
        pose_prior=cfg.pose_prior,
        cxy_prior=cfg.cxy_prior,
    )
    info = {
        "dr_loss_init": float(losses[0]),
        "dr_loss_final": float(losses[1]),
        "dr_cxy_delta": np.asarray(jax.device_get(dcxy), dtype=np.float32),
    }
    return (
        np.asarray(jax.device_get(w2c), dtype=np.float64),
        np.asarray(jax.device_get(K), dtype=np.float64),
        info,
    )


def _extrinsics_from_keypoints(
    payload: dict,
    out: dict,
    *,
    image: np.ndarray | None = None,
    use_reject: bool,
    mask_iou_thresh: float,
    dr: DRConfig,
) -> dict:
    uv = np.asarray(out["keypoints"], dtype=np.float64)
    conf = np.asarray(out["confidence"], dtype=np.float64)
    b = uv.shape[0]
    q = _match_batch(_as_batch(payload["q"], "q", 2).astype(np.float64), b, "q")
    K = _match_batch(_as_batch(payload["K"], "K", 3).astype(np.float64), b, "K")
    raster_image = None if image is None else _match_batch(_image_u8(image), b, "image").copy()

    mask_ref = None
    if use_reject or dr.use:
        if "mask" in payload:
            mask_ref = _as_mask_batch(payload["mask"])
        elif "_mask" in out:
            mask_ref = _as_mask_batch(out["_mask"])
    if mask_ref is not None:
        mask_ref = _match_batch(mask_ref, b, "mask")

    w2c = np.full((b, 4, 4), np.nan, dtype=np.float64)
    K_out = K.copy()
    valid = np.zeros((b, uv.shape[1]), dtype=bool)
    success = np.zeros((b,), dtype=bool)
    reproj = np.full((b,), np.nan, dtype=np.float64)
    mask_iou = np.full((b,), np.nan, dtype=np.float64)
    mask_iou_reject = np.zeros((b,), dtype=bool)
    dr_success = np.zeros((b,), dtype=bool)
    dr_loss_init = np.full((b,), np.nan, dtype=np.float64)
    dr_loss_final = np.full((b,), np.nan, dtype=np.float64)
    dr_cxy_delta = np.full((b, 2), np.nan, dtype=np.float64)

    for i in range(b):
        joints_rad, valid_i, w2c_i = _solve_pose_weighted(q[i], uv[i], conf[i], K[i])
        valid[i] = valid_i
        if w2c_i is None:
            continue

        K_i = K[i]
        if dr.use and mask_ref is not None:
            pts_3d = fk_keypoints(joints_rad)
            w2c_i, K_i, dr_info = _dr_refine_pose(
                joints_rad,
                w2c_i,
                K_i,
                mask_ref[i],
                pts_3d[valid_i],
                uv[i][valid_i],
                _pnp_weights(conf[i][valid_i]),
                dr,
            )
            dr_success[i] = True
            dr_loss_init[i] = dr_info["dr_loss_init"]
            dr_loss_final[i] = dr_info["dr_loss_final"]
            dr_cxy_delta[i] = dr_info["dr_cxy_delta"]

        K_out[i] = K_i
        reproj_i = _pnp_reproj_err(w2c_i, joints_rad, uv[i], valid_i, K_i)
        reproj[i] = reproj_i

        ok = True
        rast = None
        if raster_image is not None:
            h, w = raster_image[i].shape[:2]
            rast = rasterize_robot(joints_rad, w2c_i, K_i, w, h)
        elif use_reject and mask_ref is not None:
            h, w = mask_ref[i].shape[-2:]
            rast = rasterize_robot(joints_rad, w2c_i, K_i, w, h)

        if use_reject and mask_ref is not None and rast is not None:
            mask = _resize_mask(mask_ref[i], *rast.shape)
            mask_iou[i] = _mask_iou(rast, mask)
            if np.isfinite(mask_iou[i]) and mask_iou[i] < mask_iou_thresh:
                mask_iou_reject[i] = True
                ok = False

        if ok:
            success[i] = True
            w2c[i] = w2c_i
            if raster_image is not None and rast is not None:
                raster_image[i] = composite_robot(raster_image[i], rast)

    result = {
        "w2c": w2c.astype(np.float32),
        "K": K_out.astype(np.float32),
        "pnp_success": success,
        "pnp_valid": valid,
        "pnp_reproj_px": reproj.astype(np.float32),
        "mask_iou": mask_iou.astype(np.float32),
        "mask_iou_reject": mask_iou_reject,
        "dr_success": dr_success,
        "dr_loss_init": dr_loss_init.astype(np.float32),
        "dr_loss_final": dr_loss_final.astype(np.float32),
        "dr_cxy_delta": dr_cxy_delta.astype(np.float32),
    }
    if raster_image is not None:
        result["raster_image"] = raster_image
    return result


class DreamPolicy(BasePolicy):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_cfg = cfg.dream
        self.net_h, self.net_w = self.model_cfg.net_in_size
        self.out_h, self.out_w = net_out_size(self.model_cfg)
        self.num_keypoints = self.model_cfg.num_keypoints or len(KP2D_NAMES)
        self.model = make_model(self.model_cfg, self.num_keypoints)

        rng = jax.random.PRNGKey(self.model_cfg.seed)
        image = jnp.zeros((1, self.net_h, self.net_w, self.model_cfg.image_c), dtype=jnp.float32)
        self.params = self.model.init(rng, image)["params"]
        if self.cfg.path is not None:
            self.params = _load_params(self.cfg.path, self.params, self.cfg.step)

        @jax.jit
        def _predict(params, image):
            model_out, _ = self.model.apply({"params": params}, _image_to_float(image))
            heatmaps = final_pred_heatmaps(prepare_pred_heatmaps(model_out, self.out_h, self.out_w))
            uv, conf = extract_keypoints(heatmaps)
            uv_norm = uv / jnp.array([self.out_w, self.out_h], dtype=jnp.float32)
            out = {
                "keypoints": _denormalize_kp2d(uv_norm, self.net_h, self.net_w),
                "keypoints_norm": uv_norm,
                "confidence": conf,
            }
            mask = prepare_pred_mask(model_out, self.out_h, self.out_w)
            if mask is not None:
                out["_mask"] = mask
                if self.cfg.ret.mask:
                    out["mask"] = mask
            if self.cfg.ret.heatmaps:
                out["heatmaps"] = heatmaps
            return out

        self._predict = _predict
        source = "checkpoint" if self.cfg.path is not None else "random"
        print(f"loaded {source} DREAM params: {_count_params(self.params):,}")
        print(
            f"input_size={(self.net_h, self.net_w)} output_size={(self.out_h, self.out_w)} keypoints={self.num_keypoints}"
        )
        if self.cfg.warmup:
            self.warmup()

    def warmup(self):
        image = np.zeros((1, self.net_h, self.net_w, self.model_cfg.image_c), dtype=np.uint8)
        q = np.zeros((1, 8), dtype=np.float32)
        K = np.array([[[600.0, 0.0, self.net_w / 2], [0.0, 600.0, self.net_h / 2], [0.0, 0.0, 1.0]]], dtype=np.float32)
        print("warmup input spec")
        print(spec({"image": image, "q": q, "K": K}))
        out = self.step({"image": image, "q": q, "K": K})
        print("warmup output spec")
        print(spec(out))

    def reset(self, payload: dict | None = None) -> dict:
        return {"reset": True}

    def step(self, payload: dict) -> dict:
        if payload.get("reset", False):
            return self.reset(payload)
        raw_image = rearrange(np.asarray(payload["image"]), "... h w c -> (...) h w c")
        raw_h, raw_w = raw_image.shape[1:3]
        image = _shrink_crop_image_batch(raw_image, self.net_h, self.net_w, Image.BILINEAR)

        payload_net = dict(payload)
        K = _match_batch(_as_batch(payload["K"], "K", 3).astype(np.float64), image.shape[0], "K")
        payload_net["K"] = _shrink_crop_intrinsics_batch(K, raw_h, raw_w, self.net_h, self.net_w)
        if "mask" in payload:
            mask = _match_batch(_as_mask_batch(payload["mask"]), image.shape[0], "mask")
            payload_net["mask"] = _shrink_crop_mask_batch(mask, self.net_h, self.net_w)

        out = self._predict(self.params, jnp.asarray(image))
        out = jax.device_get(out)
        raster_image = image if self.cfg.ret.raster else None
        out.update(
            _extrinsics_from_keypoints(
                payload_net,
                out,
                image=raster_image,
                use_reject=self.cfg.ret.use_reject,
                mask_iou_thresh=self.cfg.ret.mask_iou_thresh,
                dr=self.cfg.dr,
            )
        )
        out.pop("_mask", None)

        if payload.get("calibrate") or self.cfg.ret.calibration:
            out.update(self._stacked_calibration(payload_net, out))

        return out

    def _stacked_calibration(self, payload_net: dict, out: dict) -> dict:
        uv = np.asarray(out["keypoints"], dtype=np.float64)
        conf = np.asarray(out["confidence"], dtype=np.float64)
        q = _match_batch(
            _as_batch(payload_net["q"], "q", 2).astype(np.float64),
            uv.shape[0],
            "q",
        )
        K = np.asarray(payload_net["K"][0], dtype=np.float64)

        pts_3d = np.stack([fk_keypoints(np.deg2rad(q_i[:7])) for q_i in q], axis=0)
        valid = np.isfinite(conf) & (conf > 0.01)
        calib = solve_stacked_pnp(pts_3d, uv, K, valid)

        return {
            "calib_w2c": (
                calib.w2c.astype(np.float32) if calib.w2c is not None else np.full((4, 4), np.nan, dtype=np.float32)
            ),
            "calib_success": np.array(calib.success, dtype=bool),
            "calib_valid": calib.valid,
            "calib_reproj_px": np.float32(calib.reproj_px),
            "calib_n_points": np.int32(calib.n_points),
        }


def main(cfg: Config):
    policy = DreamPolicy(cfg)
    server = Server(policy, host=cfg.host, port=cfg.port, metadata=None)
    print("serving DREAM on", cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
