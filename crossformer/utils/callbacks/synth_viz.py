"""Callback for visualizing predicted 2D keypoints + translucent robot overlay.

Pipeline per sample:
  1. Extract joints, kp2d, cam_intr from bundled prediction
  2. Denormalize joints (z-score → degrees → radians)
  3. FK(joints) → 3D keypoint positions
  4. Unscale cam_intr → build K matrix
  5. solvePnP(3D, 2D, K) → recover camera extrinsics
  6. Rasterize translucent robot mesh with recovered camera
  7. Composite onto upscaled image, then overlay kp2d points
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import jax.numpy as jnp
import numpy as np

from crossformer.data.grain.metadata import DatasetStatistics
from crossformer.embody import DOF, KP2D_NAMES
import wandb

# ---------------------------------------------------------------------------
# DOF ID lookups
# ---------------------------------------------------------------------------

KP2D_DOF_NAMES = tuple(f"kp2d_{n}_{ax}" for n in KP2D_NAMES for ax in ("u", "v"))
KP2D_IDS = tuple(DOF[name] for name in KP2D_DOF_NAMES)
JOINT_IDS = tuple(DOF[f"j{i}"] for i in range(7))
GRIPPER_ID = DOF["gripper"]
CAM_INTR_IDS = tuple(DOF[n] for n in ("cam_fx", "cam_fy", "cam_cx", "cam_cy"))

# landmark → URDF link name (order matches KP2D_NAMES)
LANDMARK_LINKS = (
    "link_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "link_eef",
    "link_tcp",
)

# cam_intr scaling constants (must match restructure_xarm_dream)
FX_MIN, FX_MAX = 450.0, 900.0
IMG_W, IMG_H = 640, 480

GT_COLOR = (0, 255, 0)  # green (BGR)
PRED_COLOR = (0, 0, 255)  # red
ROBOT_COLOR = np.array([0.2, 0.4, 0.9], dtype=np.float32)
ROBOT_ALPHA = 0.6


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_slots(act_base: np.ndarray, act_id: np.ndarray, dof_ids: tuple[int, ...]) -> np.ndarray | None:
    """Extract specific DOF slots from bundled action. Returns (..., len(dof_ids))."""
    id_arr = np.asarray(act_id)
    if id_arr.ndim == 1:
        id_arr = id_arr[None]
    sample_ids = id_arr[0]
    indices = []
    for dof_id in dof_ids:
        idx = np.where(sample_ids == dof_id)[0]
        if len(idx) == 0:
            return None
        indices.append(idx[0])
    return np.asarray(act_base)[..., indices]


def extract_kp2d_from_bundled(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract kp2d as (B, ..., 10, 2)."""
    flat = _extract_slots(act_base, act_id, KP2D_IDS)
    if flat is None:
        return None
    return flat.reshape(*flat.shape[:-1], 10, 2)


def extract_joints_from_bundled(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract 7-DOF joints from bundled action."""
    return _extract_slots(act_base, act_id, JOINT_IDS)


def extract_cam_intr_from_bundled(act_base: np.ndarray, act_id: np.ndarray) -> np.ndarray | None:
    """Extract cam_intr (fx, fy, cx, cy) from bundled action."""
    return _extract_slots(act_base, act_id, CAM_INTR_IDS)


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------


def denorm_joints(joints: np.ndarray, stats: DatasetStatistics) -> np.ndarray:
    """Denormalize joints from z-score back to degrees."""
    s = stats.action.get("joints") or stats.proprio.get("joints")
    if s is None:
        return joints
    return s.unnormalize(joints)


def unscale_cam_intr(cam: np.ndarray) -> np.ndarray:
    """Reverse the manual [0,1] scaling applied in restructure_xarm_dream.

    Input:  (fx_s, fy_s, cx_s, cy_s) in [0, 1]
    Output: (fx, fy, cx, cy) in pixel units
    """
    fx = cam[..., 0] * (FX_MAX - FX_MIN) + FX_MIN
    fy = cam[..., 1] * (FX_MAX - FX_MIN) + FX_MIN
    cx = cam[..., 2] * IMG_W
    cy = cam[..., 3] * IMG_H
    return np.stack([fx, fy, cx, cy], axis=-1)


def build_K(cam: np.ndarray) -> np.ndarray:
    """Build 3x3 intrinsic matrix from (fx, fy, cx, cy)."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = cam[0]
    K[1, 1] = cam[1]
    K[0, 2] = cam[2]
    K[1, 2] = cam[3]
    return K


# ---------------------------------------------------------------------------
# FK + PnP
# ---------------------------------------------------------------------------


def _get_robot_mesh():
    """Lazy-load shared _RobotMesh instance."""
    from crossformer.utils.callbacks.rast import _RobotMesh

    if not hasattr(_get_robot_mesh, "_cache"):
        _get_robot_mesh._cache = _RobotMesh(Path("xarm7_standalone.urdf"), Path("assets"))
    return _get_robot_mesh._cache


def fk_keypoints(joints_rad: np.ndarray, robot=None) -> np.ndarray:
    """Run FK and extract 3D positions for the 10 landmarks.

    Args:
        joints_rad: (7,) joint angles in radians
    Returns:
        (10, 3) world-frame 3D positions
    """
    from crossformer.utils.callbacks.rast import _poses_to_mats

    if robot is None:
        robot = _get_robot_mesh()
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    poses = robot._fk(jnp.asarray(q))
    mats = np.asarray(_poses_to_mats(poses))[0]  # (num_links, 4, 4)
    pts = []
    for link_name in LANDMARK_LINKS:
        idx = robot.link_index[link_name]
        pts.append(mats[idx, :3, 3])
    return np.stack(pts)  # (10, 3)


def solve_pnp(pts_3d: np.ndarray, pts_2d_px: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Solve PnP → 4x4 world-to-camera matrix, or None on failure.

    Args:
        pts_3d: (N, 3) world frame
        pts_2d_px: (N, 2) pixel coordinates
        K: (3, 3) intrinsic matrix
    """
    if pts_3d.shape[0] < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        pts_3d.astype(np.float64),
        pts_2d_px.astype(np.float64),
        K,
        None,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if ok:
        ok, rvec, tvec = cv2.solvePnP(
            pts_3d.astype(np.float64),
            pts_2d_px.astype(np.float64),
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec,
            tvec=tvec,
        )

    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    return w2c


# Dream style random sample consensur
def solve_pnp_ransac(
    pts_3d: np.ndarray, pts_2d_px: np.ndarray, K: np.ndarray, inlier_thresh_px: float = 5.0
) -> np.ndarray | None:
    if pts_3d.shape[0] < 4:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float64),
        pts_2d_px.astype(np.float64),
        K,
        np.array([]),
        reprojectionError=inlier_thresh_px,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inliers is None:
        return None
    inlier_idx = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(
        pts_3d[inlier_idx].astype(np.float64),
        pts_2d_px[inlier_idx].astype(np.float64),
        K,
        np.array([]),
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=True,
        rvec=rvec,
        tvec=tvec,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec.ravel()
    return w2c


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_robot(
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    robot=None,
    gripper_rad: float | None = None,
) -> np.ndarray:
    """Render robot silhouette mask (H, W) using nvdiffrast.

    Returns float32 mask in [0, 1].
    """
    from crossformer.utils.callbacks.rast import _GpuRasterizer
    from crossformer.utils.rig import GRIPPER_CLOSED_RAD

    if robot is None:
        robot = _get_robot_mesh()

    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    # Match render_robot_mask / Blender GT: gripper has its own DOF and must be set,
    # otherwise the rendered silhouette around the fingers shrinks vs. the GT mask.
    q[0, 7] = float(GRIPPER_CLOSED_RAD if gripper_rad is None else gripper_rad)
    verts = robot.posed_verts(q)  # (1, V, 4) homogeneous world-frame

    # build clip-space MVP from intrinsics + extrinsics
    # OpenCV intrinsics → OpenGL-style projection
    znear, zfar = 0.01, 10.0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0

    # OpenCV → OpenGL camera convention (flip Y and Z)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    view = flip @ w2c

    mvp = (P @ view).astype(np.float32)
    clip = np.einsum("bvi,ji->bvj", verts, mvp)

    rasterizer = _GpuRasterizer(robot.faces)
    mask = rasterizer.render_masks(clip, width, height)  # (1, H, W)
    return mask[0]


def composite_robot(img: np.ndarray, mask: np.ndarray, alpha: float = ROBOT_ALPHA) -> np.ndarray:
    """Blend translucent robot color onto image using mask."""
    if img.shape[-1] == 4:
        img = img[..., :3]  # strip alpha
    img = img.astype(np.float32) / 255.0
    w = mask[:, :, None] * alpha
    color = ROBOT_COLOR[None, None, :]
    blended = img * (1.0 - w) + color * w
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_kp2d(
    img: np.ndarray,
    kp2d: np.ndarray,
    render_h: int,
    render_w: int,
    colors: list[tuple[int, int, int]],
    radius: int = 4,
    thickness: int = 2,
    label: bool = True,
    skeleton: bool = True,
) -> np.ndarray:
    for i, (u, v) in enumerate(kp2d):
        if u == 0.0 and v == 0.0:
            continue
        px, py = int(u * render_w), int(v * render_h)
        color = colors[i % len(colors)]
        cv2.circle(img, (px, py), radius, color, thickness)
        if label:
            cv2.putText(img, KP2D_NAMES[i], (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    if skeleton:
        for i in range(len(kp2d) - 1):
            u0, v0 = kp2d[i]
            u1, v1 = kp2d[i + 1]
            if (u0 == 0 and v0 == 0) or (u1 == 0 and v1 == 0):
                continue
            p0 = (int(u0 * render_w), int(v0 * render_h))
            p1 = (int(u1 * render_w), int(v1 * render_h))
            cv2.line(img, p0, p1, colors[0], 1)
    return img


def overlay_kp2d(
    img: np.ndarray,
    kp2d: np.ndarray,
    render_h: int = 480,
    render_w: int = 640,
    pred_kp2d: np.ndarray | None = None,
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    img = np.asarray(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = cv2.resize(img, (render_w, render_h))
    img = img.copy()
    img = _draw_kp2d(img, kp2d, render_h, render_w, [GT_COLOR] * 10, radius, thickness, label=True, skeleton=True)
    if pred_kp2d is not None:
        img = _draw_kp2d(
            img, pred_kp2d, render_h, render_w, [PRED_COLOR] * 10, radius + 1, 1, label=False, skeleton=True
        )
    return img


# ---------------------------------------------------------------------------
# Full render pipeline for one sample
# ---------------------------------------------------------------------------


def render_sample(
    img: np.ndarray,
    gt_kp2d: np.ndarray,
    pred_kp2d: np.ndarray,
    pred_joints_deg: np.ndarray,
    pred_cam_scaled: np.ndarray,
    render_h: int = 480,
    render_w: int = 640,
) -> np.ndarray:
    """Full pipeline: rasterize robot + overlay kp2d.

    Args:
        img: (H, W, 3) uint8 source image (any size)
        gt_kp2d: (10, 2) normalized [0,1]
        pred_kp2d: (10, 2) normalized [0,1]
        pred_joints_deg: (7,) joint angles in degrees
        pred_cam_scaled: (4,) scaled cam_intr [0,1]
    """
    img = np.asarray(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = cv2.resize(img, (render_w, render_h)).copy()

    # recover camera
    cam_px = unscale_cam_intr(pred_cam_scaled)
    K = build_K(cam_px)
    joints_rad = np.deg2rad(pred_joints_deg.astype(np.float64))

    # FK → 3D keypoints
    try:
        robot = _get_robot_mesh()
        pts_3d = fk_keypoints(joints_rad, robot)

        # PnP
        pts_2d_px = pred_kp2d.copy()
        pts_2d_px[:, 0] *= render_w
        pts_2d_px[:, 1] *= render_h
        w2c = solve_pnp(pts_3d, pts_2d_px, K)

        # rasterize + composite
        if w2c is not None:
            mask = rasterize_robot(joints_rad, w2c, K, render_w, render_h, robot)
            img = composite_robot(img, mask)
    except Exception:
        pass  # fall through to kp2d-only overlay

    # overlay keypoints on top
    img = _draw_kp2d(img, gt_kp2d, render_h, render_w, [GT_COLOR] * 10, 4, 2, label=True, skeleton=True)
    img = _draw_kp2d(img, pred_kp2d, render_h, render_w, [PRED_COLOR] * 10, 5, 1, label=False, skeleton=True)
    return img


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@dataclass
class SynthVizCallback:
    """Overlay predicted 2D keypoints + translucent robot on images."""

    every: int = 5000
    render_h: int = 480
    render_w: int = 640
    max_samples: int = 8
    stats: DatasetStatistics | None = None

    def __call__(self, batch: dict, pred: np.ndarray) -> dict:
        act_id = np.asarray(batch["act"]["id"])
        act_base = np.asarray(batch["act"]["base"])

        pred_kp = extract_kp2d_from_bundled(pred, act_id)
        if pred_kp is None:
            return {}
        gt_kp = extract_kp2d_from_bundled(act_base, act_id)

        pred_joints = extract_joints_from_bundled(pred, act_id)
        pred_cam = extract_cam_intr_from_bundled(pred, act_id)

        # denormalize joints if stats available
        if pred_joints is not None and self.stats is not None:
            pred_joints = denorm_joints(pred_joints, self.stats)

        images = np.asarray(batch["observation"]["image_primary"])
        B = min(images.shape[0], self.max_samples)

        panels = []
        for i in range(B):
            img = images[i, 0] if images.ndim == 5 else images[i]
            gt_uv = gt_kp[i, 0, 0] if gt_kp.ndim == 5 else gt_kp[i, 0]
            pred_uv = pred_kp[i, 0, 0] if pred_kp.ndim == 5 else pred_kp[i, 0]

            # try full render with robot overlay
            if pred_joints is not None and pred_cam is not None:
                j = (
                    pred_joints[i, 0, 0]
                    if pred_joints.ndim >= 4
                    else pred_joints[i, 0]
                    if pred_joints.ndim == 3
                    else pred_joints[i]
                )
                c = pred_cam[i, 0, 0] if pred_cam.ndim >= 4 else pred_cam[i, 0] if pred_cam.ndim == 3 else pred_cam[i]
                panel = render_sample(img, gt_uv, pred_uv, j, c, self.render_h, self.render_w)
            else:
                panel = overlay_kp2d(img, gt_uv, self.render_h, self.render_w, pred_kp2d=pred_uv)

            panels.append(wandb.Image(panel, caption=f"sample {i} | GT=green, Pred=red"))

        return {"synth_kp2d": panels}
