"""Static rig calibration and CPU robot-silhouette mask rendering.

Provides per-camera (K, w2c) for the xgym rig and a thin wrapper around the
in-tree CPU mask renderer for use inside grain data workers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crossformer.utils.mask_renderer import Intrinsics, rasterize_mesh

FLU2RDF = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)

DEFAULT_EXTR_DIR = Path.home() / "data/extr/cam"
DEFAULT_URDF = Path.home() / "crossformer/xarm7_standalone.urdf"
GRIPPER_CLOSED_RAD = 0.85  # URDF drive_joint range [0, 0.85]; 0.85 = closed (fingers together)


def K_for_size(h: int, w: int, f: float = 515.0) -> np.ndarray:
    return np.array(
        [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_w2c(cam: str, extr_dir: Path = DEFAULT_EXTR_DIR) -> np.ndarray:
    HT = np.load(extr_dir / cam / "HT.npz")["HT"].astype(np.float64)
    return np.linalg.inv(HT @ FLU2RDF).astype(np.float32)


def w2c_to_renderer(w2c: np.ndarray) -> np.ndarray:
    """OpenCV col-vec w2c → MaskRenderer row-vec OpenGL form."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return (flip @ w2c.astype(np.float64)).T


def render_robot_mask(
    joints_rad: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
    gripper_rad: float = GRIPPER_CLOSED_RAD,
) -> np.ndarray:
    """Render binary robot silhouette (h, w) uint8 in {0, 255}.

    Uses pyroki FK (same as fk_keypoints) for joint convention parity, then
    rasterizes via the in-tree CPU z-buffer.
    """
    from crossformer.utils.callbacks.synth_viz import _get_robot_mesh

    robot = _get_robot_mesh()
    q = np.zeros((1, robot.actuated), dtype=np.float32)
    q[0, :7] = joints_rad
    q[0, 7] = float(gripper_rad)

    # World-frame vertices via pyroki FK.
    verts_world = robot.posed_verts(q)[0]  # (V, 4) homogeneous

    # Transform to OpenGL camera frame (-Z forward).
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c_gl = flip @ w2c.astype(np.float64)
    verts_cam = (w2c_gl @ verts_world.astype(np.float64).T).T[:, :3]

    intr = Intrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=int(w),
        height=int(h),
    )
    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    inst_buf = np.zeros((h, w), dtype=np.uint8)
    rasterize_mesh(verts_cam, robot.faces, 1, intr, depth_buf, inst_buf)
    return (inst_buf > 0).astype(np.uint8) * 255
