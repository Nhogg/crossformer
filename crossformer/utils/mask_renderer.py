"""CPU z-buffer triangle rasterizer for robot silhouette masks.

Extracted from xclients/plugins/synth/mask_renderer.py so we can avoid the
sys.path hack in workers and own the dependency. Only the parts we actually
use are kept (Intrinsics + _rasterize_mesh); the MaskRenderer class that
relied on yourdfpy/trimesh for FK is dropped — we feed world-frame vertices
from pyroki FK directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def rasterize_mesh(
    verts_cam: np.ndarray,
    tris: np.ndarray,
    link_id: int,
    intr: Intrinsics,
    depth_buf: np.ndarray,
    inst_buf: np.ndarray,
) -> None:
    """Rasterize triangles into depth + instance buffers (in place).

    verts_cam: (V, 3) in camera frame, -Z forward (OpenGL).
    tris: (T, 3) int32 triangle indices.
    """
    W, H = intr.width, intr.height
    depth = -verts_cam[:, 2]
    valid = depth > 1e-6
    safe_depth = np.where(valid, depth, 1.0)
    x_px = intr.fx * (verts_cam[:, 0] / safe_depth) + intr.cx
    y_px = intr.cy - intr.fy * (verts_cam[:, 1] / safe_depth)
    v2 = np.stack([x_px, y_px], axis=1).astype(np.float32)

    tri_valid = valid[tris].all(axis=1)
    p0 = v2[tris[:, 0]]
    p1 = v2[tris[:, 1]]
    p2 = v2[tris[:, 2]]
    d0 = depth[tris[:, 0]]
    d1 = depth[tris[:, 1]]
    d2 = depth[tris[:, 2]]

    area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    tri_valid &= np.abs(area2) > 1e-6

    xmin = np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32)
    xmax = np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32)
    ymin = np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32)
    ymax = np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32)
    xmin = np.clip(xmin, 0, W)
    xmax = np.clip(xmax, 0, W)
    ymin = np.clip(ymin, 0, H)
    ymax = np.clip(ymax, 0, H)
    tri_valid &= (xmax > xmin) & (ymax > ymin)

    idx = np.nonzero(tri_valid)[0]
    for i in idx:
        _rasterize_one(
            p0[i],
            p1[i],
            p2[i],
            d0[i],
            d1[i],
            d2[i],
            area2[i],
            int(xmin[i]),
            int(xmax[i]),
            int(ymin[i]),
            int(ymax[i]),
            link_id,
            depth_buf,
            inst_buf,
        )


def _rasterize_one(
    a,
    b,
    c,
    da,
    db,
    dc,
    area2,
    xmin,
    xmax,
    ymin,
    ymax,
    link_id,
    depth_buf,
    inst_buf,
) -> None:
    xs = np.arange(xmin, xmax, dtype=np.float32) + 0.5
    ys = np.arange(ymin, ymax, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    w0 = (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])
    w1 = (c[0] - b[0]) * (Y - b[1]) - (c[1] - b[1]) * (X - b[0])
    w2 = (a[0] - c[0]) * (Y - c[1]) - (a[1] - c[1]) * (X - c[0])
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) if area2 > 0 else (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
    if not inside.any():
        return
    inv_area = 1.0 / area2
    l0 = ((b[0] - c[0]) * (Y - c[1]) - (b[1] - c[1]) * (X - c[0])) * inv_area
    l1 = ((c[0] - a[0]) * (Y - a[1]) - (c[1] - a[1]) * (X - a[0])) * inv_area
    l2 = 1.0 - l0 - l1
    inv_z = l0 / da + l1 / db + l2 / dc
    z = 1.0 / np.maximum(inv_z, 1e-9)
    existing = depth_buf[ymin:ymax, xmin:xmax]
    hit = inside & (z < existing)
    if not hit.any():
        return
    existing[hit] = z[hit]
    inst_buf[ymin:ymax, xmin:xmax][hit] = np.uint8(link_id)
