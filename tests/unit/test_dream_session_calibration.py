from __future__ import annotations

import cv2
import numpy as np

from crossformer.run.dream import session_calibration as sc
from crossformer.run.dream.session_calibration import (
    calibrate_session_cameras,
    FramePrediction,
    get_camera_keys,
    SessionCalibrationConfig,
    solve_camera_extrinsics_from_frames,
)
from crossformer.utils.spatial.calibration import project_points


def _w2c_from_rvec_t(rvec, t):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = np.asarray(t, dtype=np.float64)
    return w2c


def _fake_fk(joints):
    base = np.array(
        [
            [-0.2, -0.1, 0.0],
            [-0.1, 0.1, 0.05],
            [0.0, -0.08, 0.02],
            [0.08, 0.12, 0.1],
            [0.18, -0.04, 0.08],
            [0.02, 0.18, 0.13],
        ],
        dtype=np.float64,
    )
    shift = np.array([joints[0], joints[1] * 0.5, joints[2] * 0.25], dtype=np.float64)
    return base + shift


def test_solves_one_fixed_w2c_per_camera(monkeypatch):
    monkeypatch.setattr(sc, "fk_keypoints", _fake_fk)
    K = np.array([[620.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    w2c = _w2c_from_rvec_t([0.2, -0.1, 0.08], [0.1, -0.05, 1.2])
    qs = np.array(
        [
            [0.0, 0.0, 0.0, 0, 0, 0, 0],
            [0.08, 0.02, 0.03, 0, 0, 0, 0],
            [-0.04, 0.06, -0.02, 0, 0, 0, 0],
            [0.05, -0.07, 0.05, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    frames = []
    for i, q in enumerate(qs):
        pts = _fake_fk(q)
        frames.append(
            FramePrediction(
                cam_name="cam1",
                frame_idx=i,
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                q=q,
                K=K,
                keypoints_px=project_points(w2c, pts, K),
                keypoints_conf=np.ones(pts.shape[0], dtype=np.float64),
            )
        )

    cfg = SessionCalibrationConfig(
        enabled=True,
        min_selected_frames=2,
        min_total_correspondences=12,
        min_total_inliers=12,
        q_degrees=False,
        reproj_threshold_px=1.0,
        frame_reproj_threshold_px=1.0,
        keypoint_links=None,
    )
    res = solve_camera_extrinsics_from_frames("cam1", frames, cfg)

    assert res.success
    assert res.num_used_frames == 4
    assert res.num_inlier_points == 24
    assert res.mean_reproj_px < 1e-6
    np.testing.assert_allclose(res.w2c, w2c, atol=1e-6)


def test_session_calibration_keeps_cameras_independent(monkeypatch):
    monkeypatch.setattr(sc, "fk_keypoints", _fake_fk)
    K = np.array([[620.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cams = {
        "cam1": _w2c_from_rvec_t([0.2, -0.1, 0.08], [0.1, -0.05, 1.2]),
        "cam2": _w2c_from_rvec_t([-0.1, 0.15, -0.05], [-0.15, 0.03, 1.4]),
    }
    q = np.array(
        [
            [0.0, 0.0, 0.0, 0, 0, 0, 0],
            [0.08, 0.02, 0.03, 0, 0, 0, 0],
            [-0.04, 0.06, -0.02, 0, 0, 0, 0],
            [0.05, -0.07, 0.05, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    session = {
        "session_id": "synth",
        "q": q,
        "K": {cam: np.repeat(K[None], len(q), axis=0) for cam in cams},
        "image": {cam: np.zeros((len(q), 480, 640, 3), dtype=np.uint8) for cam in cams},
        "keypoints_conf": {cam: np.ones((len(q), 6), dtype=np.float64) for cam in cams},
        "keypoints_px": {
            cam: np.stack([project_points(w2c, _fake_fk(q_i), K) for q_i in q], axis=0) for cam, w2c in cams.items()
        },
    }
    cfg = SessionCalibrationConfig(
        enabled=True,
        min_selected_frames=2,
        min_total_correspondences=12,
        min_total_inliers=12,
        q_degrees=False,
        reproj_threshold_px=1.0,
        frame_reproj_threshold_px=1.0,
        keypoint_links=None,
    )

    result = calibrate_session_cameras(session, cfg)

    assert get_camera_keys(session, cfg) == ["cam1", "cam2"]
    assert set(result.camera_results) == {"cam1", "cam2"}
    for cam, w2c in cams.items():
        cam_res = result.camera_results[cam]
        assert cam_res.success
        np.testing.assert_allclose(cam_res.w2c, w2c, atol=1e-6)


def test_rejects_degenerate_keypoint_coverage(monkeypatch):
    monkeypatch.setattr(sc, "fk_keypoints", _fake_fk)
    K = np.array([[620.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    w2c = _w2c_from_rvec_t([0.2, -0.1, 0.08], [0.1, -0.05, 1.2])
    frames = []
    for i in range(4):
        q = np.array([0.04 * i, 0.02 * i, 0.01 * i, 0, 0, 0, 0], dtype=np.float64)
        pts = _fake_fk(q)
        conf = np.zeros(pts.shape[0], dtype=np.float64)
        conf[:2] = 1.0
        frames.append(
            FramePrediction(
                cam_name="cam1",
                frame_idx=i,
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                q=q,
                K=K,
                keypoints_px=project_points(w2c, pts, K),
                keypoints_conf=conf,
            )
        )

    cfg = SessionCalibrationConfig(
        enabled=True,
        min_selected_frames=2,
        min_frame_inliers=1,
        min_total_correspondences=4,
        min_total_inliers=4,
        q_degrees=False,
        keypoint_links=None,
    )
    res = solve_camera_extrinsics_from_frames("cam1", frames, cfg)

    assert not res.success
    assert res.failure_reason == "insufficient_keypoint_coverage"
