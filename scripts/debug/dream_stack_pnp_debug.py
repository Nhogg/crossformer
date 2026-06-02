from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crossformer.run.dream.session_calibration import (
    _valid_keypoints,
    calibrate_session_cameras,
    decode_camera_predictions_for_session,
    select_diverse_reliable_frames,
    SessionCalibrationConfig,
    solve_camera_extrinsics_from_frames,
)
from scripts.dream_inference import (
    build_session_arec,
    DreamInferConfig,
    make_predict_fn,
    open_arec_source,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arec-name", required=True)
    p.add_argument("--arec-version", required=True)
    p.add_argument("--arec-branch", default="main")
    p.add_argument("--arec-root", type=Path, default=Path("~/.cache/arrayrecords"))
    p.add_argument("--arec-chunk", type=int, default=1)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("~/dream_stack_debug"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--max-candidate-frames", type=int, default=512)
    p.add_argument("--max-selected-frames", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--focal-px", type=float, default=600.0)
    p.add_argument("--net-h", type=int, default=400)
    p.add_argument("--net-w", type=int, default=400)
    p.add_argument("--num-keypoints", type=int, default=10)
    p.add_argument("--encoder", default="vgg")
    p.add_argument("--variant", default="full")
    p.add_argument("--decoder", default="dpt")
    p.add_argument("--camera-keys", nargs="+", required=True)
    p.add_argument("--q-radians", action="store_true")
    return p.parse_args()


def frame_stats(frames, cfg: SessionCalibrationConfig) -> dict[str, Any]:
    counts, conf_mean, conf_max, in_bounds = [], [], [], []
    per_kp_valid = None
    per_kp_in_bounds = None
    for frame in frames:
        uv = np.asarray(frame.keypoints_px)
        conf = np.asarray(frame.keypoints_conf)
        valid = _valid_keypoints(frame, cfg)
        finite = np.isfinite(uv).all(axis=-1)
        h, w = frame.image.shape[:2] if frame.image is not None else (np.inf, np.inf)
        bounded = finite & (uv[:, 0] >= 0.0) & (uv[:, 0] < w) & (uv[:, 1] >= 0.0) & (uv[:, 1] < h)
        if per_kp_valid is None:
            per_kp_valid = np.zeros(len(valid), dtype=np.int64)
            per_kp_in_bounds = np.zeros(len(valid), dtype=np.int64)
        per_kp_valid[: len(valid)] += valid.astype(np.int64)
        per_kp_in_bounds[: len(bounded)] += bounded.astype(np.int64)
        counts.append(int(valid.sum()))
        conf_mean.append(float(np.nanmean(conf)))
        conf_max.append(float(np.nanmax(conf)))
        in_bounds.append(int(bounded.sum()))
    arr = np.asarray(counts)
    return {
        "frames": len(frames),
        "valid_count_min": int(arr.min()) if len(arr) else 0,
        "valid_count_p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "valid_count_p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "valid_count_max": int(arr.max()) if len(arr) else 0,
        "frames_ge4_valid": int(np.sum(arr >= 4)),
        "conf_mean_p50": float(np.percentile(conf_mean, 50)) if conf_mean else float("nan"),
        "conf_max_p50": float(np.percentile(conf_max, 50)) if conf_max else float("nan"),
        "in_bounds_p50": float(np.percentile(in_bounds, 50)) if in_bounds else 0.0,
        "per_keypoint_valid": [] if per_kp_valid is None else per_kp_valid.tolist(),
        "per_keypoint_in_bounds": [] if per_kp_in_bounds is None else per_kp_in_bounds.tolist(),
    }


def draw_keypoints(path: Path, frame, cfg: SessionCalibrationConfig) -> None:
    img = np.asarray(frame.image).copy()
    valid = _valid_keypoints(frame, cfg)
    for i, (uv, conf) in enumerate(zip(frame.keypoints_px, frame.keypoints_conf, strict=False)):
        if not np.isfinite(uv).all():
            continue
        x, y = np.round(uv).astype(int)
        color = (0, 255, 0) if i < len(valid) and valid[i] else (0, 0, 255)
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.putText(img, f"{i}:{conf:.2f}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def write_rasters(out_dir: Path, cam: str, attempt: str, res, frames) -> None:
    base = out_dir / attempt / cam
    base.mkdir(parents=True, exist_ok=True)
    by_idx = {f.frame_idx: f for f in frames}
    for idx, mask in res.rasterized_masks.items():
        mask_u8 = (np.asarray(mask) * 255).astype(np.uint8)
        cv2.imwrite(str(base / f"frame_{idx:06d}_rast.png"), mask_u8)
        frame = by_idx.get(idx)
        if frame is None or frame.image is None:
            continue
        image = np.asarray(frame.image).copy()
        if mask_u8.shape[:2] != image.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        overlay[mask_u8 > 0] = (0.35 * overlay[mask_u8 > 0] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
        cv2.imwrite(str(base / f"frame_{idx:06d}_rast_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    for idx, overlay in res.overlays.items():
        cv2.imwrite(str(base / f"frame_{idx:06d}_overlay.png"), cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR))


def summarize_result(res) -> dict[str, Any]:
    return {
        "success": bool(res.success),
        "failure_reason": res.failure_reason,
        "solver": res.solver,
        "subset_keypoint_indices": None if res.subset_keypoint_indices is None else list(res.subset_keypoint_indices),
        "num_selected_frames": int(res.num_selected_frames),
        "num_used_frames": int(res.num_used_frames),
        "num_candidate_points": int(res.num_candidate_points),
        "num_inlier_points": int(res.num_inlier_points),
        "mean_reproj_px": float(res.mean_reproj_px),
        "median_reproj_px": float(res.median_reproj_px),
        "mean_mask_iou": float(res.mean_mask_iou),
        "used_frame_indices": list(res.used_frame_indices[:20]),
        "rejected_frame_indices": list(res.rejected_frame_indices[:20]),
    }


def suppress_keypoints(frames, indices: tuple[int, ...]):
    out = []
    for frame in frames:
        conf = np.asarray(frame.keypoints_conf).copy()
        conf[list(indices)] = -np.inf
        out.append(replace(frame, keypoints_conf=conf))
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dream_cfg = DreamInferConfig(
        net_in_size=(args.net_h, args.net_w),
        num_keypoints=args.num_keypoints,
        encoder=args.encoder,
        variant=args.variant,
        decoder=args.decoder,
    )
    predict = make_predict_fn(dream_cfg, args.checkpoint, args.step)
    src = open_arec_source(
        args.arec_root.expanduser(), args.arec_name, args.arec_version, args.arec_branch, args.arec_chunk
    )
    session = build_session_arec(
        src,
        tuple(args.camera_keys),
        predict,
        args.batch_size,
        start=args.start,
        stop=args.stop,
        max_candidate_frames=args.max_candidate_frames,
        focal_px=args.focal_px,
        q_radians=args.q_radians,
    )

    base_cfg = SessionCalibrationConfig(
        enabled=True,
        camera_keys=tuple(args.camera_keys),
        max_candidate_frames_per_camera=args.max_candidate_frames,
        max_selected_frames_per_camera=args.max_selected_frames,
        q_degrees=not args.q_radians,
    )
    attempts = {
        "current": base_cfg,
        "current_min2": replace(base_cfg, min_frame_inliers=2),
        "conf_003": replace(base_cfg, keypoint_conf_threshold=0.03),
        "conf_003_min2": replace(base_cfg, keypoint_conf_threshold=0.03, min_frame_inliers=2),
        "conf_003_min1": replace(base_cfg, keypoint_conf_threshold=0.03, min_frame_inliers=1),
        "conf_003_no_subset": replace(
            base_cfg, keypoint_conf_threshold=0.03, subset_search=False, use_best_subset=False
        ),
        "conf_000_no_bounds": replace(
            base_cfg,
            keypoint_conf_threshold=0.0,
            require_keypoints_in_bounds=False,
            min_frame_inliers=2,
        ),
    }

    report: dict[str, Any] = {"cameras": {}, "attempts": {}}
    for cam in args.camera_keys:
        frames = decode_camera_predictions_for_session(session, cam, base_cfg)
        report["cameras"][cam] = {"stats": {}}
        for name, cfg in attempts.items():
            report["cameras"][cam]["stats"][name] = frame_stats(frames, cfg)

        top_cfg = attempts["conf_003_min2"]
        ranked = sorted(frames, key=lambda f: int(_valid_keypoints(f, top_cfg).sum()), reverse=True)[:12]
        for frame in ranked:
            draw_keypoints(out_dir / "keypoints" / cam / f"frame_{frame.frame_idx:06d}.png", frame, top_cfg)

    session_result = calibrate_session_cameras(session, base_cfg)
    report["inference_equivalent"] = {cam: summarize_result(res) for cam, res in session_result.camera_results.items()}

    for attempt, cfg in attempts.items():
        report["attempts"][attempt] = {}
        for cam in args.camera_keys:
            frames = decode_camera_predictions_for_session(session, cam, cfg)
            selected = select_diverse_reliable_frames(frames, cfg)
            res = solve_camera_extrinsics_from_frames(cam, selected, cfg, num_candidate_frames=len(frames))
            report["attempts"][attempt][cam] = summarize_result(res)
            write_rasters(out_dir, cam, attempt, res, frames)
            print(attempt, cam, report["attempts"][attempt][cam])

    arm_cfg = replace(
        base_cfg,
        keypoint_conf_threshold=0.03,
        min_frame_inliers=1,
        min_total_correspondences=4,
        min_total_inliers=4,
    )
    report["attempts"]["arm_only_drop_012"] = {}
    for cam in args.camera_keys:
        frames = suppress_keypoints(decode_camera_predictions_for_session(session, cam, arm_cfg), (0, 1, 2))
        selected = select_diverse_reliable_frames(frames, arm_cfg)
        res = solve_camera_extrinsics_from_frames(cam, selected, arm_cfg, num_candidate_frames=len(frames))
        report["attempts"]["arm_only_drop_012"][cam] = summarize_result(res)
        write_rasters(out_dir, cam, "arm_only_drop_012", res, frames)
        print("arm_only_drop_012", cam, report["attempts"]["arm_only_drop_012"][cam])

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
