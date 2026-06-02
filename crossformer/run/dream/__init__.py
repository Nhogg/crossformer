from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "_checkpoint_state": ".config",
    "_save_path": ".config",
    "ADD_THRESHOLDS_MM": ".constants",
    "Config": ".config",
    "DreamCheckpointModel": ".config",
    "DreamCheckpointState": ".config",
    "DreamVizConfig": ".config",
    "KP_CONF_THRESHOLD": ".constants",
    "KP_MISSING_VALUE": ".constants",
    "KP_PEAK_AMBIGUITY_GAP": ".constants",
    "KP_PEAK_THRESHOLD": ".constants",
    "KP_SMOOTH_RADIUS": ".constants",
    "KP_SMOOTH_SIGMA": ".constants",
    "Optim": ".config",
    "SOURCE_REAL": ".constants",
    "SOURCE_SYNTH": ".constants",
    "belief_sigma": ".losses",
    "build_heatmaps": ".losses",
    "dream_loss_fn": ".losses",
    "focal_heatmap_loss": ".losses",
    "mask_loss": ".losses",
    "mask_target": ".losses",
    "_mask_iou": ".metrics",
    "_pnp_reproj_err": ".metrics",
    "_solve_pose_one": ".metrics",
    "extract_keypoints": ".metrics",
    "keypoint_metrics": ".metrics",
    "PNP_MASK_IOU_THRESH": ".metrics",
    "PNP_MIN_VALID_KP": ".metrics",
    "PNP_REPROJ_THRESH": ".metrics",
    "pose_metrics": ".metrics",
    "pose_metrics_irl": ".metrics",
    "_count_params": ".modeling",
    "_count_trainable_params": ".modeling",
    "_image_to_float": ".modeling",
    "frozen_keys": ".modeling",
    "load_tips_params": ".modeling",
    "make_model": ".modeling",
    "net_out_size": ".modeling",
    "build_multiframe_correspondences": ".session_calibration",
    "calibrate_session_cameras": ".session_calibration",
    "CameraCalibrationResult": ".session_calibration",
    "decode_camera_predictions_for_session": ".session_calibration",
    "FramePrediction": ".session_calibration",
    "get_camera_keys": ".session_calibration",
    "MultiFrameCorrespondences": ".session_calibration",
    "MultiFramePnPConfig": ".session_calibration",
    "RobustStackedPnPResult": ".session_calibration",
    "score_camera_calibration": ".session_calibration",
    "select_diverse_reliable_frames": ".session_calibration",
    "SessionCalibrationConfig": ".session_calibration",
    "SessionCalibrationResult": ".session_calibration",
    "solve_camera_extrinsics_from_frames": ".session_calibration",
    "solve_robust_stacked_pnp": ".session_calibration",
    "summarize_session_calibration": ".session_calibration",
    "final_pred_heatmaps": ".train_steps",
    "make_eval_step_dream": ".train_steps",
    "make_train_step_dream": ".train_steps",
    "predict_heatmap_out": ".train_steps",
    "prepare_pred_heatmaps": ".train_steps",
    "prepare_pred_mask": ".train_steps",
    "resize_pred_heatmaps": ".train_steps",
    "resize_pred_mask": ".train_steps",
    "main": ".main",
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
