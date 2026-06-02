from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from crossformer.data.arec.arec import ArrayRecordBuilder, unpack_record
from crossformer.data.geometry import denormalize_kp2d
from crossformer.run.dream.metrics import extract_keypoints
from crossformer.run.dream.modeling import _image_to_float, make_model, net_out_size
from crossformer.run.dream.session_calibration import (
    calibrate_session_cameras,
    SessionCalibrationConfig,
)
from crossformer.run.dream.train_steps import (
    final_pred_heatmaps,
    prepare_pred_heatmaps,
    prepare_pred_mask,
)
from crossformer.utils.rig import K_for_size


@dataclass
class DreamInferConfig:
    seed: int = 0
    net_in_size: tuple[int, int] = (400, 400)
    image_c: int = 3
    num_keypoints: int = 10
    encoder: str = "vgg"
    variant: str = "full"
    decoder: str = "dpt"
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


class MultiArrayRecordSource:
    def __init__(self, img_src, pro_src, chunk: int = 1) -> None:
        if len(img_src) != len(pro_src):
            raise ValueError("image and proprio sources must be aligned")
        self.img_src = img_src
        self.pro_src = pro_src
        self.chunk = int(chunk)

    def __len__(self) -> int:
        return len(self.img_src) - self.chunk + 1

    def __getitem__(self, i: int) -> dict:
        img_rec = unpack_record(self.img_src[i])
        idxs = list(range(i, min(i + self.chunk, len(self.pro_src))))
        pro_recs = [unpack_record(x) for x in self.pro_src.__getitems__(idxs)]
        pro_rec = jax.tree.map(lambda *xs: np.stack(xs), *pro_recs)
        return {**img_rec, **pro_rec}


def load_params(path: Path, target_params, step: int | None):
    path = path.expanduser().resolve()
    if (path / "params").exists():
        path = path / "params"

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


def make_predict_fn(cfg: DreamInferConfig, ckpt: Path, step: int | None):
    out_h, out_w = net_out_size(cfg)
    model = make_model(cfg, cfg.num_keypoints)

    dummy = np.zeros((1, *cfg.net_in_size, cfg.image_c), dtype=np.uint8)
    params0 = model.init(jax.random.PRNGKey(cfg.seed), _image_to_float(dummy))["params"]
    params = load_params(ckpt, params0, step)

    @jax.jit
    def predict(images):
        model_out, _ = model.apply({"params": params}, _image_to_float(images))
        heatmaps = final_pred_heatmaps(prepare_pred_heatmaps(model_out, out_h, out_w))
        uv_hm, conf = extract_keypoints(heatmaps)

        h, w = images.shape[1], images.shape[2]
        uv_px = denormalize_kp2d(
            uv_hm / jnp.array([out_w, out_h], dtype=jnp.float32),
            h,
            w,
        )

        pred_mask = prepare_pred_mask(model_out, out_h, out_w)
        return uv_px, conf, pred_mask

    return predict


def run_batched(predict, images: np.ndarray, batch_size: int):
    uv_rows, conf_rows, mask_rows = [], [], []
    has_mask = None

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        uv, conf, mask = predict(batch)
        uv_rows.append(np.asarray(jax.device_get(uv)))
        conf_rows.append(np.asarray(jax.device_get(conf)))

        if has_mask is None:
            has_mask = mask is not None
        if mask is not None:
            mask_np = np.asarray(jax.device_get(mask))
            if mask_np.ndim == 4 and mask_np.shape[1] == 1:
                mask_np = mask_np[:, 0]
            mask_rows.append(mask_np)

    uv = np.concatenate(uv_rows, axis=0)
    conf = np.concatenate(conf_rows, axis=0)
    pred_mask = np.concatenate(mask_rows, axis=0) if has_mask else None
    return uv, conf, pred_mask


def build_session_npz(npz, camera_keys: tuple[str, ...], predict, batch_size: int):
    session = {
        "session_id": str(npz["session_id"]) if "session_id" in npz else None,
        "q": np.asarray(npz["q"]),
        "image": {},
        "K": {},
        "keypoints_px": {},
        "keypoints_conf": {},
        "pred_mask": {},
        "gt_mask": {},
    }

    for cam in camera_keys:
        image_key = f"image_{cam}"
        K_key = f"K_{cam}"
        mask_key = f"mask_{cam}"

        if image_key not in npz:
            raise KeyError(f"missing {image_key}")
        if K_key not in npz:
            raise KeyError(f"missing {K_key}")

        images = np.asarray(npz[image_key])
        K = np.asarray(npz[K_key])

        print(f"running DREAM: {cam} images={images.shape}")
        uv, conf, pred_mask = run_batched(predict, images, batch_size)

        session["image"][cam] = images
        session["K"][cam] = K
        session["keypoints_px"][cam] = uv
        session["keypoints_conf"][cam] = conf

        if pred_mask is not None:
            session["pred_mask"][cam] = pred_mask
        if mask_key in npz:
            session["gt_mask"][cam] = np.asarray(npz[mask_key])

    return session


def open_arec_source(root: Path, name: str, version: str, branch: str, chunk: int):
    builder = ArrayRecordBuilder(name=name, version=version, branch=branch, root=str(root))
    writers = builder.meta.get("writers", {})
    if writers:
        builder.writers = builder._normalize_writers(writers)
        builder.default_writer = "data" if "data" in builder.writers else next(iter(builder.writers))
    if {"image", "proprio"}.issubset(builder.writers):
        return MultiArrayRecordSource(builder.get_source("image"), builder.get_source("proprio"), chunk=chunk)
    return builder.source


def read_record(src, idx: int) -> dict:
    x = src[idx]
    return unpack_record(x) if isinstance(x, bytes) else x


def candidate_indices(n: int, start: int, stop: int | None, max_frames: int) -> np.ndarray:
    stop = n if stop is None else min(stop, n)
    if stop <= start:
        raise ValueError(f"empty ArrayRecord range: {start=} {stop=} {n=}")
    n_frames = min(max_frames, stop - start)
    return np.linspace(start, stop - 1, n_frames, dtype=np.int64)


def cam_value(sample: dict, key: str, cam: str, default=None):
    bracket = f"{key}[{cam}]"
    if bracket in sample:
        return sample[bracket]
    underscored = f"{key}_{cam}"
    if underscored in sample:
        return sample[underscored]
    if key not in sample:
        return default
    val = sample[key]
    if isinstance(val, dict):
        return val.get(cam, default)
    return val


def sample_camera_index(sample: dict, cam: str, n: int) -> int:
    info = sample.get("info", {})
    keys = np.asarray(info.get("image_keys", []), dtype=str).reshape(-1)
    matches = [i for i, key in enumerate(keys[:n]) if key == cam or f".{cam}." in key or key.startswith(f"{cam}.")]
    if len(matches) == 1:
        return int(matches[0])
    if len(matches) > 1:
        raise KeyError(f"ambiguous camera key {cam!r}; matches {[keys[i] for i in matches]}")
    if n == 1:
        return 0
    raise KeyError(f"camera {cam!r} not found in image_keys={keys.tolist()}")


def sample_image(sample: dict, cam: str) -> np.ndarray:
    img = cam_value(sample, "image", cam)
    if img is None:
        raise KeyError(f"missing image for camera {cam}")
    img = np.asarray(img)
    return img[sample_camera_index(sample, cam, img.shape[0])] if img.ndim == 4 else img


def sample_mask(sample: dict, cam: str) -> np.ndarray | None:
    mask = cam_value(sample, "mask", cam)
    if mask is None:
        mask = cam_value(sample, "gt_mask", cam)
    if mask is None:
        return None
    mask = np.asarray(mask)
    return mask[sample_camera_index(sample, cam, mask.shape[0])] if mask.ndim == 3 else mask


def sample_q(sample: dict, q_radians: bool) -> np.ndarray:
    if "q" in sample:
        return np.asarray(sample["q"], dtype=np.float32).reshape(-1)[:8]
    if "joint_positions" in sample:
        return np.asarray(sample["joint_positions"], dtype=np.float32).reshape(-1)[:8]
    if "proprio" not in sample or "joints" not in sample["proprio"]:
        raise KeyError("record is missing q, joint_positions, or proprio['joints']")
    joints = np.asarray(sample["proprio"]["joints"], dtype=np.float32).reshape(-1, 7)[0]
    if not q_radians:
        joints = np.rad2deg(joints)
    grip = np.asarray(sample["proprio"].get("gripper", [0.0]), dtype=np.float32).reshape(-1)[:1]
    return np.concatenate([joints, grip], axis=0).astype(np.float32)


def sample_K(sample: dict, cam: str, image: np.ndarray, focal_px: float) -> np.ndarray:
    K = cam_value(sample, "K", cam)
    if K is None:
        K = cam_value(sample, "intrinsics", cam)
    if isinstance(K, dict):
        K = K.get("K")
    if K is not None:
        K = np.asarray(K, dtype=np.float32)
        if K.ndim == 3:
            K = K[sample_camera_index(sample, cam, K.shape[0])]
        return K
    h, w = image.shape[:2]
    return K_for_size(h, w, f=focal_px).astype(np.float32)


def build_session_arec(
    src,
    camera_keys: tuple[str, ...],
    predict,
    batch_size: int,
    *,
    start: int,
    stop: int | None,
    max_candidate_frames: int,
    focal_px: float,
    q_radians: bool,
):
    idxs = candidate_indices(len(src), start, stop, max_candidate_frames)
    samples = [read_record(src, int(i)) for i in idxs]
    session = {
        "session_id": f"arec:{start}:{idxs[-1]}",
        "q": np.stack([sample_q(s, q_radians) for s in samples], axis=0),
        "image": {},
        "K": {},
        "keypoints_px": {},
        "keypoints_conf": {},
        "pred_mask": {},
        "gt_mask": {},
    }

    for cam in camera_keys:
        images = np.stack([sample_image(s, cam) for s in samples], axis=0)
        K = np.stack([sample_K(s, cam, img, focal_px) for s, img in zip(samples, images, strict=True)], axis=0)
        masks = [sample_mask(s, cam) for s in samples]

        print(f"running DREAM: {cam} records={len(idxs)} images={images.shape}")
        uv, conf, pred_mask = run_batched(predict, images, batch_size)

        session["image"][cam] = images
        session["K"][cam] = K
        session["keypoints_px"][cam] = uv
        session["keypoints_conf"][cam] = conf
        if pred_mask is not None:
            session["pred_mask"][cam] = pred_mask
        if all(m is not None for m in masks):
            session["gt_mask"][cam] = np.stack(masks, axis=0)

    return session


def save_result(path: Path, result):
    out = {}
    for cam, res in result.camera_results.items():
        out[f"{cam}_success"] = np.asarray(res.success)
        out[f"{cam}_failure_reason"] = np.asarray("" if res.failure_reason is None else res.failure_reason)
        out[f"{cam}_w2c"] = res.w2c if res.w2c is not None else np.full((4, 4), np.nan)
        out[f"{cam}_K"] = res.K if res.K is not None else np.full((3, 3), np.nan)
        out[f"{cam}_mean_reproj_px"] = np.asarray(res.mean_reproj_px)
        out[f"{cam}_median_reproj_px"] = np.asarray(res.median_reproj_px)
        out[f"{cam}_num_inlier_points"] = np.asarray(res.num_inlier_points)
        out[f"{cam}_used_frame_indices"] = np.asarray(res.used_frame_indices, dtype=np.int64)
        out[f"{cam}_rejected_frame_indices"] = np.asarray(res.rejected_frame_indices, dtype=np.int64)
        out[f"{cam}_solver"] = np.asarray("" if res.solver is None else res.solver)
        out[f"{cam}_subset_keypoint_indices"] = np.asarray(
            [] if res.subset_keypoint_indices is None else res.subset_keypoint_indices,
            dtype=np.int64,
        )

    for k, v in result.summary.items():
        out[k.replace("/", "__")] = np.asarray(v)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--session-npz", type=Path)
    src_group.add_argument("--arec-name", type=str)
    p.add_argument("--arec-root", type=Path, default=Path("~/.cache/arecs"))
    p.add_argument("--arec-version", type=str, default="0.0.1")
    p.add_argument("--arec-branch", type=str, default="main")
    p.add_argument("--arec-chunk", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--max-candidate-frames", type=int, default=512)
    p.add_argument("--focal-px", type=float, default=515.0)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera-keys", nargs="+", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)

    p.add_argument("--net-h", type=int, default=400)
    p.add_argument("--net-w", type=int, default=400)
    p.add_argument("--num-keypoints", type=int, default=10)
    p.add_argument("--encoder", type=str, default="vgg")
    p.add_argument("--variant", type=str, default="full")
    p.add_argument("--decoder", type=str, default="dpt")

    p.add_argument("--q-radians", action="store_true")
    p.add_argument("--max-selected-frames", type=int, default=64)
    args = p.parse_args()

    dream_cfg = DreamInferConfig(
        net_in_size=(args.net_h, args.net_w),
        num_keypoints=args.num_keypoints,
        encoder=args.encoder,
        variant=args.variant,
        decoder=args.decoder,
    )
    predict = make_predict_fn(dream_cfg, args.checkpoint, args.step)

    if args.session_npz is not None:
        with np.load(args.session_npz, allow_pickle=True) as npz:
            session = build_session_npz(npz, tuple(args.camera_keys), predict, args.batch_size)
    else:
        src = open_arec_source(args.arec_root, args.arec_name, args.arec_version, args.arec_branch, args.arec_chunk)
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

    calib_cfg = SessionCalibrationConfig(
        enabled=True,
        camera_keys=tuple(args.camera_keys),
        max_candidate_frames_per_camera=args.max_candidate_frames,
        max_selected_frames_per_camera=args.max_selected_frames,
        q_degrees=not args.q_radians,
    )
    result = calibrate_session_cameras(session, calib_cfg)

    print(result.summary)
    for cam, res in result.camera_results.items():
        print(cam, res.success, res.failure_reason)
        print(res.w2c)

    save_result(args.out, result)


if __name__ == "__main__":
    main()
