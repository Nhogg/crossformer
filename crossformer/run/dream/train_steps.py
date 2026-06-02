from __future__ import annotations

import jax
import jax.image
import jax.numpy as jnp
import optax

from .modeling import _image_to_float


def resize_pred_heatmaps(pred_heatmaps: jax.Array, out_h: int, out_w: int) -> jax.Array:
    if tuple(pred_heatmaps.shape[-2:]) == (out_h, out_w):
        return pred_heatmaps
    pred_heatmaps = jnp.transpose(pred_heatmaps, (0, 2, 3, 1))
    pred_heatmaps = jax.image.resize(
        pred_heatmaps,
        (pred_heatmaps.shape[0], out_h, out_w, pred_heatmaps.shape[-1]),
        method="bilinear",
    )
    return jnp.transpose(pred_heatmaps, (0, 3, 1, 2))


def resize_pred_mask(pred_mask: jax.Array, out_h: int, out_w: int) -> jax.Array:
    if tuple(pred_mask.shape[-2:]) == (out_h, out_w):
        return pred_mask
    pred_mask = jnp.transpose(pred_mask, (0, 2, 3, 1))
    pred_mask = jax.image.resize(
        pred_mask,
        (pred_mask.shape[0], out_h, out_w, pred_mask.shape[-1]),
        method="bilinear",
    )
    return jnp.transpose(pred_mask, (0, 3, 1, 2))


def _stage_belief_maps(model_out):
    if isinstance(model_out, dict):
        return (model_out["heatmaps"],)
    if hasattr(model_out, "shape"):
        return (model_out,)
    if not isinstance(model_out, tuple | list):
        raise TypeError(f"unexpected DREAM output type: {type(model_out)}")
    if not model_out:
        raise ValueError("DREAM returned no outputs")

    first = model_out[0]
    if isinstance(first, dict):
        return tuple(stage["heatmaps"] for stage in model_out)
    if hasattr(first, "shape"):
        return tuple(model_out)
    if isinstance(first, tuple | list):
        return tuple(stage[0] for stage in model_out)
    raise TypeError(f"unexpected DREAM stage output type: {type(first)}")


def _stage_masks(model_out):
    if isinstance(model_out, dict):
        return (model_out["mask"],) if "mask" in model_out else ()
    if not isinstance(model_out, tuple | list) or not model_out:
        return ()

    first = model_out[0]
    if isinstance(first, dict):
        return tuple(stage["mask"] for stage in model_out if "mask" in stage)
    return ()


def prepare_pred_heatmaps(model_out, out_h: int, out_w: int):
    preds = tuple(
        resize_pred_heatmaps(jnp.transpose(stage, (0, 3, 1, 2)), out_h, out_w)
        for stage in _stage_belief_maps(model_out)
    )
    return preds[0] if len(preds) == 1 else preds


def prepare_pred_mask(model_out, out_h: int, out_w: int):
    masks = tuple(
        resize_pred_mask(jnp.transpose(stage, (0, 3, 1, 2)), out_h, out_w) for stage in _stage_masks(model_out)
    )
    return masks[-1] if masks else None


def final_pred_heatmaps(pred_heatmaps):
    return pred_heatmaps[-1] if isinstance(pred_heatmaps, tuple) else pred_heatmaps


def predict_heatmap_out(model, params, batch: dict, out_h: int, out_w: int) -> dict[str, jax.Array]:
    model_out, _ = model.apply({"params": params}, _image_to_float(batch["image"]))
    out = {"pred_heatmaps": final_pred_heatmaps(prepare_pred_heatmaps(model_out, out_h, out_w))}
    pred_mask = prepare_pred_mask(model_out, out_h, out_w)
    if pred_mask is not None:
        out["pred_mask"] = pred_mask
    return out


def make_train_step_dream(model, loss_fn, out_h: int, out_w: int, lr_fn, param_norm_fn):
    @jax.jit
    def train_step(state, batch):
        image = _image_to_float(batch["image"])

        def _loss(params):
            model_out, _ = model.apply({"params": params}, image)
            pred_heatmaps = prepare_pred_heatmaps(model_out, out_h, out_w)
            out_dict = {"pred_heatmaps": pred_heatmaps}
            pred_mask = prepare_pred_mask(model_out, out_h, out_w)
            if pred_mask is not None:
                out_dict["pred_mask"] = pred_mask
            return loss_fn(batch, out_dict)

        (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
        update_info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": param_norm_fn(state.params),
            "learning_rate": lr_fn(state.step),
            **metrics,
        }
        state = state.replace(
            step=state.step + 1,
            params=optax.apply_updates(state.params, updates),
            opt_state=opt_state,
        )
        return state, update_info

    return train_step


def make_eval_step_dream(model, loss_fn, out_h: int, out_w: int):
    @jax.jit
    def eval_step(state, batch):
        image = _image_to_float(batch["image"])
        model_out, _ = model.apply({"params": state.params}, image)
        pred_heatmaps = prepare_pred_heatmaps(model_out, out_h, out_w)
        out_dict = {"pred_heatmaps": pred_heatmaps}
        pred_mask = prepare_pred_mask(model_out, out_h, out_w)
        if pred_mask is not None:
            out_dict["pred_mask"] = pred_mask
        loss, metrics = loss_fn(batch, out_dict)
        return {"loss": loss, **metrics}

    return eval_step
