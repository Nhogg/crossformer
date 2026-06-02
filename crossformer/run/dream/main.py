from __future__ import annotations

from flax.training.train_state import TrainState
import jax
from rich import print
from rich.pretty import pprint
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm

from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import Timer
import wandb

from .config import _checkpoint_state, _save_path, Config
from .data import make_dataset, make_irl_dataset
from .losses import belief_sigma, dream_loss_fn
from .metrics import pose_metrics, pose_metrics_irl
from .modeling import (
    _count_params,
    _count_trainable_params,
    _image_to_float,
    frozen_keys,
    load_tips_params,
    make_model,
    net_out_size,
)
from .train_steps import (
    final_pred_heatmaps,
    make_eval_step_dream,
    make_train_step_dream,
    predict_heatmap_out,
    prepare_pred_heatmaps,
    prepare_pred_mask,
)
from .viz import maybe_log_viz


def _print_shapes(shapes):
    table = Table("stage", "shape")
    for name, shape in shapes:
        table.add_row(name, str(shape))
    print(table)


def main(cfg: Config):  # noqa: C901
    timer = Timer()
    ndev = len(jax.devices())
    if cfg.bs % ndev != 0:
        raise ValueError(f"bs={cfg.bs} must be divisible by device_count={ndev}")
    ds = make_dataset(cfg)
    dsit = iter(ds)
    irl_dsit = iter(make_irl_dataset(cfg)) if cfg.wandb.use and cfg.viz.every > 0 else None
    batch = next(dsit)

    print(Rule("DREAM Prepared Sample", style="bold magenta"))
    pprint(spec(batch))
    run = cfg.wandb.initialize(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    init_rng = rng
    num_keypoints = cfg.num_keypoints or int(batch["keypoints_2d_norm"].shape[1])
    print(f"raw_size={cfg.raw_size} net_in_size={cfg.net_in_size} net_out_size={net_out_size(cfg)}")
    out_h, out_w = net_out_size(cfg)
    print(f"target_sigma={belief_sigma(cfg.sigma_pct, out_h, out_w):.3f} output px")
    model = make_model(cfg, num_keypoints)
    image = _image_to_float(batch["image"])
    params = model.init(init_rng, image)["params"]
    params = load_tips_params(cfg, params)
    frozen = frozen_keys(cfg)
    tx, lr_fn, param_norm_fn = cfg.optim.create(params, steps=cfg.steps, frozen_keys=frozen)
    print(Rule("optimizer", style="bold magenta"))
    print(f"  config: {cfg.optim.kwargs(cfg.steps, frozen_keys=frozen)}")
    print(f"  tx: {tx}")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    if cfg.save_dir is not None:
        save_dir = _save_path(cfg)
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir — checkpoints disabled[/]")

    loss_fn = lambda batch, out_dict: dream_loss_fn(
        batch, out_dict, sigma_pct=cfg.sigma_pct, mask_weight=cfg.mask_weight
    )
    train_step = make_train_step_dream(
        model, loss_fn, out_h=out_h, out_w=out_w, lr_fn=lr_fn, param_norm_fn=param_norm_fn
    )
    eval_step = make_eval_step_dream(model, loss_fn, out_h=out_h, out_w=out_w)

    model_out, shapes = model.apply({"params": state.params}, image)
    pred_heatmaps = prepare_pred_heatmaps(model_out, out_h, out_w)
    out_dict = {"pred_heatmaps": pred_heatmaps}
    pred_mask = prepare_pred_mask(model_out, out_h, out_w)
    if pred_mask is not None:
        out_dict["pred_mask"] = pred_mask
    _, init_metrics = loss_fn(batch, out_dict)

    print(Rule("DREAM Forward", style="bold magenta"))
    _print_shapes(shapes)
    print(f"params={_count_params(state.params):,}")
    print(f"trainable_params={_count_trainable_params(state.params, frozen):,}")
    final_pred = final_pred_heatmaps(out_dict["pred_heatmaps"])
    print(f"pred_heatmaps.shape={final_pred.shape}")
    if "pred_mask" in out_dict:
        print(f"pred_mask.shape={out_dict['pred_mask'].shape}")
    if tuple(final_pred.shape[-2:]) != (out_h, out_w):
        raise ValueError(
            f"expected net_out_size={(out_h, out_w)} from variant={cfg.variant}, got {tuple(final_pred.shape[-2:])}"
        )
    print(f"init_loss={float(init_metrics['loss']):.6f}")
    maybe_log_viz(cfg, batch, out_dict, step=0)
    if irl_dsit is not None:
        irl_batch = next(irl_dsit)
        irl_out = predict_heatmap_out(model, state.params, irl_batch, out_h, out_w)
        maybe_log_viz(cfg, irl_batch, irl_out, step=0, prefix="irl")

    print(Rule("DREAM Train Loop", style="bold magenta"))
    for step in tqdm(range(cfg.steps)):
        with timer("data"):
            batch = next(dsit)
        with timer("train_step"):
            state, metrics = train_step(state, batch)

        if step % cfg.log_every == 0:
            with timer("data"):
                eval_batch = next(dsit)
            with timer("eval_step"):
                eval_metrics = eval_step(state, eval_batch)
            with timer("pose"):
                model_out, _ = model.apply({"params": state.params}, _image_to_float(eval_batch["image"]))
                eval_out = {"pred_heatmaps": final_pred_heatmaps(prepare_pred_heatmaps(model_out, out_h, out_w))}
                pnp_metrics = pose_metrics(cfg, eval_batch, eval_out)
            times = {f"timer/{k}": v for k, v in timer.get_average_times().items()}
            cfg.wandb.log({"train": metrics, "eval": eval_metrics, "pose": pnp_metrics, **times}, step=step)
            print({**metrics, **eval_metrics, **pnp_metrics, **times})
        if cfg.viz.every > 0 and step % cfg.viz.every == 0:
            out_dict = predict_heatmap_out(model, state.params, batch, out_h, out_w)
            maybe_log_viz(cfg, batch, out_dict, step=step)
            if irl_dsit is not None:
                irl_batch = next(irl_dsit)
                irl_out = predict_heatmap_out(model, state.params, irl_batch, out_h, out_w)
                maybe_log_viz(cfg, irl_batch, irl_out, step=step, prefix="irl")
                with timer("irl_pose"):
                    irl_pose = pose_metrics_irl(cfg, irl_batch, irl_out)
                cfg.wandb.log({"irl_pose": irl_pose}, step=step)
        if cfg.save_interval > 0 and (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback.save(_checkpoint_state(state), step + 1)

    if save_dir is not None:
        save_callback.save(_checkpoint_state(state), cfg.steps)
        save_callback.wait()
    if cfg.verbose:
        print(model.tabulate(init_rng, _image_to_float(batch["image"]), depth=2))
    cfg.wandb.finish()
