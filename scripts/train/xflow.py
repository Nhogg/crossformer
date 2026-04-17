"""End-to-end XFlowHead training: CrossFormerModel + real data.

Bundled action format: uses act.base / act.id from the grain embody
pipeline instead of per-head action extraction.

Usage:
    uv run scripts/train/xflow.py
    uv run scripts/train/xflow.py --steps 500 --lr 3e-4
    uv run scripts/train/xflow.py --mix xgym_sweep --batch-size 4
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from rich import print
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset import DataSourceE
from crossformer.cn.dataset.dataset import Loader
from crossformer.cn.model_factory import Vision
from crossformer.data.grain.embody import decode_embody_name
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.train_step import lookup_guide, make_train_step
from crossformer.run.xflow_eval import extract_bundled_actions, flatten_obs, XFlowEvalCallbacks, XFlowEvalLoop
from crossformer.utils.callbacks.rast import RastConfig
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.callbacks.val_mse import ValMSEConfig
from crossformer.utils.callbacks.viz import (
    ChunkVizCallback,
    HistVizCallback,
    VizConfig,
)
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import create_optimizer, Timer, TrainState
import wandb

# -- config -------------------------------------------------------------------


@dataclass
class Config:
    """XFlowHead end-to-end training config."""

    name: str = ""
    steps: int = 1_000_000  # training steps (1 for debug)
    lr: float = 1e-3  # learning rate
    log_every: int = 100  # log interval
    batch_size: int = 256  # global batch size
    mix: str = "xgym_sweep"  # dataset mix name
    horizon: int = 20  # action horizon from data pipeline
    verbose: bool = False  # print model tabulation during init
    model: cn.ModelFactory = default(
        cn.ModelFactory(
            size=cn.Size.DUMMY,
            window=20,
            image_keys=(),
            proprio_keys=(),
            vision=Vision(use_film=False, encoder="resnetv2-50-film"),
        )
    )

    # Optimizer
    weight_decay: float = 1e-4  # adamw weight decay
    warmup_steps: int = 0  # lr warmup steps (0 = no warmup)
    lr_schedule: str = "constant"  # constant | cosine | rsqrt
    clip_gradient: float | None = 1.0  # global gradient clipping (None to disable)
    frozen_keys: tuple[str, ...] = ()  # fnmatch patterns for frozen params

    # Token guidance
    use_guidance: bool = False  # enable guidance tokens
    guidance_drop_prob: float = 0.5  # prob of dropping guidance each step
    compress_guidance: bool = False  # compress via perceiver latents
    num_guidance_latents: int = 4  # latent count when compress=True
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")  # dot-paths into batch for guidance signal

    # Checkpointing
    save_dir: str | None = Path().home().expanduser()  # checkpoint root dir (None to disable)
    save_interval: int = 25_000  # save every N steps

    train_loader: Loader = default(Loader(use_grain=True))
    mp: int = 8  # grain multiproc (for data loading)
    patch_prob: float = 0.0  # prob of occluding each (sample, time, view)
    patch_min_frac: float = 0.05
    patch_max_frac: float = 0.5
    eval_frames: int = 64  # eval examples to poll for rast videos
    hist_every: int = 0  # histogram log interval
    quit_after_model: bool = False  # stop after model creation for debugging
    viz: VizConfig = default(VizConfig())
    val_mse: ValMSEConfig = default(ValMSEConfig())
    rast: RastConfig = default(RastConfig())

    wandb: cn.Wandb = default(cn.Wandb())


# -- helpers ------------------------------------------------------------------


def infer_model_keys(obs):
    """Infer image and proprio tokenizer keys from a real observation batch."""
    image_keys = tuple(k.removeprefix("image_") for k in sorted(obs) if k.startswith("image_"))
    proprio_keys = tuple(k.removeprefix("proprio_") for k in sorted(obs) if k.startswith("proprio_"))
    return image_keys, proprio_keys


def per_embodiment_metrics(batch, update_info):
    """Compute per-embodiment loss from sample_mse and act.embody.

    Returns dict like {"embodiment/single": mse, "embodiment/dual_arm": mse, ...}.
    """
    embody_arr = np.array(batch["act"]["embody"])  # (B, 32) uint8
    sample_mse = np.array(update_info["sample_mse"])  # (B,)
    names = [decode_embody_name(embody_arr[i]) for i in range(embody_arr.shape[0])]
    groups: dict[str, list[float]] = {}
    for name, mse in zip(names, sample_mse):
        groups.setdefault(name, []).append(float(mse))
    return {f"embodiment/{k}": sum(v) / len(v) for k, v in groups.items()}


def shard_batch(batch, mesh):
    """Shard a host-local batch across the data axis."""
    return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))


def make_data_cfg(mix: str, batch_size: int, loader: Loader) -> cn.Train:
    """Build a Train config for a specific loader."""
    return cn.Train(
        data=cn.Dataset(
            mix=DataSourceE[mix],
            loader=replace(loader, global_batch_size=batch_size),
        ),
        seed=42,
        verbosity=0,
    )


def _save_path(cfg: Config) -> str:
    if cfg.save_dir is None:
        raise ValueError("save_dir is None")
    return str((Path(cfg.save_dir).expanduser() / cfg.wandb.project / (cfg.wandb.group or "") / cfg.name).resolve())


# -- main ---------------------------------------------------------------------


def main(cfg: Config):
    initialize_compilation_cache()
    devices = jax.devices()
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    print(Rule("XFlowHead + CrossFormerModel: bundled actions", style="bold magenta"))
    print(f"  backend={jax.default_backend()} devices={len(devices)}")
    if cfg.batch_size % len(devices) != 0:
        raise ValueError(f"batch_size={cfg.batch_size} must be divisible by devices={len(devices)}")

    max_h = cfg.horizon
    run = cfg.wandb.initialize(cfg)

    # Load data
    print(Rule("loading data"))
    from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory

    _apply_fd_limit(512**2)
    train_cfg = make_data_cfg(cfg.mix, cfg.batch_size, cfg.train_loader)
    eval_cfg = make_data_cfg(
        cfg.mix,
        cfg.batch_size,
        Loader(
            use_grain=True,
            shuffle_buffer=1,
            threads_traj_transform=16,
            threads_traj_read=16,
            threads_frame_transform=16,
            prefetch=8,
        ),
    )
    dataset = GrainDataFactory(
        mp=cfg.mp, patch_prob=cfg.patch_prob, patch_min_frac=cfg.patch_min_frac, patch_max_frac=cfg.patch_max_frac
    ).make(train_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=True)
    dsit = iter(dataset.dataset)
    example_batch = next(dsit)
    eval_dataset = GrainDataFactory(
        mp=0,
        shuffle=False,
        mask_slot=False,
        shuffle_slot=False,
        imaug=False,
        patch_prob=cfg.patch_prob,
        patch_min_frac=cfg.patch_min_frac,
        patch_max_frac=cfg.patch_max_frac,
    ).make(eval_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=False)
    print(spec(example_batch))
    inferred_image_keys, inferred_proprio_keys = infer_model_keys(example_batch["observation"])
    obs_keys: tuple[str, ...] = ()
    print(f"  image_keys: {inferred_image_keys}")
    print(f"  proprio_keys (available): {inferred_proprio_keys}")

    guide_example = None
    if cfg.use_guidance:
        guide_example = lookup_guide(example_batch, cfg.guide_keys)

    # Get max_a from the bundled action shape
    max_a = example_batch["act"]["id"].shape[-1]
    print(f"  max_h={max_h}  max_a={max_a}")
    print(f"  act.base shape: {example_batch['act']['base'].shape}")
    print(f"  act.id   shape: {example_batch['act']['id'].shape}")

    # Build model
    print(Rule("building CrossFormerModel"))
    max_w = example_batch["observation"]["timestep_pad_mask"].shape[1]
    cfg.model.window = max_w
    cfg.model.image_keys = inferred_image_keys
    cfg.model.proprio_keys = ()
    cfg.model.xflow.max_dofs = max_a
    cfg.model.xflow.max_horizon = max_h
    cfg.model.xflow.use_guidance = cfg.use_guidance
    cfg.model.xflow.guidance_input_dim = None if guide_example is None else guide_example.shape[-1]
    example_obs = flatten_obs(example_batch["observation"], obs_keys)

    init_obs = dict(example_obs)
    init_obs |= {
        k: v
        for k, v in example_batch["observation"].items()
        if any(k == f"image_{name}" for name in cfg.model.image_keys)
    }
    print(f"  image keys in init_obs: {[k for k in init_obs if 'image' in k or 'depth' in k]}")
    init_batch = {
        "observation": init_obs,
        "task": example_batch.get("task", {"pad_mask_dict": {}}),
    }
    wandb.config.update(
        {
            "example_batch_spec": spec(example_batch),
            "obs_keys": obs_keys,
            "max_a": max_a,
            **cfg.model.create(),
        },
        allow_val_change=True,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, train_rng, pred_rng = jax.random.split(rng, 3)

    model = CrossFormerModel.from_config(
        cfg.model.create(),
        init_batch,
        text_processor=None,
        verbose=cfg.verbose,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    if cfg.quit_after_model:
        print("quit_after_model=True; stopping after model creation")
        return
    model = model.replace(
        params=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.params),
        example_batch=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.example_batch),
    )
    n_params = sum(x.size for x in jax.tree.leaves(model.params))
    print(f"  params: {n_params:,}")
    print(f"  heads: {list(model.module.heads.keys())}")
    if cfg.frozen_keys:
        print(f"  frozen_keys: {list(cfg.frozen_keys)}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # Guidance config sanity check
    if cfg.use_guidance:
        print(Rule("guidance encoder"))
        print(f"  guide_keys={cfg.guide_keys} shape={guide_example.shape}")

    # Optimizer + state
    lr_kwargs: dict = {
        "learning_rate": (
            {
                "name": cfg.lr_schedule,
                "init_value": 0.0,
                "peak_value": cfg.lr,
                "warmup_steps": cfg.warmup_steps,
                **({"decay_steps": cfg.steps} if cfg.lr_schedule == "cosine" else {}),
            }
            if cfg.warmup_steps > 0
            else cfg.lr
        ),
        "weight_decay": cfg.weight_decay,
    }
    if cfg.clip_gradient is not None:
        lr_kwargs["clip_gradient"] = cfg.clip_gradient
    if cfg.frozen_keys:
        lr_kwargs["frozen_keys"] = list(cfg.frozen_keys)
    tx, lr_callable, param_norm_callable = create_optimizer(model.params, **lr_kwargs)
    state = TrainState.create(model=model, tx=tx, rng=train_rng)
    train_step = make_train_step(model.module, lr_callable, param_norm_callable)

    # Checkpointing
    if cfg.save_dir is not None:
        save_dir = _save_path(cfg)
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir — checkpoints disabled[/]")

    hist_cb = HistVizCallback(stats=dataset.dataset_statistics)
    chunk_cb = ChunkVizCallback(stats=dataset.dataset_statistics)
    viz_cb = cfg.viz.create()
    rast_cb = cfg.rast.create()
    val_mse_cb = cfg.val_mse.create(stats=dataset.dataset_statistics, guide_keys=cfg.guide_keys)
    eval_loop = XFlowEvalLoop(
        loader=eval_dataset.dataset,
        obs_keys=obs_keys,
        pred_rng=pred_rng,
        callbacks=XFlowEvalCallbacks(
            hist_cb=hist_cb,
            chunk_cb=chunk_cb,
            viz_cb=viz_cb,
            rast_cb=rast_cb,
            val_mse_cb=val_mse_cb,
            wandb_log=cfg.wandb.log,
            hist_every=cfg.hist_every,
            viz_every=cfg.viz.every,
            val_every=cfg.val_mse.every,
            eval_frames=cfg.eval_frames,
            use_guidance=cfg.use_guidance,
            guide_keys=cfg.guide_keys,
        ),
    )

    # Train
    print(Rule("training"))
    table = Table(title="training")
    table.add_column("step", justify="right", style="cyan")
    table.add_column("loss", justify="right")
    table.add_column("|grad|", justify="right", style="dim")

    losses = []
    timer = Timer()
    lowdim_rng = np.random.default_rng(42)
    guide_rng = np.random.default_rng(43)
    for step in tqdm(range(cfg.steps)):
        timer.tick("total")
        with timer("dataset"):
            batch = next(dsit)
            obs = flatten_obs(batch["observation"], obs_keys)
            task = batch.get("task", {"pad_mask_dict": {}})
            pad_mask = obs["timestep_pad_mask"]
            lowdim_active = True
            # if cfg.lowdim_drop_prob > 0.0 and lowdim_rng.random() < cfg.lowdim_drop_prob:
            # raise NotImplementedError("lowdim_drop_prob > 0 is not implemented yet")
            # obs = zero_lowdim_obs(obs, obs_keys)
            # lowdim_active = False

            guide_input = None
            if cfg.use_guidance:
                guide_input = lookup_guide(batch, cfg.guide_keys)
                if cfg.guidance_drop_prob > 0.0 and guide_rng.random() < cfg.guidance_drop_prob:
                    guide_input = None

            actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h)
        with timer("train"):
            state, update_info = train_step(
                state,
                obs,
                task,
                pad_mask,
                actions,
                dof_ids,
                chunk_steps,
                guide_input=guide_input,
            )
        timer.tock("total")
        print(jax.tree.map(lambda x: x.shape, batch))
        update_info = jax.device_get(update_info)
        total_loss = float(update_info["loss"])
        losses.append(total_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            embody_metrics = {}
            if "embody" in batch.get("act", {}):
                embody_metrics = per_embodiment_metrics(batch, update_info)
            print(f"\n[bold]step={step} loss={total_loss}:[/]")
            if embody_metrics:
                for k, v in sorted(embody_metrics.items()):
                    print(f"  {k}: {v:.4f}")
            row = [str(step), f"{total_loss:.4f}"]
            row.append(f"{float(update_info['grad_norm']):.4f}")
            table.add_row(*row)
            cfg.wandb.log(
                {
                    "training": update_info,
                    "timer": timer.get_average_times(),
                    "lowdim_active": lowdim_active,
                    "guidance_active": guide_input is not None,
                    **embody_metrics,
                },
                step=step,
            )
        eval_loop(model, state.model.params, step, is_last=step == cfg.steps - 1)

        if (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback(state, step + 1)

    # Final checkpoint
    if save_dir is not None:
        save_callback(state, cfg.steps)
        save_callback.wait()

    print(table)

    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    ratio = last / first if first > 0 else float("inf")
    print(f"\nloss: {first:.4f} -> {last:.4f}  ({ratio:.2%} of initial)")
    cfg.wandb.log({"summary": {"loss_first": first, "loss_last": last, "loss_ratio": ratio}}, step=cfg.steps - 1)

    if ratio < 0.5:
        print("[bold green]loss decreased — training works[/]")
    else:
        print("[bold yellow]loss did not decrease much — check lr or architecture[/]")

    print("\n[bold green]done.[/]")
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
