from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import json
import logging
from pathlib import Path
from typing import Callable, Mapping

import flax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tqdm

# from xgym.rlds.util import apply_persp, perspective_projection
# from xgym.viz.mano import overlay_palm
from crossformer.data.dataset import make_single_dataset
from crossformer.data.oxe import HEAD_TO_DATASET
from crossformer.data.utils.text_processing import TextProcessor
from crossformer.utils.mytyping import Any, Data, Sequence
from crossformer.utils.train_utils import TrainState
import wandb


class Callback:
    def __call__(self, train_state: TrainState, step: int):
        raise NotImplementedError


def create_validation_dataset(
    dataset_kwargs: dict,
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool = False,
):
    """Creates a dataset for validation and visualization purposes.

    Takes the training configuration and overwrites default parameters with more conservative
    options to ensure stable memory consumption.
    """
    return make_single_dataset(
        dataset_kwargs={
            **dataset_kwargs,
            "num_parallel_reads": 4,
            "num_parallel_calls": 4,
            "shuffle": False,
        },
        traj_transform_kwargs={
            **traj_transform_kwargs,
            "num_parallel_calls": 4,
        },
        frame_transform_kwargs={
            **frame_transform_kwargs,
            "num_parallel_calls": 16,
        },
        train=train,
    )


@dataclass
class SaveCallback(Callback):
    """Callback that saves checkpoints under ``save_dir``."""

    save_dir: Path | str | None

    def __post_init__(self):
        if self.save_dir is None:
            return

        # Ensure save_dir is a Path
        self.save_dir = Path(self.save_dir).expanduser().resolve()

        # Only process 0 makes dirs (safe in multi-host)
        if jax.process_index() == 0:
            (self.save_dir / "state").mkdir(parents=True, exist_ok=True)
            (self.save_dir / "params").mkdir(parents=True, exist_ok=True)
            logging.info(f"Created checkpoint dirs under {self.save_dir}")

        # --- Managers ---
        # 1. Keep only the latest full TrainState
        self.ckpt_path = self.save_dir / "state"
        self.params_path = self.save_dir / "params"
        self.state_mngr = ocp.CheckpointManager(
            self.ckpt_path,
            ocp.StandardCheckpointer(),
            # {"state": ocp.StandardCheckpointer()},
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                enable_async_checkpointing=True,
                enable_background_delete=True,
            ),
        )

        # 2. Keep all params-only checkpoints
        self.params_mngr = ocp.CheckpointManager(
            self.save_dir / "params",
            ocp.StandardCheckpointer(),
            # {"params": ocp.StandardCheckpointer()},
            options=ocp.CheckpointManagerOptions(
                max_to_keep=None,  # unlimited
                enable_async_checkpointing=True,
                enable_background_delete=True,
            ),
        )

    def __call__(self, train_state: TrainState, step: int):
        if self.save_dir is None:
            return

        # Save params every step
        self.params_mngr.save(
            step,
            args=ocp.args.StandardSave(
                item=train_state.model.params,
                # save_args=orbax_utils.save_args_from_target(self.params),
            ),
        )
        self.save_extra(train_state)

        # Save full TrainState (retains latest only)
        if self.state_mngr.should_save(step):
            self.state_mngr.save(
                step,
                args=ocp.args.StandardSave(
                    item=train_state,
                    # save_args= orbax_utils.save_args_from_target(train_state),
                ),
            )

    def wait(self):
        """Wait until all async checkpointing is done."""
        if self.save_dir is None:
            return
        self.params_mngr.wait_until_finished()
        self.state_mngr.wait_until_finished()

    def save_extra(self, train_state: TrainState):
        if jax.process_index() != 0:
            return
        model = train_state.model

        # save config
        config_path = self.ckpt_path / "config.json"
        if not config_path.exists():
            with config_path.open("w") as f:
                json.dump(model.config, f)

        # save example batch
        example_batch_path = self.ckpt_path / "example_batch.msgpack"
        if not example_batch_path.exists():
            with example_batch_path.open("wb") as f:
                f.write(flax.serialization.msgpack_serialize(model.example_batch))

        # save dataset statistics
        dataset_statistics_path = self.ckpt_path / "dataset_statistics.json"
        if not dataset_statistics_path.exists():
            with dataset_statistics_path.open("w") as f:
                json.dump(
                    jax.tree_map(lambda x: x.tolist(), model.dataset_statistics),
                    f,
                )


def remove_text(tasks: Data, zero_text_encoding: Data | None):
    """Replaces language encoding inside task dict with that of the empty string.

    zero_text_encoding = jax.tree_map(lambda x: x[0], text_processor.encode([""]))
    """
    if zero_text_encoding is None:
        zero_text_encoding = jnp.zeros((1,))
    if "language_instruction" in tasks:
        new_language = jax.tree_map(
            lambda x, example: jnp.broadcast_to(example[None], x.shape),
            tasks["language_instruction"],
            zero_text_encoding,
        )
        new_pad_dict = flax.core.copy(
            tasks["pad_mask_dict"],
            {"language_instruction": jnp.zeros_like(tasks["pad_mask_dict"]["language_instruction"])},
        )
        tasks = flax.core.copy(tasks, {"language_instruction": new_language, "pad_mask_dict": new_pad_dict})
    return tasks


def remove_images(tasks: Data):
    """Replaces images inside task dict with zero (black) images."""
    updates = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
    updates["pad_mask_dict"] = flax.core.copy(
        tasks["pad_mask_dict"],
        {k: jnp.zeros_like(v) for k, v in tasks["pad_mask_dict"].items() if "image" in k},
    )
    return flax.core.copy(tasks, updates)


@dataclass
class ValidationCallback(Callback):
    loss_fn: Callable
    process_batch_fn: Callable[[Data], Data]
    text_processor: TextProcessor | None
    val_dataset_kwargs_list: Sequence[Mapping[str, Any]]
    dataset_kwargs: Mapping[str, Any]
    val_shuffle_buffer_size: int
    num_val_batches: int
    modes_to_evaluate: Sequence[str] = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(lambda x: x[0], self.text_processor.encode(""))
        else:
            self.zero_text = None
        self.val_iterators = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=self.train,
            )
            val_iterator = (
                val_dataset.unbatch()
                .shuffle(self.val_shuffle_buffer_size)
                .repeat()
                .batch(self.dataset_kwargs["batch_size"])
                .iterator(prefetch=0)
            )
            val_iterator = map(self.process_batch_fn, val_iterator)
            self.val_iterators[single_dataset_kwargs["name"]] = val_iterator

        def eval_step(state: TrainState, batch: Data):
            loss_fn_partial = partial(
                self.loss_fn,
                params=state.model.params,
                rng=state.rng,
                train=False,
            )
            all_tasks = {}

            if "base" in self.modes_to_evaluate:
                all_tasks["base"] = batch["task"]
            if "image_conditioned" in self.modes_to_evaluate:
                all_tasks["image_conditioned"] = remove_text(batch["task"], self.zero_text)
            if "text_conditioned" in self.modes_to_evaluate:
                all_tasks["text_conditioned"] = remove_images(batch["task"])

            if "unconditioned" in self.modes_to_evaluate:
                all_tasks["unconditioned"] = remove_text(remove_images(batch["task"]), self.zero_text)
            return {
                k: loss_fn_partial(batch=flax.core.copy(batch, {"task": tasks}))[1] for k, tasks in all_tasks.items()
            }

        self.eval_step = eval_step

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        for name, val_data_iter in self.val_iterators.items():
            metrics = []
            for _, batch in tqdm.tqdm(
                zip(range(self.num_val_batches), val_data_iter),
                total=self.num_val_batches,
                desc=name,
            ):
                metrics.append(self.eval_step(train_state, batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_metrics[f"validation_{name}"] = metrics
        return wandb_metrics


@dataclass
class VisCallback(ValidationCallback):
    stats: dict = field(default_factory=dict)

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        for name, val_data_iter in self.val_iterators.items():
            metrics = []
            videos = []
            for _, batch in tqdm.tqdm(
                zip(range(self.num_val_batches), val_data_iter),
                total=self.num_val_batches,
                desc=name,
            ):
                metric = self.eval_step(train_state, batch)

                if name in HEAD_TO_DATASET.get("mano", []):
                    # print(metric.keys())
                    k = next(iter(metric.keys()))[0]
                    assert "vis" in metric[k], "dont use VisCallback if no vis key in metric"
                    vis = {k: v.pop("vis") for k, v in metric.items()}[k]

                    # NOTE you might want/need to use more of the dataset_kwargs
                    # to figure out what needs to be plotted
                    videos += self.plot(batch, vis, name)

                metrics.append(metric)
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)

            metrics["video"] = videos[:32]  # avoid mean pooling metrics
            wandb_metrics[f"validation_{name}"] = metrics
        return wandb_metrics

    def denormalize(self, thing, name, key="action"):
        """denormalizes the thing with mean and std where mask is True"""
        mask = self.stats[name][key]["mask"]
        mean = self.stats[name][key]["mean"]
        std = self.stats[name][key]["std"]
        thing = np.where(mask, (thing * std + mean), thing)
        return thing

    def plot(self, batch, vis: dict, name):
        """returns list of wandb Video objects"""

        batch = jax.tree.map(np.array, batch)
        deltas = np.array(vis["deltas"])

        """ Tried to save... makes slow
        d = {
            "batch": batch,
            "vis": vis,
            "stats": self.stats,
        }
        """

        act = self.denormalize(deltas, name)  # action deltas
        b, w, h, actdim = act.shape

        useop = actdim > 7  # use openpose or palm... for now assume False

        proprio = batch["observation"]["proprio_mano"]  # doesnt have a horizon dim
        joints, focal = proprio[..., :7], proprio[..., -1]
        # joints, focal = proprio[..., : 21 * 3], proprio[..., -1]

        # TODO ndimentional vmap and jit all the np functions
        videos = []
        for i in range(b):  # samples of batch
            imgs = []
            for j in range(w):  # windows of time
                img = batch["observation"]["image_primary"][i, j]

                p3d = joints[i, j, :3]
                for k in range(-1, h):  # horizons of prediction
                    if k == -1:
                        grip = joints[i, j, -1]
                    else:
                        p3d += act[i, j, k, :3]
                        grip = act[i, j, k, -1]
                    # print(f"grip: {grip} {grip.shape}")

                    H, W = img.shape[0], img.shape[0]
                    f = focal[i, j]
                    P = perspective_projection(f, H, W)

                    # expects b,points,3
                    p2d = apply_persp(np.array([[p3d]]), P)  # expand to get batch dim 1
                    palm = p2d[0, 0]

                    x, y = int(palm[0]), int(palm[1])
                    size = float(p3d[2])  # float(grip)
                    sigmoid = lambda x: 1 / (1 + np.exp(-x))
                    size = int(4 * sigmoid(size) ** 0.5) + 2
                    # xgym.logger.info(f"x,y,s: {(x,y,size)}")

                    img = overlay_palm(img, x=x, y=y, opacity=k, size=size)

                    imgs.append(img)  # debug
                imgs.append(img)
            imgs = np.array(imgs).transpose(0, 3, 1, 2)  # colors first T-CWH
            videos.append(wandb.Video(imgs, fps=10))

        return videos[:32]
