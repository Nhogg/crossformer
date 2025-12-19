from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from typing import Any, ClassVar
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import xgym
from xgym.rlds.util.trajectory import binarize_gripper_actions as binarize
from xgym.rlds.util.trajectory import scan_noop

log = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)

threshold = 1e-3


class Builder:
    """DatasetBuilder Base for LUC XGym"""

    # VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES: ClassVar = {
        "1.0.0": "Initial release.",
        "2.0.0": "more data and overhead cam",
        "3.0.0": "relocated setup",
        "4.0.0": "50hz data",
    }

    def __init__(self, workers=32):
        # super().__init__()
        self.spec = lambda arr: jax.tree.map(lambda x: x.shape if hasattr(x, "shape") else type(x), arr)
        self.threshold = 1e-3
        self.workers = workers

    """Dataset metadata (homepage, citation,...)."""
    """
    def _info(self) -> tfds.core.DatasetInfo:

        def feat_im(doc):
            return tfds.features.Image(
                shape=(224, 224, 3),
                dtype=np.uint8,
                encoding_format="png",
                doc=doc,
            )

        def feat_prop():
            return tfds.features.FeaturesDict(
                {
                    "joints": tfds.features.Tensor(
                        shape=[7],
                        dtype=np.float32,
                        doc="Joint angles. radians",
                    ),
                    "position": tfds.features.Tensor(
                        shape=[6],
                        dtype=np.float32,
                        doc="Joint positions. xyz millimeters (mm) and rpy",
                    ),
                    "gripper": tfds.features.Tensor(
                        shape=[1],
                        dtype=np.float32,
                        doc="Gripper position. 0-850",
                    ),
                }
            )

        features = tfds.features.FeaturesDict(
            {
                "steps": tfds.features.Dataset(
                    {
                        "observation": tfds.features.FeaturesDict(
                            {
                                "image": tfds.features.FeaturesDict(
                                    {
                                        "worm": feat_im(
                                            doc="Low front logitech camera RGB observation."
                                        ),
                                        "side": feat_im(
                                            doc="Low side view logitech camera RGB observation."
                                        ),
                                        "overhead": feat_im(
                                            doc="Overhead logitech camera RGB observation."
                                        ),
                                        "wrist": feat_im(
                                            doc="Wrist realsense camera RGB observation."
                                        ),
                                    }
                                ),
                                "proprio": feat_prop(),
                            }
                        ),
                        "action": feat_prop(),  # TODO does it make sense to store proprio and  actions?
                        #
                        "discount": tfds.features.Scalar(
                            dtype=np.float32,
                            doc="Discount if provided, default to 1.",
                        ),
                        "reward": tfds.features.Scalar(
                            dtype=np.float32,
                            doc="Reward if provided, 1 on final step for demos.",
                        ),
                        "is_first": tfds.features.Scalar(
                            dtype=np.bool_, doc="True on first step of the episode."
                        ),
                        "is_last": tfds.features.Scalar(
                            dtype=np.bool_, doc="True on last step of the episode."
                        ),
                        "is_terminal": tfds.features.Scalar(
                            dtype=np.bool_,
                            doc="True on last step of the episode if it is a terminal step, True for demos.",
                        ),
                        "language_instruction": tfds.features.Text(
                            doc="Language Instruction."
                        ),
                        "language_embedding": tfds.features.Tensor(
                            shape=(512,),
                            dtype=np.float32,
                            doc="Kona language embedding. "
                            "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                        ),
                    }
                ),
                "episode_metadata": tfds.features.FeaturesDict({}),
            }
        )

        # self.file_format=tfds.core.FileFormat.ARRAY_RECORD
        return tfds.core.DatasetInfo(
            builder=self,
            features=features,
        )
    """

    def build(self):
        """Define data splits."""

        # if not self.root:
        # print(self.root)
        # self.root = Path.home() / f"{self.name}:{ str(self.VERSION)[0]}"
        self.files = list(self.root.rglob("*.dat"))
        if self.limit:
            self.files = self.files[: self.limit]
        return self._generate_examples(self.files)

    def dict_unflatten(self, flat, sep="."):
        """Unflatten a flat dictionary to a nested dictionary."""

        nest = {}
        for key, value in flat.items():
            keys = key.split(sep)
            d = nest
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        return nest

    def _parse_example(self, path):
        log.debug(path)

        try:
            info, ep = xgym.viz.memmap.read(path)
            cams = [k for k in info["schema"] if "camera" in k]
            if len(cams) < 2:
                raise ValueError(f"Not enough cameras {cams}")
        except Exception as e:
            xgym.logger.error(f"Error reading {path}")
            xgym.logger.error(e)
            return None

        n = len(ep["time"])

        # pprint(self.spec(ep))

        ### cleanup and remap keys
        ep.pop("time")
        ep.pop("gello_joints")

        ep["robot"] = {}
        ep["robot"]["joints"] = ep.pop("xarm_joints")
        ep["robot"]["position"] = ep.pop("xarm_pose")
        ep["robot"]["gripper"] = ep.pop("xarm_gripper")
        # ep['robot'] = {'joints': joints, 'position': np.concatenate((pose, grip), axis=1)}

        if "/xgym/camera/worm" not in ep or "/xgym/camera/low" in ep:
            ep["/xgym/camera/worm"] = ep.pop("/xgym/camera/low")
        if "/xgym/camera/wrist" not in ep:
            ep["/xgym/camera/wrist"] = ep.pop("/xgym/camera/rs")

        ep["image"] = {k: ep.pop(f"/xgym/camera/{k}") for k in ["worm", "side", "wrist"]}

        ### scale and binarize
        ep["robot"]["gripper"] /= 850
        ep["robot"]["position"][:, :3] /= 1e3
        _binarize = partial(binarize, open=0.95, close=0.4)  # doesnt fully close
        ep["robot"]["gripper"] = np.array(_binarize(jnp.array(ep["robot"]["gripper"])))

        ### filter noop cartesian
        pos = np.concatenate((ep["robot"]["position"], ep["robot"]["gripper"]), axis=1)
        noops = np.array(scan_noop(jnp.array(pos), threshold=self.threshold))
        mask = ~noops
        # filter noop joints
        jpos = np.concatenate([ep["robot"]["joints"], ep["robot"]["gripper"]], axis=1)
        jnoop = np.array(scan_noop(jnp.array(jpos), threshold=self.threshold))
        jmask = ~jnoop
        mask = np.logical_and(mask, jmask)

        log.debug(f"Kept {mask.sum()} of {n} steps")

        ep = jax.tree.map(select := lambda x: x[mask], ep)

        ### calculate action
        action = jax.tree.map(lambda x: x[1:] - x[:-1], ep["robot"])  # pose and joint action
        action["gripper"] = ep["robot"]["gripper"][1:]  # gripper is absolute
        ep = jax.tree.map(lambda x: x[:-1], ep)
        # ep["action"] = action # action is not an observation

        # pprint(self.spec(ep))

        ### final remaps
        ep["proprio"] = ep.pop("robot")

        geti = lambda x, i: jax.tree.map(lambda y: y[i], x)
        episode = [
            {
                "observation": geti(ep, i),
                "action": geti(action, i),
                "discount": 1.0,
                "reward": float(i == (len(ep) - 1)),
                "is_first": i == 0,
                "is_last": i == (len(ep) - 1),
                "is_terminal": i == (len(ep) - 1),
                "language_instruction": self.task,
                "language_embedding": self.lang,
            }
            for i in range(len(ep["proprio"]["position"]))
        ]

        # if you want to skip an example for whatever reason, simply return None
        sample = {"steps": episode, "episode_metadata": {}}
        id = f"{path.parent.name}_{path.stem}"
        return episode if len(episode) > 0 else None
        return id, sample

    def _generate_examples(self, ds) -> Iterator[tuple[str, Any]]:
        """Generator of examples for each split."""

        self.taskfile = next(self.root.glob("*.npy"))  # from: cwd
        self.task = self.taskfile.stem.replace("_", " ")
        self.lang = np.load(self.taskfile)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for result in ex.map(self._parse_example, ds):
                if result is not None:
                    yield result

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
