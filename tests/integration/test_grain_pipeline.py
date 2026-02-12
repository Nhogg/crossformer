"""Integration: grain data pipeline contract — synthetic data flows through the full pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crossformer.data.grain import builders, pipelines
from crossformer.utils.spec import ModuleSpec

from .conftest import ACTION_HORIZON, HEAD_NAME, requires_gpu


def _make_trajectory(length: int, *, language: str, seed: int = 0) -> dict:
    """Synthetic trajectory matching the grain conftest pattern."""
    rng = np.random.default_rng(seed)
    observation = {
        "img1": rng.integers(0, 255, size=(length, 16, 16, 3), dtype=np.uint8),
        "state": {
            "cartesian": rng.normal(size=(length, 6)).astype(np.float32),
            "joints": rng.normal(size=(length, 7)).astype(np.float32),
            "gripper": rng.normal(size=(length, 1)).astype(np.float32),
        },
        "language_embedding": rng.normal(size=(length, 8)).astype(np.float32),
    }
    return {
        "observation": observation,
        "action": rng.normal(size=(length, 7)).astype(np.float32),
        "language_instruction": np.array([language] * length, dtype=object),
    }


def _standardize(traj: dict) -> dict:
    obs = dict(traj["observation"])
    state = obs.pop("state")
    for key, value in state.items():
        obs[f"state_{key}"] = value
    traj = dict(traj)
    traj["observation"] = obs
    return traj


@requires_gpu
@pytest.mark.integration
class TestGrainPipeline:
    def test_pipeline_produces_frames(self, tmp_path: Path):
        """Full pipeline: source -> standardize -> window -> chunk -> output frames."""
        trajectories = [
            _make_trajectory(10, language="pick up the cup", seed=0),
            _make_trajectory(8, language="place on table", seed=1),
        ]

        cfg = builders.GrainDatasetConfig(
            name="test_dataset",
            source=lambda: trajectories,
            standardize_fn=ModuleSpec.create(_standardize),
            image_obs_keys={"main": "img1"},
            depth_obs_keys={},
            proprio_obs_keys={
                "cartesian": "state_cartesian",
                "joints": "state_joints",
                "gripper": "state_gripper",
            },
            proprio_obs_dims={"cartesian": 6, "joints": 7, "gripper": 1},
            language_key="language_instruction",
            statistics_save_dir=str(tmp_path / "stats"),
        )

        result = pipelines.make_single_dataset(
            cfg,
            train=False,
            traj_transform_kwargs={
                "window_size": 1,
                "action_horizon": ACTION_HORIZON,
                "skip_unlabeled": True,
                "goal_relabeling_strategy": "uniform",
                "max_action": 10.0,
                "max_proprio": 10.0,
                "head_to_dataset": {HEAD_NAME: ["test_dataset"]},
            },
            shuffle_buffer_size=4,
            seed=42,
        )

        frames = list(result.dataset)
        assert len(frames) > 0, "Pipeline produced no frames"

        frame = frames[0]
        assert frame["dataset_name"] == "test_dataset"
        assert "observation" in frame or "action" in frame
        assert result.statistics.num_trajectories == 2

    def test_pipeline_statistics_computed(self, tmp_path: Path):
        """Dataset statistics (mean/std) should be computed from trajectories."""
        trajectories = [_make_trajectory(6, language="wave", seed=99)]

        cfg = builders.GrainDatasetConfig(
            name="stats_test",
            source=lambda: trajectories,
            standardize_fn=ModuleSpec.create(_standardize),
            image_obs_keys={"main": "img1"},
            depth_obs_keys={},
            proprio_obs_keys={"cartesian": "state_cartesian"},
            proprio_obs_dims={"cartesian": 6},
            language_key="language_instruction",
            statistics_save_dir=str(tmp_path / "stats"),
        )

        result = pipelines.make_single_dataset(
            cfg,
            train=False,
            traj_transform_kwargs={
                "window_size": 1,
                "action_horizon": 2,
                "skip_unlabeled": False,
                "max_action": 10.0,
                "max_proprio": 10.0,
                "head_to_dataset": {HEAD_NAME: ["stats_test"]},
            },
            shuffle_buffer_size=2,
            seed=0,
        )

        stats = result.statistics
        assert stats.num_trajectories == 1
        assert stats.action.mean.shape[-1] == 7  # action dim from trajectory
