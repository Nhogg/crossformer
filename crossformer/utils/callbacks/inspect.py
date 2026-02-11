from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping

import jax
import numpy as np

from crossformer.utils.mytyping import Any, Data
import wandb


@dataclass
class InspectCallback:
    """Collects per-joint histograms of the target actions for WandB logging."""

    log_every: int = 100
    prefix: str = "inspect"
    # @codex todo
    # keys: Sequence[str] = field(default_factory=lambda: ["action"])
    # @codex todo add tests

    def every(self, batch: Data, step: int) -> dict[str, Any]:
        if (step + 1) % self.log_every != 0:
            return {}
        return self(batch)

    def __call__(self, batch: Data) -> dict[str, Any]:
        if jax.process_index() != 0:
            return {}

        action = batch["action"]
        mask = batch.get("action_pad_mask", None)
        mask_array = None
        if mask is not None:
            mask_array = np.asarray(jax.device_get(mask)).astype(bool).reshape(-1)

        return self._collect_logs(action, mask_array, self.prefix)

    def _collect_logs(self, value: Any, mask: np.ndarray | None, prefix: str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            logs: dict[str, Any] = {}
            for key, sub_value in value.items():
                logs.update(self._collect_logs(sub_value, mask, f"{prefix}/{key}"))
            return logs
        return self._hist_for_array(value, mask, prefix)

    def _hist_for_array(self, value: Any, mask: np.ndarray | None, prefix: str) -> dict[str, Any]:
        arr = np.asarray(jax.device_get(value))
        arr = arr.reshape(-1, 1) if arr.ndim <= 1 else arr.reshape(-1, arr.shape[-1])

        if mask is not None:
            if mask.shape[0] == arr.shape[0]:
                arr = arr[mask]
            else:
                logging.warning(f"mask has incompatible shape {mask.shape} (expected {arr.shape})")

        if arr.size == 0:
            return {}

        logs = {}
        for idx in range(arr.shape[-1]):
            logs[f"{prefix}/joint_{idx:02d}"] = wandb.Histogram(arr[:, idx])
        return logs
