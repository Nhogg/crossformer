from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import tyro

from crossformer.data.grain.arec.arec import ArrayRecordBuilder


def _record_mode(sample: dict) -> Literal["episode", "step"]:
    return "step" if "step_id" in sample else "episode"


def _episode_length(sample: dict) -> int:
    def _find(x) -> int | None:
        if isinstance(x, np.ndarray) and x.ndim:
            return x.shape[0]
        if isinstance(x, dict):
            for v in x.values():
                l = _find(v)
                if l is not None:
                    return l
        return None

    length = _find(sample)
    if length is None:
        raise ValueError("Unable to infer episode length from sample.")
    return length


def _slice_sample(sample: dict, idx: int, length: int):
    if isinstance(sample, dict):
        return {k: _slice_sample(v, idx, length) for k, v in sample.items()}
    if isinstance(sample, np.ndarray) and sample.ndim and sample.shape[0] == length:
        return sample[idx]
    return sample


def _episode_to_steps(sample: dict) -> Iterator[dict]:
    length = _episode_length(sample)
    for idx in range(length):
        step = _slice_sample(sample, idx, length)
        step.setdefault("step_id", idx)
        yield step


def _stack_values(values: list):
    first = values[0]
    if isinstance(first, dict):
        return {k: _stack_values([v[k] for v in values]) for k in first}
    if isinstance(first, np.ndarray):
        return np.stack(values)
    if all(v == first for v in values):
        return first
    return np.asarray(values)


def _steps_to_episode(steps: list[dict]) -> dict:
    keys = steps[0].keys()
    return {k: _stack_values([s[k] for s in steps]) for k in keys}


def _episode_stream_to_steps(first: dict, rest: Iterable[dict]) -> Iterator[dict]:
    for sample in chain([first], rest):
        yield from _episode_to_steps(sample)


def _steps_stream_to_episodes(first: dict, rest: Iterable[dict]) -> Iterator[dict]:
    current = None
    buffer: list[dict] = []
    for sample in chain([first], rest):
        episode_id = sample.get("episode_id")
        if current is not None and episode_id != current:
            yield _steps_to_episode(buffer)
            buffer = []
        current = episode_id
        buffer.append(sample)
    if buffer:
        yield _steps_to_episode(buffer)


@dataclass
class ReformatConfig:
    name: str
    version: str
    to: Literal["episode", "step"]
    root: Path = Path("~/.cache/arrayrecords")
    branch: str = "main"
    shard_size: int = 100_000
    writer_options: str | None = "group_size:1"


def main(cfg: ReformatConfig):
    reader = ArrayRecordBuilder(
        name=cfg.name,
        version=cfg.version,
        branch=cfg.branch,
        root=cfg.root,
        shard_size=cfg.shard_size,
        writer_options=cfg.writer_options,
    )

    source = iter(reader)
    try:
        first = next(source)
    except StopIteration:
        raise SystemExit("Source dataset is empty.")

    mode = _record_mode(first)

    if cfg.to == "step":
        build_fn = lambda: _episode_stream_to_steps(first, source) if mode == "episode" else chain([first], source)
    else:
        build_fn = (
            lambda: _steps_stream_to_episodes(first, source)
            if mode == "step"
            else chain([first], source)
        )

    writer = ArrayRecordBuilder(
        name=cfg.name,
        version=cfg.version,
        branch=f"to_{cfg.to}",
        root=cfg.root,
        shard_size=cfg.shard_size,
        writer_options=cfg.writer_options,
    )
    writer.prepare(build_fn)


if __name__ == "__main__":
    main(tyro.cli(ReformatConfig))
