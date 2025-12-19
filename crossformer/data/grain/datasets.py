"""Dataset helpers for working with ArrayRecord sources."""

from __future__ import annotations

import collections
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Generic, Iterable, Sequence, TypeVar

from array_record.python.array_record_data_source import ArrayRecordDataSource
import grain
import jax
import jax.numpy as jnp
import numpy as np
from rich.pretty import pprint

from crossformer.data.arec.arec import unpack_record
from crossformer.data.grain.util.deco import logbar
from crossformer.utils.jax_utils import cpu, npstr2jax, str2jax

log = logging.getLogger(__name__)


class _DecodedArrayRecord:
    """Decode ArrayRecord shards into Python dictionaries."""

    def __init__(self, shards: Iterable[Path], unpack: bool = False) -> None:
        self._shards = tuple(sorted(Path(p) for p in shards))
        self._ds = ArrayRecordDataSource([str(p) for p in self._shards])
        self._unpack = unpack

    @property
    def shards(self) -> tuple[Path, ...]:
        return self._shards

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._ds)

    def __getitem__(self, index: int):  # pragma: no cover - simple delegation
        return unpack_record(self._ds[index])

    def __getitems__(self, indices: Sequence[int]) -> list[dict]:
        if self._unpack:
            return [unpack_record(x) for x in self._ds.__getitems__(indices)]
        return self._ds.__getitems__(indices)


def _postprocess_episode(items: Sequence[dict[str, Any]], device=cpu, steps=True) -> Sequence[dict[str, jnp.Array]]:
    """postprocess msgpack-decoded episode data into jax arrays.
    unfortunately cannot jit this"""

    def maybe_str2jax(x: Any) -> Any:
        if isinstance(x, str):
            return str2jax(x, device=cpu)
        if isinstance(x, np.ndarray) and x.dtype.kind == "U":
            return npstr2jax(x, device=cpu)
        return x

    def stack_items(*xs):
        return jnp.stack(xs)

    items = jax.tree.map(maybe_str2jax, items)
    items = jax.tree.map(partial(jnp.array, device=cpu), items)
    # we dont stack if it is stepwise
    return items if not steps else jax.tree.map(stack_items, *_items)


class EpisodeInfo:
    def __init__(self, ds, mix, *, cache: bool = True, cache_dir: Path | None = None):
        self.ds = ds
        self._cache_enabled = cache
        self.mix = mix
        self.shards = mix.get_shards()
        self._cache_dir = self._resolve_cache_dir(cache_dir)

        # self._episode_indices = self.get_lengths()

    def get_lengths(self) -> list[list[int]]:
        cached = self._load_cached_indices()
        return cached if cached else self.do_lengths()

    def do_lengths(self):
        log.info("No cached episode indices found; grouping from scratch.")

        _bs = 1024
        mpds = (
            self.ds.to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=512))
            # .batch(_bs)
            .mp_prefetch(grain.MultiprocessingOptions(num_workers=32))
        )
        mpit = iter(mpds)

        lengths = collections.defaultdict(int)
        for x in logbar(mpit, desc="Computing episode lengths...", total=len(self.ds)):
            eid = int(x["episode_id"])
            sid = int(x["step_id"])
            lengths[eid] = max(lengths[eid], sid + 1)

        self._store_cached_indices(lengths)
        log.info(f"Cached idxs: {len(cached)}ep : {sum([len(x) for x in cached])}it")
        return lengths

    def _resolve_cache_dir(self, cache_dir: Path | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return root / "arrayrecords"

    def _cache_key(self) -> str | None:
        shards = self.shards
        if not shards:
            return None

        fingerprint_parts = []
        for shard in shards:
            try:
                stat = shard.stat()
            except OSError:
                return None
            fingerprint_parts.append(f"{shard.resolve()}::{int(stat.st_mtime)}::{stat.st_size}")

        digest = hashlib.sha1("\n".join(fingerprint_parts).encode()).hexdigest()
        return digest

    @property
    def path(self) -> Path | None:
        key = self._cache_key()
        d = self._cache_dir / self.mix.loc / str(key) / "episode_indices.json"
        return None if not self._cache_enabled and key else d

    def _load_cached_indices(self) -> list[list[int]] | None:
        path = self.path
        pprint(path)
        if path is None or not path.exists():
            return
        log.info(f"Loading cached episode indices from {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return payload

    def _store_cached_indices(self, indices: list[list[int]]) -> None:
        path = self.path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(indices, handle)
        except OSError:
            return

    def __len__(self) -> int:
        return len(self._episode_indices)

    @property
    def n_steps(self) -> int:
        cached = self._load_cached_indices()
        return sum([len(x) for x in cached])

    def lengths(self) -> list[int]:
        """Return the lengths of all records in the dataset."""
        return [len(idxs) for idxs in self._episode_indices]

    def __iter__(self) -> Iterator[dict]:
        self._i = 0
        return self


T = TypeVar("T")


def run_in_background(fn, /, *args, **kwargs) -> threading.Thread:
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()


class CacheIter(Iterator[T], Generic[T]):
    """
    Wraps a parent iterator/sequence.
    - __getitem__(i): returns parent[i], caching the result.
    - __next__(): returns next(parent), caching by increasing index (0,1,2,...).
    - __iter__(): standard iterator protocol (returns self so caching applies during iteration).
    """

    def __init__(self, parent):
        self._parent = parent
        self._cache: dict[int, T] = {}
        self._i = 0  # next index to assign for __next__-driven iteration
        self._pool = ThreadPoolExecutor(max_workers=16)
        self.preload()

    def preload(self):
        """Preload the entire parent into the cache in a background thread."""
        for i in range(len(self._parent)):
            _f = self._pool.submit(self.__getitem__, i)  # uses our __getitem__, which caches

    def __getitem__(self, idx: int) -> T:
        if str(idx) in self._cache:
            return self._cache[str(idx)]
        log.warning(f"CacheIter miss item {idx}")
        # Prefer direct delegation if the parent supports random access
        if hasattr(self._parent, "__getitem__"):
            val = self._parent[idx]
            self._cache[idx] = val
            return val
        # Fallback: advance the parent until we reach idx
        while self._i <= idx:
            _ = next(self)  # uses our __next__, which caches
        return self._cache[idx]

    def __iter__(self) -> CacheIter[T]:
        # Return self so iteration goes through our __next__ and gets cached.
        return self

    def __next__(self) -> T:
        val = next(self._parent)
        self._cache[self._i] = val
        self._i += 1
        return val

    def __len__(self) -> int:
        return len(self._parent)

    @property
    def cache(self) -> dict[int, T]:
        return self._cache


def drop(tree: dict, keys: list[str]) -> dict:
    return {k: v for k, v in tree.items() if k not in keys}


class _DropKeyDataset(Sequence[dict]):
    """Dataset wrapper that filters observation keys."""

    def __init__(self, dataset: Sequence[dict], drop_keys: Sequence[str] = ()):
        self._dataset = dataset
        self._drop_keys = set(drop_keys)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        traj = self._dataset[index]
        if not self._drop_keys:
            return traj
        return _drop_observation_keys(traj, self._drop_keys)


def _drop_observation_keys(traj: dict, drop_keys: set[str]) -> dict:
    obs = traj.get("observation")
    if not isinstance(obs, dict) or not drop_keys:
        return traj

    filtered: dict[str, Any] = {}
    for key, value in obs.items():
        if _should_drop(drop_keys, key):
            continue
        if isinstance(value, dict):
            if key == "proprio":
                nested = {
                    sub_key: sub_val
                    for sub_key, sub_val in value.items()
                    if not _should_drop(drop_keys, f"{key}_{sub_key}", sub_key, f"{key}/{sub_key}")
                }
                if nested:
                    filtered[key] = nested
                continue
            filtered[key] = value
            continue
        filtered[key] = value

    new_traj = dict(traj)
    new_traj["observation"] = filtered
    return new_traj


def _should_drop(drop_keys: set[str], *candidates: str) -> bool:
    for candidate in candidates:
        if candidate is None:
            continue
        for alias in _expand_aliases(candidate):
            if alias in drop_keys:
                return True
    return False


def _expand_aliases(key: str) -> set[str]:
    if not key:
        return {""}

    aliases = {key}
    cleaned = key.lstrip("/")
    aliases.add(cleaned)

    if "_" in cleaned:
        prefix, suffix = cleaned.split("_", 1)
        aliases.add(suffix)
        if prefix:
            aliases.add(f"{prefix}/{suffix}")

    if "/" in cleaned:
        prefix, suffix = cleaned.split("/", 1)
        if prefix:
            aliases.add(f"{prefix}_{suffix}")
        aliases.add(suffix)

    aliases.add(cleaned.split("/")[-1])

    return {alias for alias in aliases if alias}
