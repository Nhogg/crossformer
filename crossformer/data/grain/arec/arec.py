from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from array_record.python.array_record_data_source import ArrayRecordDataSource
from array_record.python.array_record_module import ArrayRecordWriter
import grain.python as grain
import jax
import msgpack
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm

# from array_record.python import reader as ar_reader

Shape = tuple[int, ...]
Spec = dict[str, Shape]


# ---------- Encoding helpers (msgpack) ----------


def _ndarray_to_serializable(arr: np.ndarray) -> dict[str, Any]:
    return {
        "__ndarray__": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),  # raw bytes
    }


def _ndarray_from_serializable(d: dict[str, Any]) -> np.ndarray:
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"])


def _default_pack(obj):
    if isinstance(obj, np.ndarray):
        return _ndarray_to_serializable(obj)
    if isinstance(obj, (np.integer)):
        return int(obj)
    if isinstance(obj, (np.floating)):
        return float(obj)
    if isinstance(obj, (bytes | bytearray | memoryview)):
        return bytes(obj)
    raise TypeError(f"Unsupported type for msgpack encoding: {type(obj)}")


def _default_unpack(obj):
    # msgpack feeds us dictionaries for custom types
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return _ndarray_from_serializable(obj)
    return obj


def pack_record(obj: Any) -> bytes:
    """Serialize a Python object (with optional numpy arrays) to bytes."""
    return msgpack.packb(obj, default=_default_pack, use_bin_type=True)


def unpack_record(buf: bytes) -> Any:
    """Deserialize bytes back to Python object (and numpy arrays)."""
    return msgpack.unpackb(buf, object_hook=_default_unpack, raw=False)


# ---------- File layout & metadata ----------


def _meta_path(root: Path, name: str) -> Path:
    return root / "meta.json"


def _shard_glob(root: Path, name: str) -> str:
    return str(root / name / "data-*.arrayrecord")


def _shard_path(root: Path, name: str, idx: int) -> Path:
    return root / f"data-{idx:05d}.arrayrecord"


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _schema_fingerprint(version: str, extra: dict[str, Any] | None = None) -> str:
    h = hashlib.sha256()
    h.update(version.encode("utf-8"))
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


# ---------- The template dataset ----------


class ArrayRecordBuilder:
    """
    A minimal dataset compiler/loader with on-disk ArrayRecord cache.

    Usage:
        ds = ArrayRecordBuilder(
            name="tinystories",
            root="~/.cache/arrds",
            version="v1",  # bump when schema changes
            shard_size=200_000,  # records per shard
            writer_options="group_size:32",  # passed through to ArrayRecordWriter
        )

        # 1) First run: build from a raw stream (yields Python dicts/arrays)
        ds.prepare(lambda: my_generator())

        # 2) Later runs: instantly uses cached ArrayRecords
        for row in ds:      # yields Python objects
            ...

        # Random access:
        item = ds[123]
        n = len(ds)

    Notes:
      - No TensorFlow/TFDS dependency.
      - Data is msgpack-encoded; NumPy arrays supported.
      - Reading uses ArrayRecordDataSource.read(start, end) for speed.
    """

    def __init__(
        self,
        name: str,
        version: str,
        root: str = Path("~/.cache/arrayrecords"),
        shard_size: int = 100_000,
        writer_options: str | None = "group_size:1",
        build_meta: dict[str, Any] | None = None,  # things that affect schema
    ):
        self.name = name
        self.root = Path(root).expanduser() / name / version
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self.version = version
        self.shard_size = int(shard_size)
        self.writer_options = writer_options
        self.build_meta = build_meta or {}
        self._meta: dict[str, Any] | None = None
        self._ds: ArrayRecordDataSource | None = None

    # ----- public API -----

    def prepare(self, build_fn: Callable[[], Iterable[Any]]) -> None:
        """
        Ensure ArrayRecord shards exist; if not, build from `build_fn()`.
        Rebuilds when version/schema fingerprint differs.
        """
        meta_path = _meta_path(self.root, self.name)
        existing = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)

        fp = _schema_fingerprint(self.version, self.build_meta)
        # shards = sorted(Path().glob(_shard_glob(self.root, self.name)))
        shards = sorted(self.root.glob("data-*.arrayrecord"))

        needs_build = (not existing) or existing.get("schema_fingerprint") != fp or not shards

        if needs_build:
            self._build_from_stream(build_fn)
        else:
            self._meta = existing

    def __len__(self) -> int:
        self._ensure_reader()
        return len(self._ds)

    def __getitem__(self, i: int) -> Any:
        # print('get item')
        # print(len(self))
        self._ensure_reader()
        # Use parallelized read by batched range when possible.
        rec = self._ds[i]
        return unpack_record(rec)

    def __iter__(self):
        """Chunked iteration using ArrayRecordDataSource.__getitems__ (batched).
        Chunk size is tunable; 16k is a good starting point.
        """
        self._ensure_reader()

        n = len(self._ds)
        chunk = 16_384

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # batched fast-path → list[bytes]
            for b in self._ds.__getitems__(list(range(s, e))):
                yield unpack_record(b)

    # ----- build path -----

    def spec(self, arr) -> Spec:
        return jax.tree.map(
            lambda x: (x.shape, x.dtype.name) if hasattr(x, "shape") else type(x).__name__,
            arr,
        )

    def _build_from_stream(self, build_fn: Callable[[], Iterable[Any]]) -> None:
        root = self.root
        root.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        count = 0
        shard_idx = 0
        writer: ArrayRecordWriter | None = None
        last_sample: Any | None = None

        def open_writer(si: int):
            p = _shard_path(self.root, self.name, si)
            return ArrayRecordWriter(str(p), options=self.writer_options or "")

        try:
            writer = open_writer(shard_idx)
            for sample in build_fn():
                blob = pack_record(sample)
                writer.write(blob)
                count += 1
                last_sample = sample
                if count % self.shard_size == 0:
                    writer.close()
                    shard_idx += 1
                    writer = open_writer(shard_idx)

            spec_path = root / "spec.json"
            if last_sample is not None:
                dataspec = self.spec(last_sample)
                with spec_path.open("w", encoding="utf-8") as f:
                    pprint(dataspec)
                    json.dump(dataspec, f, ensure_ascii=False, indent=2)
            elif spec_path.exists():
                spec_path.unlink()

        finally:
            if writer is not None:
                writer.close()

        meta = {
            "name": self.name,
            "version": self.version,
            "schema_fingerprint": _schema_fingerprint(self.version, self.build_meta),
            "writer_options": self.writer_options or "",
            "shard_size": self.shard_size,
            "num_records": count,
            "created_unix": int(time.time()),
            "build_seconds": round(time.time() - t0, 3),
        }
        _atomic_write_json(_meta_path(self.root, self.name), meta)
        self._meta = meta
        # Re-open reader with fresh shard list.
        self._ds = None
        self._ensure_reader()

    def _ensure_reader(self) -> None:
        if self._ds is not None:
            return
        # shard_paths = sorted(Path().glob(_shard_glob(self.root, self.name)))
        shard_paths = sorted((self.root).glob("data-*.arrayrecord"))
        if not shard_paths:
            raise FileNotFoundError(f"No ArrayRecord shards found under {self.root}.")
        self._ds = ArrayRecordDataSource([str(p) for p in shard_paths])
        if self._meta is None:
            # best-effort: reconstruct minimal meta
            self._meta = {
                "name": self.name,
                "version": self.version,
                "num_records": len(self._ds),
            }

    # ----- convenience -----

    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            mp = _meta_path(self.root, self.name)
            if mp.exists():
                with mp.open("r", encoding="utf-8") as f:
                    self._meta = json.load(f)
            else:
                # Fallback: probe reader
                self._ensure_reader()
                self._meta = {
                    "name": self.name,
                    "version": self.version,
                    "num_records": len(self._ds),
                }
        return self._meta


# ---------- The template dataset ----------


def stream_tiny(total=250, shape=(128,)):
    """raw stream (replace with your own loader)"""
    for i in range(total):
        yield {"id": i, "vec": np.random.randn(*shape).astype("float32")}


class ChunkedIndexSampler(grain.Sampler):
    def __init__(self, num_records: int, chunk: int = 16384, shuffle: bool = False, seed: int = 0):
        self._n = num_records
        self._chunk = int(chunk)
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self):
        idx = list(range(self._n))
        if self._shuffle:
            import random

            rnd = random.Random(self._seed)
            rnd.shuffle(idx)
        for s in range(0, self._n, self._chunk):
            yield idx[s : min(s + self._chunk, self._n)]  # << list[int] batch

    def __len__(self):
        return math.ceil(self._n / self._chunk)


"""
class DecodingArrayRecordSource(grain.sour
    def __init__(self, src: ArrayRecordDataSource):
        self._src = src
    def __len__(self): return len(self._src)
    def __getitem__(self, i):
        return _unpack_msgpack(self._src[i])  # single index path
    def __getitems__(self, indices):
        return [_unpack_msgpack(b) for b in self._src.__getitems__(indices)]  # batched path
"""


def build_fn_per_step(*, episodes=None, fn=None):
    assert episodes or fn
    iter = episodes if episodes else fn()
    for ep_id, ep in enumerate(iter):
        for s, step in enumerate(ep):
            rec = {"episode_id": ep_id, "step_id": s, **step}
            yield rec


stackem = lambda *xs: jax.tree.map(lambda *ys: np.stack(ys), *xs)


def build_fn_per_episode(*, episodes=None, fn=None):
    assert episodes or fn
    iter = episodes if episodes else fn()
    for ep_id, ep in enumerate(iter):
        print(type(ep))
        if isinstance(ep, list):
            rec = {"episode_id": ep_id, **stackem(*ep)}
        elif isinstance(ep, dict):
            rec = {"episode_id": ep_id, **ep}
        else:
            raise ValueError(f"Unexpected episode type: {type(ep)}")
        yield rec


def random_episodes(
    spec: Spec,
    *,
    episodes: int,
    steps: int | tuple[int, int],  # e.g., 128 or (96, 160)
    dtype=np.float32,
    seed: int = 0,
) -> Iterator[Iterator[dict[str, np.ndarray]]]:
    """
    Yields `episodes` episodes.
    Each episode is an *iterator* of step dicts; each step has keys from `spec`,
    with random floats in the given shape.
    """
    rng = np.random.default_rng(seed)

    def steps_in_episode(n: int, ep_id) -> Iterator[dict[str, np.ndarray]]:
        for s in range(n):
            step = jax.tree.map(
                lambda s: rng.random(s, dtype=np.float64).astype(dtype, copy=False),
                spec,
            )
            rec = {"episode_id": ep_id, "step": s, **step}
            yield rec

    for ep_id in tqdm(range(episodes), desc="Generating episodes"):
        if isinstance(steps, int):
            n = steps
        else:
            lo, hi = steps
            n = int(rng.integers(lo, hi + 1))
        yield from steps_in_episode(n, ep_id)


def main():
    """
    path = str(Path('tmp.arr'))
    writer = ArrayRecordWriter(path, 'group_size:1')

    for row in tqdm(stream_tiny(250), total=250):
        pack = pack_record(row)
        writer.write(pack)

    writer.close()
    """

    total = 250_000

    spec = {
        "img1": (64, 64, 3),
        "img2": (64, 64, 3),
        "img_wrist": (64, 64, 3),
        "state": {
            "cartesian": (6,),
            "joints": (7,),
            "gripper": (1,),
        },
    }
    mk_episodes = partial(
        random_episodes,
        spec,
        episodes=200,
        steps=(300, 500),
        dtype=np.float32,
        seed=42,
    )
    # mk_episodes = partial(build_fn_per_step, mk_episodes()) # build into random_episodes

    ds = ArrayRecordBuilder(
        name="tmp_spec",
        root="~/.cache/arecs",
        version="v1",  # bump when schema/layout changes
        shard_size=1000,  # records per shard
        writer_options="group_size:1",  # passed directly to ArrayRecordWriter
    )

    # Build once, then reuse: builds if missing or version changed
    # ds.prepare(partial(stream_tiny,total=total))
    ds.prepare(mk_episodes)

    index_sampler_example = grain.IndexSampler(
        num_records=len(ds),
        num_epochs=1,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
        shuffle=False,
        seed=0,
    )
    chunked_sampler = ChunkedIndexSampler(num_records=len(ds), chunk=16384, shuffle=False, seed=0)

    """
    3) Tune knobs (quick wins)
    chunk size: start at 8_192-32_768. Faster disks/net → larger chunks.
    worker_count:
    If I/O-bound (network/disk): 2-8 is often plenty.
    If decode-bound (lots of msgpack/NumPy): up to number of cores (or 2x if workers are processes).
    worker_buffer_size: 2-8; raise if your pipeline consumer is bursty.
    group_size (writer): prefer 32-256 for streaming training; 1 only for heavy random access.
    """

    loader = grain.DataLoader(
        data_source=ds._ds,
        operations=[],
        sampler=index_sampler_example,
        worker_count=2,
        worker_buffer_size=2,
    )

    # for element in tqdm(loader, total=len(ds), desc='Load with grain'):
    # a =  unpack_record(element)
    # print(a)

    print()

    print(len(ds), ds.meta)  # size + metadata
    x0 = ds[0]  # random access
    for x in tqdm(ds):  # fast sequential iterator (chunked reads)
        pass


if __name__ == "__main__":
    main()
