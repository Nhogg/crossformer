from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import pickle
from typing import (
    Any,
)

import arec
import generate
import msgpack  # type: ignore
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
import tyro

_HAS_MSGPACK = True


LoaderResult = Any


def _load_pickle(path: Path) -> LoaderResult:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_numpy(path: Path) -> LoaderResult:
    # Try np.load; allow pickled objects for flexibility
    with np.load(path, allow_pickle=True) as npz:  # works for .npz; for raw .npy np.load returns array
        return {k: npz[k] for k in npz.files}


def _load_numpy_generic(path: Path) -> LoaderResult:
    obj = np.load(path, allow_pickle=True)
    return obj


def _load_memmap(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> LoaderResult:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _load_json(path: Path) -> LoaderResult:
    with path.open("r") as f:
        return json.load(f)


def _load_msgpack(path: Path) -> LoaderResult:
    with path.open("rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_any(path: Path) -> LoaderResult:
    """Try a sequence of formats to load an object from a .dat file.

    Order: pickle, numpy (npz/npy), json, msgpack (if available).
    """
    # 1) Pickle
    try:
        return _load_pickle(path)
    except Exception:
        pass

    # 2) NumPy (npz)
    try:
        return _load_numpy(path)
    except Exception:
        pass

    # 2b) NumPy (generic npy)
    try:
        return _load_numpy_generic(path)
    except Exception:
        pass

    # 3) JSON (text)
    try:
        return _load_json(path)
    except Exception:
        pass

    # 4) msgpack
    try:
        return _load_msgpack(path)
    except Exception:
        pass

    raise ValueError(f"Unrecognized or unsupported file format for {path}")


# -----------------------------
# Record streaming
# -----------------------------
Record = Mapping[str, Any]


def iter_records(obj: Any) -> Iterator[Record]:
    """Yield dict-like records from a loaded object.

    Rules:
      - If obj is a dict-like, yield it once.
      - If obj is a sequence/iterable of dict-like, yield each dict.
      - If obj is a NumPy structured array, yield row dicts.
      - Otherwise, wrap into a single record under key "value".
    """
    if isinstance(obj, Mapping):
        yield obj  # type: ignore
        return

    # Structured numpy array → dict per row
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None:
        for row in obj:
            rec = {name: row[name] for name in obj.dtype.names}
            yield rec
        return

    # Generic sequence of dicts
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        for el in obj:
            if isinstance(el, Mapping):
                yield el  # type: ignore
            else:
                yield {"value": el}
        return

    # Fallback
    yield {"value": obj}


# -----------------------------
# Spec inference & merging
# -----------------------------
Spec = dict[str, Any]


def _leaf_spec_for_value(x: Any) -> Any:
    """Return a leaf spec for a value.

    - np.ndarray → {"shape": tuple, "dtype": str}
    - scalar numbers/strings → type label
    - sequences → summarize element specs if homogenous else list of unique
    - mappings → recurse (handled elsewhere)
    """
    if isinstance(x, np.ndarray):
        return {"shape": tuple(int(d) for d in x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (np.generic,)):
        return {"shape": (), "dtype": str(np.asarray(x).dtype)}
    if isinstance(x, (int, float, bool, str)):
        return type(x).__name__
    if isinstance(x, (bytes, bytearray)):
        return f"bytes(len={len(x)})"
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        # Summarize element specs (best-effort)
        elems = list(x)
        if not elems:
            return "list(empty)"
        specs = [_leaf_spec_for_value(e) if not isinstance(e, Mapping) else infer_spec(e) for e in elems]
        return _reduce_to_union(specs)
    return type(x).__name__


def infer_spec(obj: Any) -> Spec:
    if isinstance(obj, Mapping):
        out: Spec = {}
        for k, v in obj.items():
            if isinstance(v, Mapping):
                out[k] = infer_spec(v)
            else:
                out[k] = _leaf_spec_for_value(v)
        return out
    # Wrap non-mapping under a sentinel key
    return {"value": _leaf_spec_for_value(obj)}


def _merge_leaf(a: Any, b: Any) -> Any:
    if a == b:
        return a
    # Merge shapes/dtypes if both look like array specs
    if (
        isinstance(a, Mapping)
        and isinstance(b, Mapping)
        and set(a.keys()) == {"shape", "dtype"}
        and set(b.keys()) == {"shape", "dtype"}
    ):
        shape = a["shape"] if a["shape"] == b["shape"] else list({tuple(a["shape"]), tuple(b["shape"])})
        dtype = a["dtype"] if a["dtype"] == b["dtype"] else list({a["dtype"], b["dtype"]})
        return {"shape": shape, "dtype": dtype}
    # Otherwise produce a union
    return _reduce_to_union([a, b])


def _reduce_to_union(values: Sequence[Any]) -> Any:
    """Reduce a list of leaf/spec values into a compact union representation."""
    # Normalize dicts recursively
    if all(isinstance(v, Mapping) and not ({"shape", "dtype"} <= set(v.keys())) for v in values):
        # All are nested specs → deep-merge
        merged: dict[str, Any] = {}
        for v in values:
            for k, sub in v.items():
                if k in merged:
                    merged[k] = _merge_any(merged[k], sub)
                else:
                    merged[k] = sub
        return merged

    # Collapse to sorted unique reprs for readability
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) == 1:
        return uniq[0]
    return {"anyOf": uniq}


def _merge_any(a: Any, b: Any) -> Any:
    if (
        isinstance(a, Mapping)
        and isinstance(b, Mapping)
        and not (set(a.keys()) == {"shape", "dtype"} and set(b.keys()) == {"shape", "dtype"})
    ):
        # Deep merge for nested specs
        keys = set(a.keys()) | set(b.keys())
        out: dict[str, Any] = {}
        for k in keys:
            if k in a and k in b:
                out[k] = _merge_any(a[k], b[k])
            else:
                out[k] = a.get(k, b.get(k))
        return out
    return _merge_leaf(a, b)


def merge_specs(a: Spec | None, b: Spec) -> Spec:
    if a is None:
        return b
    return _merge_any(a, b)


class MyBuilder(generate.Builder):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name")
        self.root = kwargs.get("dir")
        self.VERSION = kwargs.get("version", "0.5.0")

        self.limit = kwargs.get("limit")


@dataclass
class Config:
    dir: Path  # Directory containing .dat files (searched recursively)
    name: str = "my_dataset"  # Dataset name
    workers: int = 16  # *.dat reader workers
    verbose: bool = False  # Print per-file details while streaming
    version: str = "0.5.0"  # Dataset version

    limit: int | None = None  # Optional limit on number of records to process
    by_step: bool = False  # Whether to build by step (True) or by episode (False)


def main(cfg: Config) -> None:
    if not cfg.dir.exists() or not cfg.dir.is_dir():
        raise SystemExit(f"Directory not found: {cfg.dir}")
    dat_files = sorted(cfg.dir.rglob("*.dat"))
    if not dat_files:
        raise SystemExit("No .dat files found.")

    builder = MyBuilder(**asdict(cfg))

    ds = arec.ArrayRecordBuilder(
        name=builder.name,
        root=str(Path("~/.cache/arrayrecords")),
        version=cfg.version,  # bump when schema/layout changes
        shard_size=1,  # records per shard
        writer_options="group_size:1",  # passed directly to ArrayRecordWriter
    )
    if cfg.by_step:
        ds.prepare(partial(arec.build_fn_per_step, fn=builder.build))
    else:
        ds.prepare(partial(arec.build_fn_per_episode, fn=builder.build))

    pprint(builder.spec(ds[0]))

    for x in tqdm(ds):
        pass

    print("done")


if __name__ == "__main__":
    main(tyro.cli(Config))
