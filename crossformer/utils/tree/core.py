from __future__ import annotations

from collections.abc import Callable
import fnmatch

import flax.traverse_util as ftu
import jax


def flat(tree):
    return {".".join(k): v for k, v in ftu.flatten_dict(tree).items()}


def unflat(tree):
    return ftu.unflatten_dict({tuple(k.split(".")): v for k, v in tree.items()})


def merge(lhs, rhs):
    lhs, rhs = flat(lhs), flat(rhs)
    out = lhs | rhs
    return unflat(out)


def drop(tree: dict, keys: list[str]) -> dict:
    return {k: v for k, v in tree.items() if k not in keys}


def do_fn_key(x: dict, keymatch: str, fn: Callable):
    """apply fn to keys that match keymatch. glob pattern"""
    matches = fnmatch.filter(flat(x).keys(), keymatch)
    y = {k: item for k, item in flat(x).items() if k in matches}
    if not y:
        raise KeyError(f"No keys match {keymatch}")

    return merge(x, unflat(jax.tree.map(fn, y)))
    # y = jax.tree.map(fn, y)
    # x = flat(x) | y
    # x = unflat(x)
    # return x
