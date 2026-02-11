"""Utility helpers for the Grain data pipeline.

The TensorFlow based pipeline under :mod:`crossformer.data.dataset` relies on
TensorFlow primitives for a large collection of small utilities such as
recursive tree mapping, padding helpers, or dictionary merges.  The Grain based
pipeline needs the same functionality but implemented using NumPy/JAX friendly
operations so that it remains completely framework agnostic.

Only a very small subset of helpers are required for the initial Grain
implementation and they are intentionally kept lightweight.  They operate on
standard Python containers (``dict``/``list``) and NumPy/JAX arrays so they can
be freely reused both inside tests and production code without pulling in
TensorFlow as a dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from crossformer.utils.deco import deprecate
from crossformer.utils.tree import flat

Tree = Mapping[str, Any] | dict[str, Any]


@deprecate("use info.length", strict=True)
def traj_len(traj: dict, flatkey=None) -> int:
    x = flat(traj)
    k = flatkey if flatkey else next(iter(x.keys()))
    n = len(x[k])
    return n


def is_padding(value: Any) -> np.ndarray:
    """Returns a boolean mask indicating which entries correspond to padding.

    The heuristic mirrors the TensorFlow implementation:

    * numerical arrays are considered padding if every element is ``0``;
    * byte/Unicode arrays are padding if they are empty strings;
    * boolean arrays use ``False`` as the padding sentinel;
    * nested dictionaries are processed recursively with logical ``and``.

    The return type is always a boolean NumPy array matching the input shape.
    """

    if value.dtype.kind in {"U", "S"}:
        return value == ""
    return 0


def to_padding(value: Any) -> Any:
    """Creates a padding value with the same shape/dtype as ``value``."""

    value = np.asarray(value)
    if value.dtype.kind in {"U", "S"}:
        return np.zeros_like(value, dtype=value.dtype)
    if value.dtype == np.bool_:
        return np.zeros_like(value, dtype=bool)
    return np.zeros_like(value)
