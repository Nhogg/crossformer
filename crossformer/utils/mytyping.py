from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias

import jax
from jax.typing import ArrayLike  # noqa
from jaxtyping import (  # noqa
    Array,
    Float,
    jaxtyped,
)
from typeguard import typechecked  # noqa

PRNGKey: TypeAlias = jax.Array  # keys are just arrays: jax.random.PRNGKey
PyTree: TypeAlias = jax.typing.ArrayLike | Mapping[str, "PyTree"]
Config: TypeAlias = Any | Mapping[str, "Config"]
Params: TypeAlias = Mapping[str, PyTree]
Data: TypeAlias = Mapping[str, PyTree]
Shape: TypeAlias = Sequence[int]
Dtype: TypeAlias = jax.typing.DTypeLike


class DeprecatedError(Exception):
    pass
