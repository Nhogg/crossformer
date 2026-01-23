from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, cast, Mapping, TypeAlias, TypeVar

from beartype import beartype
from beartype.roar import BeartypeException
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Array, Float, jaxtyped, TypeCheckError
import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def jtyped(fn: F) -> F:
    """
    Decorator: applies @jaxtyped(typechecker=beartype) and, on failure,
    prints/raises only the final compact message (no traceback tree).

    Catches:
      - jaxtyping.TypeCheckError
      - any beartype.roar.BeartypeException (in case it escapes)
    """
    checked = jaxtyped(typechecker=beartype)(fn)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return checked(*args, **kwargs)
        except (TypeCheckError, BeartypeException) as e:
            # "Final error" = stringified exception; suppress chained context.
            raise TypeCheckError(str(e)) from None

    return cast(F, wrapper)


class _StackedDim:
    dim: str

    def __class_getitem__(cls, item):
        if hasattr(item, "leaftype"):
            return jt.PyTree[jt.Shaped[item.leaftype, cls.dim]]
        return jt.Shaped[item, cls.dim]


class Stacked:
    def __class_getitem__(cls, dim: str):
        return type(f"Stacked[{dim!r}]", (_StackedDim,), {"dim": dim})


# shape anno
Batched = Stacked["batch"]
Windowed = Stacked["win"]
Chunked = Stacked["chunk"]

# basic types
Image: TypeAlias = jt.Float[jt.Array, "H W C"]
Proprio: TypeAlias = jt.Float[jt.Array, "prop"]
Observation: TypeAlias = Image | Proprio

# only Actions are chunked
Action: TypeAlias = jt.Float[jt.Array, "act"]

Mask: TypeAlias = jt.Bool[jt.Array, "1"]

"""
def full(size: int, fill: float) -> Float[jax.Array, "{size}"]:
    return jax.numpy.full((size,), fill)

class SomeClass:
    some_value = 5

    def full(self, fill: float) -> Float[jax.Array, "{self.some_value}+3"]:
"""

from dataclasses import fields
from typing import Iterator


@dataclass
class DataMap(Mapping[str, Any]):
    def __iter__(self) -> Iterator[str]:
        for f in fields(self):
            yield f.name

    def __len__(self) -> int:
        return len(fields(self))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class Step(DataMap):  # total=False makes extra keys allowed
    observation: jt.PyTree[Windowed[Observation]]
    action: jt.PyTree[Windowed[Chunked[Action]]]

    action_head_masks: Any
    action_pad_mask: Any
    task: Any

    meta: dict | None = None

    def __post_init__(self):
        raise NotImplementedError("not working yet")
        # TODO /oc
        # we cannot validate the keys


@dataclass
class Batch(Step):
    observation: jt.PyTree[Batched[Windowed[Observation]]]
    action: jt.PyTree[Batched[Windowed[Chunked[Action]]]]


# ------------------------
# beartype demo (runtime)
# ------------------------
@beartype
def join(words: list[str]) -> str:
    return ",".join(words)


# ------------------------
# jaxtyped demo (runtime)
# ------------------------
# @jaxtyped(typechecker=beartype)
@jtyped
def add(x: Float[Array, "B T"], y: Float[Array, "B T"]) -> Float[Array, "B T"]:
    return x + y


def main():
    img: Image = jnp.zeros((64, 64, 3))
    wimg: ImageWindow = jnp.stack([img, img, img], axis=0)  # window of 3 images

    @jtyped
    def add_img(a: Image, b: Image) -> Image:
        return a + b

    add_img(img, img)  # OK
    add_img(img, wimg)  # OK

    # print(join(["ok", "this", "is", "fine"]))        # OK
    # print(join([b"ok", b"this", b"is", b"fine"]))    # ❌ beartype error

    a = jnp.zeros((2, 3), dtype=np.float32)
    b = jnp.ones((2, 3), dtype=np.float32)
    c = jnp.ones((2, 4), dtype=np.float32)

    print(add(a, b))  # OK
    # print(add(a, c))  # ❌ jaxtyped error: axis 'T' mismatch


if __name__ == "__main__":
    main()
