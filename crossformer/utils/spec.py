from __future__ import annotations

from collections.abc import Callable
from functools import partial
import importlib
from typing import Any, Hashable, Iterable, TypedDict

import jax
from orbax import checkpoint as ocp

Spec = dict[Hashable, tuple[Iterable[int], Any]]


def spec(tree: dict[str, Any], simple=True) -> Spec:
    """Create a spec dictionary for the given tree structure."""

    sd = ocp.utils.to_shape_dtype_struct

    def toshape(x):
        return (x.shape, x.dtype) if getattr(x, "shape", None) else x

    return jax.tree.map(sd if not simple else toshape, tree)


def _norm_shape(s) -> tuple[int, ...]:
    # Accept list/tuple/np shape-like → tuple[int,...]
    return tuple(s) if s is not None else ()


def _norm_dtype(dt) -> str:
    # Works for strings, numpy/jax dtypes, and objects with .name or .__name__
    if dt is None:
        return "None"
    if hasattr(dt, "name"):  # numpy/jax dtype
        return str(dt.name)
    if hasattr(dt, "__name__"):  # Python types like int/float
        return dt.__name__
    return str(dt)


def diff(a: Spec, b: Spec, simple=True):
    """
    Compare two flat specs {key: (shape, dtype)} and report added/removed/changed.
    Returns:
        {
          "added":   {k: (shape, dtype)},
          "removed": {k: (shape, dtype)},
          "changed": {k: {"from": (shape, dtype), "to": (shape, dtype)}},
        }
    """
    keys_a, keys_b = set(a), set(b)

    added = {k: b[k] for k in (keys_b - keys_a)}
    removed = {k: a[k] for k in (keys_a - keys_b)}

    changed = {}
    for k in keys_a & keys_b:
        if simple:
            sa, da = a[k][0], a[k][1]
            sb, db = b[k][0], b[k][1]
        else:
            print(a)
            sa, da = a[k].shape, a[k].dtype
            sb, db = b[k].shape, b[k].dtype
        if _norm_shape(sa) != _norm_shape(sb) or _norm_dtype(da) != _norm_dtype(db):
            changed[k] = {"from": a[k], "to": b[k]}

    return {"added": added, "removed": removed, "changed": changed}


def ezdiff(a: dict[str, Any], b: dict[str, Any], simple=True):
    from crossformer.data.grain.utils import flat

    a, b = spec(flat(a), simple=simple), spec(flat(b), simple=simple)
    from rich.pretty import pprint

    pprint(diff(a, b, simple=simple))


class ModuleSpec(TypedDict):
    """A JSON-serializable representation of a function or class with some default args and kwargs to pass to
    it. Useful for specifying a particular class or function in a config file, while keeping it serializable
    and overridable from the command line using ml_collections.

    Usage:

        # Preferred way to create a spec:
        >>> from crossformer.model.components.transformer import Transformer
        >>> spec = ModuleSpec.create(Transformer, num_layers=3)
        # Same as above using the fully qualified import string:
        >>> spec = ModuleSpec.create("crossformer.model.components.transformer:Transformer", num_layers=3)

        # Usage:
        >>> ModuleSpec.instantiate(spec) == partial(Transformer, num_layers=3)
        # can pass additional kwargs at instantiation time
        >>> transformer = ModuleSpec.instantiate(spec, num_heads=8)

    Note: ModuleSpec is just an alias for a dictionary (that is strongly typed), not a real class. So from
    your code's perspective, it is just a dictionary.

    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    """

    module: str
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @staticmethod
    def create(callable_or_full_name: str | Callable, *args, **kwargs) -> ModuleSpec:  # type: ignore
        """Create a module spec from a callable or import string.

        Args:
            callable_or_full_name (str or object): Either the object itself or a fully qualified import string
                (e.g. "crossformer.model.components.transformer:Transformer")
        args (tuple, optional): Passed into callable upon instantiation.
        kwargs (dict, optional): Passed into callable upon instantiation.
        """
        if isinstance(callable_or_full_name, str):
            assert callable_or_full_name.count(":") == 1, (
                "If passing in a string, it must be a fully qualified import string "
                "(e.g. 'crossformer.model.components.transformer:Transformer')"
            )
            module, name = callable_or_full_name.split(":")
        else:
            module, name = _infer_full_name(callable_or_full_name)

        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)

    @staticmethod
    def instantiate(spec: ModuleSpec):  # type: ignore
        if set(spec.keys()) != {"module", "name", "args", "kwargs"}:
            raise ValueError(
                f"Expected ModuleSpec, but got {spec}. "
                "ModuleSpec must have keys 'module', 'name', 'args', and 'kwargs'."
            )
        cls = _import_from_string(spec["module"], spec["name"])
        return partial(cls, *spec["args"], **spec["kwargs"])

    @staticmethod
    def to_string(spec: ModuleSpec):  # type: ignore
        return (
            f"{spec['module']}:{spec['name']}"
            f"({', '.join(spec['args'])}"
            f"{', ' if spec['args'] and spec['kwargs'] else ''}"
            f"{', '.join(f'{k}={v}' for k, v in spec['kwargs'].items())})"
        )


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__name__"):
        return o.__module__, o.__name__
    else:
        raise ValueError(
            f"Could not infer identifier for {o}. "
            "Please pass in a fully qualified import string instead "
            "e.g. 'crossformer.model.components.transformer:Transformer'"
        )


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as e:
        raise ValueError(f"Could not import {module_string}:{name}") from e
