from __future__ import annotations

from functools import wraps
import logging
import os
from typing import Any, Sequence

import jax
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
import numpy as np

from crossformer.utils.mytyping import PyTree

cpu = jax.devices("cpu")[0]


class JaxCPUProxy:
    """A proxy for jax.numpy that forces all operations to run on CPU.
    it is up to the user to know when to use
    """

    def __getattr__(self, name):
        fn = getattr(jnp, name)
        if callable(fn):

            @wraps(fn)
            def wrapper(*args, **kwargs):
                out = fn(*args, **kwargs, device=cpu)
                return out

            return wrapper
        return fn


def with_device_context(device):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with jax.default_device(device):
                return fn(*args, **kwargs)

        return wrapped

    return deco


with_device = with_device_context


def viz(tree: PyTree):
    jax.debug.visualize_array_sharding(tree)


def str2jax(s: str, device=cpu) -> jax.Array:
    """Convert a string to a JAX array by encoding each character as its Unicode code point."""
    return jnp.array(np.frombuffer(s.encode("utf-8"), dtype=np.uint8), device=device)


def npstr2jax(arr: np.ndarray, device=cpu) -> jax.Array:
    if arr.dtype.kind != "U":
        raise TypeError("expected Unicode array")
    # Encode each string to UTF-8 bytes, then pack
    encoded = [s.encode("utf-8") for s in arr]
    # Pad to same length if needed
    maxlen = max(len(e) for e in encoded)
    padded = np.array([np.frombuffer(e.ljust(maxlen, b"\0"), dtype=np.uint8) for e in encoded])
    return jnp.array(padded, device=device)


def jax2str(x: jax.Array) -> str:
    """Convert a JAX array back to a string by decoding each Unicode code point."""
    return np.array(x.astype(jnp.uint8)).tobytes().decode("utf-8")


def jpad_str(x: jax.Array, length: int) -> str:
    """Pad a jax uint string to a fixed length with 0s."""
    x = jnp.array(x.astype(jnp.uint8))
    shape = (*x.shape[:-1], length)
    padded = jnp.zeros(shape, dtype=jnp.uint8)
    return padded + x


def host_broadcast_str(x: str) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    multihost_utils.assert_equal(len(x), f"String lengths are not equal: got {len(x)} for {jax.process_index()}")
    encoded = np.array([ord(c) for c in x], dtype=np.uint8)
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    return "".join([chr(u) for u in encoded])


def shard_along_axis(x: Any, devices: Sequence[jax.Device], axis: int = 0) -> jax.Array:
    """Shard a PyTree of arrays along a given axis, putting them on device in
    the process. Works in multi-host setting as long as PyTrees are equal on all
    hosts."""
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )
    x = jax.tree.map(jnp.array, x)
    return jax.tree.map(
        lambda arr: jax.make_array_from_callback(arr.shape, sharding, lambda index: arr[index]),
        x,
    )


def merge_along_axis(x: Any, axis: int = 0) -> jax.Array:
    """Convert a PyTree of host-local arrays to a global array, concatenating and sharding along
    `axis`."""
    return multihost_utils.host_local_array_to_global_array(
        x,
        jax.sharding.Mesh(jax.devices(), "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )


def split_along_axis(x: Any, axis: int = 0) -> jax.Array:
    """Convert a PyTree of global arrays to a host-local array, splitting along `axis`."""
    return multihost_utils.global_array_to_host_local_array(
        x,
        jax.sharding.Mesh(jax.devices(), "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )


def replicate(x: Any, devices: Sequence[jax.Device] | None = None) -> jax.Array:
    """Replicate a PyTree of arrays across devices. Works in multi-host setting
    as long as PyTrees are equal on all hosts."""
    if devices is None:
        devices = jax.devices()
    sharding = jax.sharding.PositionalSharding(devices).replicate()
    x = jax.tree.map(jnp.array, x)
    return jax.tree.map(
        lambda arr: jax.make_array_from_callback(arr.shape, sharding, lambda index: arr[index]),
        x,
    )


def initialize_compilation_cache(
    cache_dir=os.path.expanduser("~/.jax_compilation_cache"),
):
    """Initializes the Jax persistent compilation cache."""
    compilation_cache.initialize_cache(cache_dir)
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.addFilter(lambda record: "Not writing persistent cache entry for" not in record.getMessage())
