from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
from jax._src import config as jax_internal_config
import jax.numpy as jnp

# Flax expects newer JAX builds that expose define_bool_state; alias to the
# modern equivalent when running tests on newer releases.
if not hasattr(jax.config, "define_bool_state"):
    fallback = getattr(jax_internal_config, "bool_state", None)
    if fallback is None:
        fallback = jax_internal_config.define_bool_state
    jax.config.define_bool_state = fallback

# Recent JAX versions removed jax.experimental.maps, but Flax still imports it.
if "jax.experimental.maps" not in sys.modules:
    maps_module = types.ModuleType("jax.experimental.maps")
    empty_mesh = types.SimpleNamespace(devices=types.SimpleNamespace(shape=()))
    maps_module.thread_resources = types.SimpleNamespace(env=types.SimpleNamespace(physical_mesh=empty_mesh))
    sys.modules["jax.experimental.maps"] = maps_module
    if hasattr(jax, "experimental"):
        setattr(jax.experimental, "maps", maps_module)

if not hasattr(jax.nn, "normalize"):

    def _normalize(x, axis=-1, epsilon=1e-12):
        norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + epsilon)

    jax.nn.normalize = _normalize
