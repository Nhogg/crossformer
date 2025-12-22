from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

from grain._src.core import transforms
from grain._src.python.dataset import dataset
import grain.experimental as ge
import jax
import numpy as np

from crossformer.data.grain.utils import traj_len
from crossformer.utils.jax_utils import cpu, with_device

T = TypeVar("T")  # flat element type after transform
S = TypeVar("S")  # parent element type


class PrivilegedFlatMapMapDataset(dataset.MapDataset[T], Generic[T, S]):
    """Flatmap using known lengths_dict, permanent RAM cache for parent + splits."""

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        parent: dataset.MapDataset[S],
        transform: transforms.FlatMapTransform,
        lengths_dict: dict[int, int],
        *,
        preserve_insertion_order: bool = True,
    ):
        super().__init__(parent)
        self._transform = transform
        lengths_dict = {int(k): v for k, v in lengths_dict.items()}

        # element ordering
        if preserve_insertion_order:
            eids = list(lengths_dict.keys())
        else:
            try:
                eids = sorted(lengths_dict.keys(), key=int)
            except Exception:
                eids = sorted(lengths_dict.keys())

        self._elem_indices = [int(e) for e in eids]
        self._lens = [int(lengths_dict[e]) for e in eids]

        self._starts = []
        acc = 0
        for L in self._lens:
            self._starts.append(acc)
            acc += L
        self._total_len = acc

        # permanent caches
        self._elem_cache: dict[int, S] = {}
        self._splits_cache: dict[int, Sequence[T]] = {}

    def __len__(self) -> int:
        return self._total_len

    def __str__(self) -> str:
        return f"PrivilegedFlatMapMapDataset(total={self._total_len})"

    def _locate(self, flat_index: int):
        from bisect import bisect_right

        k = bisect_right(self._starts, flat_index) - 1
        if k < 0 or k >= len(self._starts):
            raise IndexError(flat_index)
        return k, flat_index - self._starts[k]

    def __getitem__(self, index: int) -> T:
        if isinstance(index, slice):
            return self.slice(index)
        if index < 0 or index >= self._total_len:
            raise IndexError(index)

        with self._stats.record_self_time():
            k, split_idx = self._locate(index)
            parent_idx = self._elem_indices[k]

            # permanent parent cache
            if k not in self._elem_cache and k not in self._splits_cache:
                elem = self._parent[k]
                items = self._transform.flat_map(elem)
                self._splits_cache[k] = {i: items[i] for i in range(len(items))}

            # n = sum([len(items) for items in self._splits_cache.values()])
            out = self._splits_cache[k][split_idx]
            return self._stats.record_output_spec(out)


@dataclass
class UnpackFlatMap(ge.FlatMapTransform):
    max_fan_out: int = 1500
    key: str = "info.id.step_id"  # key to use to determine trajectory length
    use_np: bool = False

    @with_device(cpu)
    def flat_map(self, element) -> Sequence[Any]:
        """splits a single element."""
        n = int(traj_len(element, self.key))
        # why do we have to materialize on cpu? jax makes copy
        element = jax.tree.map(lambda x: jax.device_put(x, cpu), element)
        # iscpu = jax.tree.map(lambda x: x.platform() == 'cpu' , element)
        # iscpu_all = jax.tree.reduce(lambda x, y: x and y, iscpu)
        # pprint(iscpu)
        if self.use_np:
            _e = jax.tree.map(np.array, element)
            return [jax.tree.map(lambda x: np.array(x[i]), _e) for i in range(n)]

        with jax.default_device(cpu):
            return [jax.tree.map(lambda x: jax.device_put(x[i], cpu), element) for i in range(n)]


@dataclass
class ShuffleUnpackFlatMap(ge.FlatMapTransform):
    max_fan_out: int = 1500
    key: str = "info.id.step_id"

    def flat_map(self, element) -> Sequence[Any]:
        """splits a single element."""
        raise NotImplementedError("ShuffleUnpackFlatMap is not implemented yet.")
        n = traj_len(element, self.key)
        return [jax.tree.map(lambda x: x[i], element) for i in range(n)]
