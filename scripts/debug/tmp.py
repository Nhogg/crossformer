from __future__ import annotations

import grain
from rich import print

sa = grain.MapDataset.range(10).map(lambda x: x * 2)
sb = grain.MapDataset.range(10).map(lambda x: x * 100)
ds = grain.MapDataset.mix([sa, sb], weights=[1.0, 1.0])

it = iter(ds)
for _ in range(len(ds)):
    print(next(it))

print("len", len(ds))
