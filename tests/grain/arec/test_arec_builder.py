from __future__ import annotations

import json
from pathlib import Path

from crossformer.data.grain.arec.arec import ArrayRecordBuilder


def test_prepare_empty_stream(tmp_path):
    builder = ArrayRecordBuilder(
        name="empty_ds",
        root=str(tmp_path),
        version="v0",
        shard_size=2,
    )

    builder.prepare(lambda: iter(()))

    meta = builder.meta
    assert meta["num_records"] == 0
    assert meta["name"] == "empty_ds"
    assert meta["version"] == "v0"

    spec_path = Path(tmp_path) / "empty_ds" / "v0" / "main" / "spec.json"
    assert not spec_path.exists()

    meta_path = Path(tmp_path) / "empty_ds" / "v0" / "main" / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        stored_meta = json.load(f)
    assert stored_meta["num_records"] == 0
