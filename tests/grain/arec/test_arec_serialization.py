from __future__ import annotations

import numpy as np
import pytest

from crossformer.data.arec.arec import _schema_fingerprint, pack_record, unpack_record


def _assert_nested_equal(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
        return

    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return

    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right)
        for l_item, r_item in zip(list(left), list(right)):
            _assert_nested_equal(l_item, r_item)
        return

    assert left == right


@pytest.mark.parametrize(
    "payload",
    [
        {
            "meta": {
                "tokens": np.arange(6, dtype=np.int32).reshape(2, 3),
                "mask": np.array([[True, False, True]], dtype=bool),
            },
            "summary": (
                {
                    "embedding": np.linspace(0.0, 1.0, num=4, dtype=np.float32),
                    "id": np.int64(42),
                },
                {
                    "embedding": np.linspace(-1.0, 1.0, num=4, dtype=np.float64),
                    "payload": b"\x00\xff\x7f",
                },
            ),
            "attributes": [
                {"name": "alpha", "score": np.float16(0.5)},
                {"name": "beta", "score": 1},
            ],
        },
    ],
)
def test_pack_unpack_roundtrip(payload):
    encoded = pack_record(payload)
    decoded = unpack_record(encoded)
    _assert_nested_equal(payload, decoded)


def test_schema_fingerprint_changes_when_version_differs():
    build_meta = {"a": 1, "b": "two"}
    baseline = _schema_fingerprint("v1", build_meta)
    updated = _schema_fingerprint("v2", build_meta)
    assert updated != baseline


@pytest.mark.parametrize(
    "initial_meta, mutated_meta",
    [
        ({"a": 1}, {"a": 2}),
        ({"nested": {"x": 1}}, {"nested": {"x": 1, "y": 0}}),
        ({"flags": [True, False]}, {"flags": [True, True]}),
    ],
)
def test_schema_fingerprint_changes_when_build_meta_differs(initial_meta, mutated_meta):
    version = "v3"
    baseline = _schema_fingerprint(version, initial_meta)
    updated = _schema_fingerprint(version, mutated_meta)
    assert updated != baseline


@pytest.mark.parametrize(
    "build_meta_a, build_meta_b",
    [
        ({"shards": 4, "tokenizer": "v1"}, {"tokenizer": "v1", "shards": 4}),
        ({"nested": {"foo": 1, "bar": 2}}, {"nested": {"bar": 2, "foo": 1}}),
    ],
)
def test_schema_fingerprint_is_order_invariant(build_meta_a, build_meta_b):
    fp_a = _schema_fingerprint("v3", build_meta_a)
    fp_b = _schema_fingerprint("v3", build_meta_b)
    assert fp_a == fp_b
