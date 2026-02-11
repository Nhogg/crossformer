from __future__ import annotations

import numpy as np


def acroll(arr, shift: int, ax: int = 0):
    """
    acyclic roll of an array along a given axis.
    Fills the emptied positions with edge values.
    """

    assert len(arr) > np.abs(shift), (
        f"Shift magnitude {np.abs(shift)} must be less than array size along the specified axis {len(arr)}."
    )
    k = abs(shift)
    if shift == 0:
        return arr.copy()

    sl = [slice(None)] * arr.ndim
    pad = [(0, 0)] * arr.ndim

    if shift > 0:
        sl[ax] = slice(0, -k)
        pad[ax] = (k, 0)
    else:
        sl[ax] = slice(k, None)
        pad[ax] = (0, k)

    return np.pad(arr[tuple(sl)], pad, mode="edge")


def acroll_stacked(arr, shift: int, ax: int = 0, offset: int | None = None):
    """
    acyclic roll of an array along a given axis for stacked data.
    Fills the emptied positions with edge values.
    builds stacked matrix of shape (n, abs(shift)) and takes min along axis 1
        offset: number of initial elements to skip before applying acroll
    """

    offset = np.sign(shift) if offset is None else offset
    n = arr.shape[ax]

    i = np.arange(n)[:, None]  # (n, 1)
    j = np.arange(np.abs(shift))[None, :]  # (1, shift)
    j = j * np.sign(shift)

    idx = np.minimum(i + j + offset, n - 1)  # clamp
    idx = np.maximum(idx, 0)  # clamp
    return np.take(arr, idx, axis=ax)


def tile(x: dict, key: str, shape):
    if isinstance(shape, str):
        shape = (len(x[shape]), 1)

    x[key] = np.tile(x[key], shape)
    return x


def add_step_id(x: dict):
    """adds step ids into the dict based on episode ids."""
    n = len(x["episode_id"])
    x["step_id"] = np.arange(n)
    return x


def to_unified_structure(x: dict) -> dict:
    c = 20
    sid, eid = x["step_id"], x["episode_id"]
    n = len(sid)
    chunk = acroll_stacked(sid, c)

    return {
        "info": {
            "id": {"episode": x["episode_id"], "step": x["step_id"], "len": np.full((n,), n)},
        },
        "action": {
            "k3ds": x["k3ds"][chunk],
        },
        "observation": {
            "proprio": {
                "k3ds": x["k3ds"],
            },
            "image": {
                "low": x["low"],
                "over": x["over"],
                "side": x["side"],
            },
        },
    }


def test_acroll():
    a = np.arange(10)
    print(a)
    b = acroll(a, -5)
    print(b)

    c = acroll_stacked(a, 5)
    print(c)
    print(acroll_stacked(a, 1).reshape(-1))


if __name__ == "__main__":
    test_acroll()
