"""Stable RNG streams independent of iteration order and Python hash salts."""

import hashlib
import json

import numpy as np


def stable_rng(seed: int, *identity: str) -> np.random.Generator:
    payload = json.dumps([int(seed), *identity], ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
