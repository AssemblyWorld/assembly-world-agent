"""Per-sample evaluation inputs cached next to prepared configurations.

The cache holds everything scoring needs from the source side: the deterministic
1000-point clouds, ground-truth poses, the shared scale divisor and the resolved
equivalence groups. A hit makes evaluation independent of Hugging Face; a miss is
computed as before and written for the next evaluation. Entries are keyed by the
preparation identity, the initial episode checksum and the evaluation protocols,
and a key that differs is reported, never overwritten.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from ..artifacts import cache_directory, now, write_json
from ..models import PROTOCOL_VERSION, Pose
from ..similarity import SimilarityConfig, resolve_equivalence
from ..utils.scale import evaluation_scale

CACHE_VERSION = 1
CACHE_FILE = "evaluation.json"


def evaluation_cache_path(initial_path):
    directory = cache_directory(initial_path)
    return None if directory is None else directory / CACHE_FILE


def cache_key(sample_id, identity, expected, *, evaluation_protocol, similarity=None):
    """Everything the cached values depend on; ``similarity`` only affects equivalence."""
    similarity = similarity or SimilarityConfig()
    return dict(
        sample_id=sample_id,
        identity=identity,
        initial_sha256=expected["sha256"],
        parts=expected["parts"],
        preparation_protocol=PROTOCOL_VERSION,
        evaluation_protocol=evaluation_protocol,
        similarity=similarity.protocol(),
    )


def _code_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def compute_cache(sample, key, *, similarity=None):
    """Derive the cached evaluation inputs from a prepared sample."""
    similarity = similarity or SimilarityConfig()
    if similarity.protocol() != key["similarity"]:
        raise ValueError("Similarity configuration differs from the cache key")
    return cache_entry(sample, key, resolve_equivalence(sample, config=similarity))


def cache_entry(sample, key, equivalence):
    """Assemble a cache entry from a prepared sample and its resolved equivalence."""
    parts = sorted(sample.parts, key=lambda part: part.part_id)
    if len(parts) != key["parts"]:
        raise ValueError("Source part count differs from the cache key")
    if sample.sample_id != key["sample_id"] or sample.revision != key["identity"]["revision"]:
        raise ValueError("Prepared sample identity differs from the cache key")
    if (sample.config.surface_points, sample.config.fps_points) != (4096, 1000):
        raise ValueError("Evaluation requires the 4096/1000 preparation sampling protocol")
    if equivalence["protocol"] != key["similarity"]:
        raise ValueError("Equivalence protocol differs from the cache key")
    return dict(
        version=CACHE_VERSION,
        key=key,
        provenance=dict(created_at=now(), code_commit=_code_commit(), dataset=sample.dataset),
        part_ids=[part.part_id for part in parts],
        scale_divisor=float(evaluation_scale(sample)),
        points={part.part_id: np.asarray(part.points, dtype=float).tolist() for part in parts},
        gt_poses={
            part.part_id: [
                *np.asarray(part.gt_pose.position, dtype=float).tolist(),
                *np.asarray(part.gt_pose.quaternion, dtype=float).tolist(),
            ]
            for part in parts
        },
        equivalence=equivalence,
    )


class CacheKeyMismatch(ValueError):
    """The cached entry belongs to a different configuration, protocol or episode."""


def read_cache(path, key):
    """Return the cached entry matching ``key``, None when absent.

    A differing ``similarity`` protocol is a miss (another policy or threshold can be
    evaluated without disturbing the stored entry); any other difference is an error.
    """
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get("version") != CACHE_VERSION:
        raise CacheKeyMismatch(f"Unsupported evaluation cache version: {path}")
    stored = value["key"]
    if {k: v for k, v in stored.items() if k != "similarity"} != {
        k: v for k, v in key.items() if k != "similarity"
    }:
        raise CacheKeyMismatch(f"Evaluation cache key differs: {path}")
    if stored["similarity"] != key["similarity"]:
        return None
    if len(value["part_ids"]) != key["parts"]:
        raise CacheKeyMismatch(f"Evaluation cache part count differs: {path}")
    return value


def write_cache(path, value):
    """Write once; an existing entry is left untouched (another worker may own it)."""
    path = Path(path)
    if path.exists():
        return False
    write_json(path, value)
    return True


def cached_inputs(value):
    """Unpack a cache entry into the arrays scoring consumes."""
    ids = list(value["part_ids"])
    if sorted(ids) != ids or len(set(ids)) != len(ids):
        raise ValueError("Cached part IDs must be sorted and unique")
    points, poses = [], []
    for pid in ids:
        cloud = np.asarray(value["points"][pid], dtype=float)
        if cloud.shape != (1000, 3) or not np.isfinite(cloud).all():
            raise ValueError(f"Cached point cloud is invalid: {pid}")
        pose = np.asarray(value["gt_poses"][pid], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"Cached ground-truth pose is invalid: {pid}")
        points.append(cloud)
        poses.append(Pose(pose[:3].copy(), pose[3:].copy()))
    divisor = float(value["scale_divisor"])
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Cached scale divisor is invalid")
    equivalence = value["equivalence"]
    if sorted(pid for group in equivalence["groups"] for pid in group) != ids:
        raise ValueError("Cached equivalence groups do not partition the part IDs")
    return dict(
        part_ids=ids, points=points, gt_poses=poses, divisor=divisor, equivalence=equivalence
    )
