"""Grounded planar placement with conservative full-mesh separation."""

import numpy as np

from ..models import Pose
from .random import stable_rng
from .transforms import make_pose


def separated(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    """Whether two (min, max) XY boxes have the required gap on at least one axis."""
    return bool(np.any(a[1] + gap <= b[0]) or np.any(b[1] + gap <= a[0]))


def place_parts(
    vertices: dict[str, np.ndarray],
    *,
    dataset: str,
    sample_id: str,
    seed: int,
    gap: float,
    attempts: int = 128,
    expansions: int = 5,
) -> dict[str, Pose]:
    """Random yaw and rejection-sampled XY positions; bounded grid fallback."""
    if not vertices or not np.isfinite(gap) or gap <= 0:
        raise ValueError("Placement needs parts and a finite positive gap")
    ids = sorted(vertices)
    rotated, rotations, bounds = {}, {}, {}
    for part_id in ids:
        rng = stable_rng(seed, dataset, sample_id, part_id, "yaw")
        yaw = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = vertices[part_id] @ rotation.T
        rotated[part_id], rotations[part_id] = points, rotation
        bounds[part_id] = np.array([points[:, :2].min(0), points[:, :2].max(0)])
    rng = stable_rng(seed, dataset, sample_id, "placement")
    order = [ids[i] for i in rng.permutation(len(ids))]
    max_extent = max(float(np.max(b[1] - b[0])) for b in bounds.values())
    side = (max_extent + gap) * np.ceil(np.sqrt(len(ids)))
    positions = {}
    for level in range(expansions):
        boxes = []
        positions = {}
        for part_id in order:
            for _ in range(attempts):
                xy = rng.uniform(-side / 2, side / 2, size=2)
                box = bounds[part_id] + xy
                if all(separated(box, other, gap) for other in boxes):
                    positions[part_id] = xy
                    boxes.append(box)
                    break
            else:
                break
        if len(positions) == len(ids):
            break
        side *= 1.5
    if len(positions) != len(ids):
        columns = int(np.ceil(np.sqrt(len(ids))))
        # A small numerical margin keeps >= gap true after floating arithmetic.
        spacing = max_extent + gap + 1e-12
        for slot, part_id in enumerate(order):
            center = np.array([slot % columns, slot // columns]) * spacing
            positions[part_id] = center - bounds[part_id].mean(0)
    boxes = [bounds[pid] + positions[pid] for pid in ids]
    center = (np.min([b[0] for b in boxes], axis=0) + np.max([b[1] for b in boxes], axis=0)) / 2
    result = {}
    for part_id in ids:
        xy = positions[part_id] - center
        result[part_id] = make_pose(np.r_[xy, -rotated[part_id][:, 2].min()], rotations[part_id])
    final_boxes = [bounds[pid] + result[pid].position[:2] for pid in ids]
    for i, box in enumerate(final_boxes):
        if any(not separated(box, other, gap - 1e-12) for other in final_boxes[:i]):
            raise ValueError("Placement failed the final non-overlap check")
    return result
