"""Configurable equivalence of input part shapes, independent of assembly scoring."""

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from .utils import pca_frame
from .utils.registration import IMPROVEMENT_TOLERANCE, MAX_ITERATIONS, align, transform
from .utils.scale import evaluation_scale


@dataclass(frozen=True)
class SimilarityConfig:
    policy: str = "geometry"
    threshold: float = 1e-4

    def __post_init__(self):
        if self.policy not in {"source", "geometry"}:
            raise ValueError("Similarity policy must be source or geometry")
        if (
            isinstance(self.threshold, bool)
            or not np.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise ValueError("Similarity threshold must be finite and nonnegative")

    def protocol(self):
        return {
            **asdict(self),
            "threshold": self.threshold if self.policy == "geometry" else None,
            "distance": "sum of bidirectional mean squared nearest-neighbor distances",
            "scale": "shared largest part vertex-PCA AABB diagonal = 1",
            "sampling": "4096 area-weighted candidates, 1000 FPS; preparation sampling seed",
            "alignment": "24 proper PCA starts; symmetric ICP; SE(3) only; no pose seeds",
            "max_iterations": MAX_ITERATIONS,
            "improvement_tolerance": IMPROVEMENT_TOLERANCE,
            "grouping": "connected components of CD <= threshold"
            if self.policy == "geometry"
            else "source atomic relations",
        }


def similarity_clouds(sample):
    """Canonical input clouds; no prediction, target pose or assembly relation is read."""
    if (sample.config.surface_points, sample.config.fps_points) != (4096, 1000):
        raise ValueError("Similarity requires 4096/1000 sampling")
    divisor = evaluation_scale(sample)
    clouds = []
    for part in sorted(sample.parts, key=lambda p: p.part_id):
        if part.points.shape != (1000, 3) or not np.isfinite(part.points).all():
            raise ValueError("Similarity requires finite prepared 1000-point clouds")
        center, basis = pca_frame(part.mesh.vertices)
        clouds.append((part.points - center) @ basis / divisor)
    return clouds, divisor


def groups_from_distances(ids, distances, threshold):
    """Deterministic transitive closure, including diagnostics for non-edge pairs."""
    distances = np.asarray(distances, dtype=float)
    if (
        distances.shape != (len(ids), len(ids))
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
        or not np.allclose(distances, distances.T)
    ):
        raise ValueError("Expected finite nonnegative symmetric distance matrix")
    parent = list(range(len(ids)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in combinations(range(len(ids)), 2):
        if distances[i, j] <= threshold:
            a, b = root(i), root(j)
            parent[max(a, b)] = min(a, b)
    components = {}
    for i in range(len(ids)):
        components.setdefault(root(i), []).append(i)
    groups, diagnostics = [], []
    for indices in components.values():
        group = [ids[i] for i in indices]
        pairs = [(float(distances[i, j]), ids[i], ids[j]) for i, j in combinations(indices, 2)]
        worst = max(pairs, default=(0.0, None, None))
        groups.append(group)
        diagnostics.append(
            dict(
                parts=group,
                max_chamfer=worst[0],
                worst_pair=list(worst[1:]) if pairs else None,
                above_threshold_pairs=[
                    dict(part_ids=[a, b], chamfer=v) for v, a, b in pairs if v > threshold
                ],
                above_threshold_pair_count=sum(v > threshold for v, a, b in pairs),
            )
        )
    return groups, diagnostics


def inspect_similarity_pair(sample, part_id, target_part_id):
    """Return transient pair clouds and rigid transform for notebook inspection."""
    ids = sorted(p.part_id for p in sample.parts)
    clouds, divisor = similarity_clouds(sample)
    a, b = sorted((ids.index(part_id), ids.index(target_part_id)))
    # Keep the same fixed direction as resolve_equivalence and the protocol smoke.
    rotation, translation, info = align(clouds[b], clouds[a])
    return dict(
        part_ids=[ids[b], ids[a]],
        prediction=clouds[b],
        target=clouds[a],
        aligned=transform(clouds[b], rotation, translation),
        alignment=info,
        scale_divisor=divisor,
    )


def resolve_equivalence(sample, *, config=None):
    """Resolve a prepared sample into stable part-ID groups, on demand only."""
    config = config or SimilarityConfig()
    ids = sorted(p.part_id for p in sample.parts)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected unique nonempty part IDs")
    source = sample.source_equivalence
    base = dict(
        policy=config.policy,
        protocol=config.protocol(),
        part_ids=ids,
        source_annotation_missing=source.get("missing", True),
        ignored_composite_self_groups=source.get("ignored_composite_self_groups", []),
    )
    if config.policy == "source":
        groups = source.get("groups", [[pid] for pid in ids])
        if sorted(pid for group in groups for pid in group) != ids or any(not g for g in groups):
            raise ValueError("Source equivalence must partition actual part IDs")
        return dict(**base, groups=sorted(sorted(g) for g in groups), group_diagnostics=[])
    clouds, divisor = similarity_clouds(sample)
    distances = np.zeros((len(ids), len(ids)))
    registrations = []
    for i in range(len(ids)):
        for j in range(i):
            _, _, info = align(clouds[i], clouds[j])
            distances[i, j] = distances[j, i] = info["chamfer"]
            registrations.append(
                dict(
                    part_ids=[ids[i], ids[j]],
                    chamfer=info["chamfer"],
                    seed=info["seed"],
                    iterations=info["iterations"],
                    converged=info["converged"],
                )
            )
    groups, diagnostics = groups_from_distances(ids, distances, config.threshold)
    return dict(
        **base,
        groups=groups,
        group_diagnostics=diagnostics,
        distances=distances.tolist(),
        registrations=registrations,
        scale_divisor=divisor,
    )


def equivalence_metadata(result):
    """Compact reproducibility and chaining diagnostics for persisted score rows."""
    return {k: v for k, v in result.items() if k not in {"distances", "registrations"}}
