"""Reusable dataset geometry and deterministic sampling utilities."""

from .geometry import farthest_point_sample, mesh_from_record, sample_surface, triangulate
from .placement import place_parts, separated
from .random import stable_rng
from .transforms import (
    apply_pose,
    assembly_normalization,
    make_pose,
    pca_frame,
    pose_from_values,
    rotation_matrix,
    transform_points,
)

__all__ = [
    "apply_pose",
    "assembly_normalization",
    "farthest_point_sample",
    "make_pose",
    "mesh_from_record",
    "pca_frame",
    "place_parts",
    "pose_from_values",
    "rotation_matrix",
    "sample_surface",
    "separated",
    "stable_rng",
    "transform_points",
    "triangulate",
]
