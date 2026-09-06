"""In-memory source and prepared samples; these are not episode wire formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PROTOCOL_VERSION = "assembly-preparation-v1"


@dataclass(frozen=True)
class Pose:
    """Rigid local-to-world pose, with a scalar-first unit quaternion."""

    position: FloatArray
    quaternion: FloatArray


@dataclass(frozen=True)
class Mesh:
    vertices: FloatArray
    faces: tuple[tuple[int, ...], ...]
    normals: FloatArray = field(default_factory=lambda: np.empty((0, 3)))
    face_normal_indices: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class SourcePart:
    part_id: str
    mesh: Mesh
    assembled_pose: Pose
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SourceSample:
    dataset: str
    sample_id: str
    revision: str
    parts: tuple[SourcePart, ...]
    source_to_z_up: FloatArray
    metadata: dict[str, Any]
    source_splits: Any
    manual_pages: tuple[dict[str, Any], ...]
    manual: Any
    steps: tuple[dict[str, Any], ...]
    annotations: dict[str, Any]


@dataclass(frozen=True)
class PreparationConfig:
    surface_points: int = 4096
    fps_points: int = 1000
    sampling_seed: int = 0
    initialization_seed: int = 0
    min_gap: float = 0.02

    def __post_init__(self) -> None:
        for key in ("surface_points", "fps_points", "sampling_seed", "initialization_seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 0 < self.fps_points <= self.surface_points:
            raise ValueError("Require 0 < fps_points <= surface_points")
        if not np.isfinite(self.min_gap) or self.min_gap <= 0:
            raise ValueError("min_gap must be finite and positive")


@dataclass(frozen=True)
class AssemblyPart:
    part_id: str
    mesh: Mesh
    points: FloatArray
    initial_pose: Pose
    gt_pose: Pose
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AssemblySample:
    dataset: str
    sample_id: str
    revision: str
    parts: tuple[AssemblyPart, ...]
    source_to_world: FloatArray
    world_to_source: FloatArray
    config: PreparationConfig
    metadata: dict[str, Any]
    source_splits: Any
    manual_pages: tuple[dict[str, Any], ...]
    manual: Any
    steps: tuple[dict[str, Any], ...]
    annotations: dict[str, Any]
    protocol_version: str = PROTOCOL_VERSION
