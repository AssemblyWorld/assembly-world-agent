"""Explicit rigid and similarity transforms using column-vector matrices."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import Pose


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def make_pose(position: np.ndarray, rotation: np.ndarray) -> Pose:
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Expected a finite position with shape (3,)")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Expected a finite rotation with shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-10, rtol=0
    ):
        raise ValueError("Rotation must be orthonormal and right-handed")
    q = Rotation.from_matrix(rotation).as_quat(canonical=True)[[3, 0, 1, 2]]
    return Pose(position.copy(), q)


def pose_from_values(values) -> Pose:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Expected finite tx ty tz qw qx qy qz")
    return make_pose(values[:3], rotation_matrix(values[3:]))


def apply_pose(points: np.ndarray, pose: Pose) -> np.ndarray:
    return np.asarray(points) @ rotation_matrix(pose.quaternion).T + pose.position


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine matrix, including uniform scale if present."""
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def assembly_normalization(vertices: np.ndarray, source_to_z_up: np.ndarray) -> np.ndarray:
    """Normalize assembled AABB diagonal to one and place its bottom at z=0."""
    make_pose(np.zeros(3), source_to_z_up)
    rotated = vertices @ source_to_z_up.T
    low, high = rotated.min(axis=0), rotated.max(axis=0)
    diagonal = np.linalg.norm(high - low)
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Degenerate assembled bounding box")
    origin = (low + high) / 2
    origin[2] = low[2]
    matrix = np.eye(4)
    matrix[:3, :3] = source_to_z_up / diagonal
    matrix[:3, 3] = -origin / diagonal
    return matrix


def pca_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return origin and right-handed basis; smallest principal axis is local Z.

    Repeated-eigenvalue subspaces are resolved by projecting canonical world axes,
    rather than relying on arbitrary eigenvectors returned by a LAPACK backend.
    """
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    centered = vertices - vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    values, vectors = values[::-1], vectors[:, ::-1]
    tolerance = max(float(values[0]), np.finfo(float).tiny) * 1e-10
    axes = []
    start = 0
    while start < 3:
        end = start + 1
        while end < 3 and abs(values[end] - values[start]) <= tolerance:
            end += 1
        subspace = vectors[:, start:end]
        projector = subspace @ subspace.T
        chosen = []
        for axis in np.eye(3):
            candidate = projector @ axis
            for previous in chosen:
                candidate -= previous * np.dot(previous, candidate)
            length = np.linalg.norm(candidate)
            if length > 1e-8:
                candidate /= length
                if candidate[np.argmax(np.abs(candidate))] < 0:
                    candidate *= -1
                chosen.append(candidate)
            if len(chosen) == end - start:
                break
        axes.extend(chosen)
        start = end
    basis = np.column_stack(axes)
    basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
    return center, basis
