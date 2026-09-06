"""Non-mutating mesh validation and sampling derivatives."""

from numbers import Integral

import mapbox_earcut
import numpy as np

from ..models import Mesh


def mesh_from_record(record: dict) -> Mesh:
    vertices = np.asarray(record["vertices"], dtype=np.float64).copy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Mesh vertices must have nonempty shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite")
    faces = []
    for face in record["faces"]:
        if len(face) < 3 or any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(vertices)
            for i in face
        ):
            raise ValueError("Invalid polygon vertex indices")
        faces.append(tuple(int(i) for i in face))
    if not faces:
        raise ValueError("Mesh has no faces")
    normals = np.asarray(record.get("normals", []), dtype=np.float64).copy()
    if normals.size == 0:
        normals = np.empty((0, 3))
    if normals.ndim != 2 or normals.shape[1] != 3 or not np.isfinite(normals).all():
        raise ValueError("Invalid mesh normals")
    indices = record.get("face_normal_indices", [])
    if indices:
        if len(indices) != len(faces) or any(
            len(normal_face) != len(face)
            or any(
                isinstance(i, bool) or not isinstance(i, Integral) or not -1 <= i < len(normals)
                for i in normal_face
            )
            for face, normal_face in zip(faces, indices)
        ):
            raise ValueError("Invalid corner normal indices")
    return Mesh(vertices, tuple(faces), normals, tuple(tuple(f) for f in indices))


def triangulate(mesh: Mesh) -> np.ndarray:
    """Triangulate polygon copies with earcut, including concave polygons.

    The original topology is untouched. Nonplanar polygons use their best-fit
    plane for triangulation and retain the original 3D vertices. Zero-area
    triangles are omitted from the sampling derivative, never from the mesh.
    """
    triangles = []
    for face in mesh.faces:
        indices = np.asarray(face, dtype=np.int64)
        if len(indices) == 3:
            triangles.append(indices[None, :])
            continue
        polygon = mesh.vertices[indices]
        centered = polygon - polygon.mean(axis=0)
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        if singular[1] <= np.finfo(float).eps * singular[0]:
            continue
        projected = np.ascontiguousarray(centered @ axes[:2].T)
        result = mapbox_earcut.triangulate_float64(
            projected, np.array([len(indices)], dtype=np.uint32)
        ).reshape(-1, 3)
        if len(result) != len(indices) - 2:
            raise ValueError("Polygon could not be fully triangulated")
        triangles.append(indices[result])
    if not triangles:
        raise ValueError("Mesh has zero effective surface area")
    result = np.concatenate(triangles)
    xyz = mesh.vertices[result]
    area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    valid = area > 0
    if not np.isfinite(area).all() or not valid.any():
        raise ValueError("Mesh has zero or invalid effective surface area")
    return result[valid]


def sample_surface(mesh: Mesh, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError("Surface sample count must be positive")
    xyz = mesh.vertices[triangulate(mesh)]
    areas = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    selected = xyz[rng.choice(len(xyz), size=count, p=areas / areas.sum())]
    random = rng.random((count, 2))
    root = np.sqrt(random[:, 0])
    weights = np.column_stack((1 - root, root * (1 - random[:, 1]), root * random[:, 1]))
    return np.einsum("ni,nij->nj", weights, selected)


def farthest_point_sample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices, with seeded first point and stable first-index tie breaks."""
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("FPS expects finite points with shape (N, 3)")
    if not 0 < count <= len(points):
        raise ValueError("FPS count must be between one and the candidate count")
    indices = np.empty(count, dtype=np.int64)
    minimum = np.full(len(points), np.inf)
    selected = np.zeros(len(points), dtype=bool)
    current = int(rng.integers(len(points)))
    for i in range(count):
        indices[i] = current
        selected[current] = True
        difference = points - points[current]
        minimum = np.minimum(minimum, np.einsum("ij,ij->i", difference, difference))
        minimum[selected] = -np.inf
        current = int(np.argmax(minimum))
    return indices
