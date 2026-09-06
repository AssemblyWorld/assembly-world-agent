import numpy as np
import pytest

from assembly_world_agent.utils import (
    apply_pose,
    farthest_point_sample,
    mesh_from_record,
    pca_frame,
    place_parts,
    sample_surface,
    separated,
    stable_rng,
    triangulate,
)


def test_concave_polygon_sampling_preserves_area_and_topology():
    # L shape: area 3; a triangle fan would incorrectly fill its missing corner.
    mesh = mesh_from_record(
        {
            "vertices": [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]],
            "faces": [[0, 1, 2, 3, 4, 5]],
        }
    )
    triangles = mesh.vertices[triangulate(mesh)]
    areas = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        / 2
    )
    assert areas.sum() == pytest.approx(3)
    points = sample_surface(mesh, 4096, stable_rng(0, "surface"))
    assert not np.any((points[:, 0] > 1) & (points[:, 1] > 1))
    assert mesh.faces == ((0, 1, 2, 3, 4, 5),)
    assert np.all(points[:, 2] == 0)


def test_surface_sampling_uses_area_not_face_count():
    mesh = mesh_from_record(
        {
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [13, 0, 0], [10, 3, 0]],
            "faces": [[0, 1, 2], [3, 4, 5]],
        }
    )
    points = sample_surface(mesh, 20000, stable_rng(0, "area"))
    assert np.mean(points[:, 0] > 5) == pytest.approx(0.9, abs=0.015)


def test_fps_spread_and_duplicate_candidates():
    points = np.column_stack((np.arange(20), np.zeros((20, 2))))
    indices = farthest_point_sample(points, 5, stable_rng(0))
    distances = np.abs(points[:, None, 0] - points[indices, 0])
    assert distances.min(axis=1).max() <= 4
    assert len(set(indices)) == 5
    repeated = farthest_point_sample(np.zeros((10, 3)), 10, stable_rng(0))
    assert len(set(repeated)) == 10


@pytest.mark.parametrize("dimensions", [(1, 1, 1), (2, 2, 0.01), (10, 0.1, 0.01)])
def test_pca_degeneracy_and_smallest_axis(dimensions):
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * dimensions
    center, basis = pca_frame(vertices)
    _, permuted = pca_frame(vertices[::-1])
    np.testing.assert_allclose(basis, permuted, atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) == pytest.approx(1)
    variances = np.var((vertices - center) @ basis, axis=0)
    assert variances[2] <= min(variances[:2]) + 1e-12


@pytest.mark.parametrize("count", [1, 2, 49])
@pytest.mark.parametrize("fallback", [False, True])
def test_grounded_nonoverlap_with_long_parts_and_grid_fallback(count, fallback):
    vertices = {
        str(i): np.array(
            [[-4.0, -0.01, -0.2], [4.0, -0.01, -0.2], [4.0, 0.01, 0.2], [-4.0, 0.01, 0.2]]
        )
        for i in range(count)
    }
    poses = place_parts(
        vertices,
        dataset="test",
        sample_id="long",
        seed=7,
        gap=0.02,
        attempts=0 if fallback else 128,
    )
    boxes = []
    for pid, pose in poses.items():
        points = apply_pose(vertices[pid], pose)
        assert points[:, 2].min() == pytest.approx(0, abs=1e-12)
        box = np.array([points[:, :2].min(0), points[:, :2].max(0)])
        assert all(separated(box, other, 0.02 - 1e-12) for other in boxes)
        boxes.append(box)
