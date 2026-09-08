"""Small projected axes must still form a valid rigid coordinate frame."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from assembly_world_agent.utils import make_pose, pca_frame, rotation_matrix


@pytest.mark.parametrize("offset", [3.1e-8, 4.1e-8, 1e-7])
def test_pca_projection_cancellation_preserves_rigid_frame(offset):
    vertices = np.array(
        [
            [offset, 1, 0],
            [-offset, 1, 0],
            [offset, -1, 0],
            [-offset, -1, 0],
            [3, 0, 0],
            [-3, 0, 0],
            [0, 0, 0.5],
            [0, 0, -0.5],
        ]
    )
    rotation = Rotation.from_euler("xyz", [17, 29, 43], degrees=True).as_matrix()
    vertices = vertices @ rotation.T
    original = vertices.copy()
    center, basis = pca_frame(vertices)
    pose = make_pose(center, basis)
    local = (vertices - center) @ basis
    reconstructed = local @ rotation_matrix(pose.quaternion).T + pose.position
    np.testing.assert_allclose(reconstructed, original, atol=1e-12, rtol=0)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14, rtol=0)
    assert np.linalg.det(basis) == pytest.approx(1, abs=1e-14)
    np.testing.assert_array_equal(vertices, original)
