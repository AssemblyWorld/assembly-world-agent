"""Shared maximum-part diagonal scale, independent of assembly poses."""

import numpy as np

from .transforms import pca_frame


def evaluation_scale(sample):
    """Task-space divisor making the largest part PCA AABB diagonal one."""
    diagonals = []
    for part in sample.parts:
        center, basis = pca_frame(part.mesh.vertices)
        canonical = (part.mesh.vertices - center) @ basis
        diagonals.append(float(np.linalg.norm(np.ptp(canonical, axis=0))))
    divisor = max(diagonals)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Invalid evaluation scale")
    return divisor
