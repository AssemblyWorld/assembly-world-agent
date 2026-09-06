"""IKEA's assembled meshes are Y-up; see the coordinate inspection notes in README.md."""

import numpy as np

from .common import build_sample

REPO_ID = "AssemblyWorld/ikea-manual"
SAMPLE_ID_FIELD = "object_id"
DEFAULT_REVISION = "d2367e6f86610d38f2d2cc65278f680c9a444d83"
SOURCE_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
SOURCE_TO_Z_UP.flags.writeable = False


def adapt(row: dict, revision: str):
    return build_sample(
        row,
        dataset=REPO_ID,
        revision=revision,
        source_to_z_up=SOURCE_TO_Z_UP,
        sample_id=row["object_id"],
    )
