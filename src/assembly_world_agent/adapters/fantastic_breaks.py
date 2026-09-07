"""Broken and synthetic repair meshes share the supplied assembly frame.

Identity axis conversion is a task convention, not a verified physical up axis.
The complete reference and the unverified annotation matrix remain annotations.
"""

import numpy as np

from .common import build_sample

REPO_ID = "AssemblyWorld/fantastic-breaks"
SAMPLE_ID_FIELD = "object_id"
DEFAULT_REVISION = "654d94e30e3cb246a04f97aab1bc4ca9f0ad0af9"
LOAD_KWARGS = {"writer_batch_size": 1}
SOURCE_TO_Z_UP = np.eye(3)
SOURCE_TO_Z_UP.flags.writeable = False


def adapt(row: dict, revision: str):
    return build_sample(
        row,
        dataset=REPO_ID,
        revision=revision,
        source_to_z_up=SOURCE_TO_Z_UP,
        sample_id=row["object_id"],
    )
