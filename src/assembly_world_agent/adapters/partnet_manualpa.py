"""ManualPA objs share a Y-up shape frame; never apply placement matrices."""

import numpy as np

from .common import build_sample

REPO_ID = "AssemblyWorld/partnet-manualpa"
SAMPLE_ID_FIELD = "object_id"
DEFAULT_REVISION = "e79907b38a868589184063debd9727e59f480cf3"
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
