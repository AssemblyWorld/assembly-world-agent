"""Decoded fragments share a Z-up source frame; identity is per fracture."""

import numpy as np

from .common import build_sample

REPO_ID = "AssemblyWorld/breaking-bad-volume-constrained"
SAMPLE_ID_FIELD = "sample_id"
DEFAULT_REVISION = "aa6c781cdba90b2131ba090457f502e144819953"
SOURCE_TO_Z_UP = np.eye(3)
SOURCE_TO_Z_UP.flags.writeable = False


def adapt(row: dict, revision: str):
    return build_sample(
        row,
        dataset=REPO_ID,
        revision=revision,
        source_to_z_up=SOURCE_TO_Z_UP,
        sample_id=row["sample_id"],
    )
