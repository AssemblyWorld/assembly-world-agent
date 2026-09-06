"""Apply final view-0 poses to source-normalized local meshes before scaling."""

import numpy as np

from ..utils import pose_from_values
from .common import build_sample

REPO_ID = "AssemblyWorld/assemblybench"
SAMPLE_ID_FIELD = "object_id"
DEFAULT_REVISION = "266e883e1676af42c44c41484c77b662b987b966"
SOURCE_TO_Z_UP = np.eye(3)
SOURCE_TO_Z_UP.flags.writeable = False


def adapt(row: dict, revision: str):
    views = row.get("poses", {})
    steps = views.get("0", {})
    if not steps or not row.get("steps"):
        raise ValueError("AssemblyBench requires view-0 assembled poses and steps")
    final_step = str(max(int(step["step_id"]) for step in row["steps"]))
    if final_step not in steps:
        raise ValueError(f"Missing final AssemblyBench pose step {final_step}")
    poses = {part_id: pose_from_values(values) for part_id, values in steps[final_step].items()}
    return build_sample(
        row,
        dataset=REPO_ID,
        revision=revision,
        source_to_z_up=SOURCE_TO_Z_UP,
        sample_id=row["object_id"],
        poses=poses,
    )
