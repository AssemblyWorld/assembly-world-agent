"""Apply final view-0 poses to source-normalized local meshes before scaling."""

import numpy as np

from ..utils import pose_from_values
from .common import build_sample

REPO_ID = "AssemblyWorld/assemblybench"
SAMPLE_ID_FIELD = "object_id"
DEFAULT_REVISION = "266e883e1676af42c44c41484c77b662b987b966"
SOURCE_TO_Z_UP = np.eye(3)
SOURCE_TO_Z_UP.flags.writeable = False

REFERENCE_COLUMNS = ("steps",)


def reference_pages(row: dict, mode: str):
    """Use exactly one ordinary view-zero diagram per requested assembly step."""
    steps = [int(step["step_id"]) for step in row.get("steps", [])]
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("AssemblyBench requires unique assembly steps")
    selected = [max(steps)] if mode == "final-image" else sorted(steps)
    by_step = {step: [] for step in selected}
    for page in row.get("manual_pages", []):
        view = page.get("view_id")
        if page.get("kind") != "diagram" or not str(view).isdigit() or int(view) != 0:
            continue
        step = page.get("step_id")
        if step in by_step:
            by_step[step].append(page)
    for step, pages in by_step.items():
        if len(pages) != 1:
            raise ValueError(
                f"AssemblyBench step {step}: expected one view-0 diagram, got {len(pages)}"
            )
    return [by_step[step][0] for step in selected]


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
