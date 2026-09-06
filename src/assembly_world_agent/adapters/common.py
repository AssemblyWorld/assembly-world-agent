"""Source record conversion shared by the four thin dataset adapters."""

from copy import deepcopy

import numpy as np

from ..models import Pose, SourcePart, SourceSample
from ..utils import make_pose, mesh_from_record


def build_sample(
    row: dict,
    *,
    dataset: str,
    revision: str,
    source_to_z_up: np.ndarray,
    sample_id: str,
    poses: dict[str, Pose] | None = None,
) -> SourceSample:
    parts = []
    ids = set()
    for record in row["parts"]:
        part_id = record["part_id"]
        if not isinstance(part_id, str) or not part_id or part_id in ids:
            raise ValueError("Part IDs must be nonempty unique strings")
        ids.add(part_id)
        if poses is not None and part_id not in poses:
            raise ValueError(f"Missing assembled pose for part {part_id}")
        parts.append(
            SourcePart(
                part_id=part_id,
                mesh=mesh_from_record(record),
                assembled_pose=poses[part_id]
                if poses is not None
                else make_pose(np.zeros(3), np.eye(3)),
                metadata=deepcopy(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in {"vertices", "faces", "normals", "face_normal_indices"}
                    }
                ),
            )
        )
    if not parts or len(parts) != row["parts_ct"]:
        raise ValueError("Source part count does not match nonempty geometry")
    if poses is not None and set(poses) != ids:
        raise ValueError("Final assembled poses and mesh part IDs differ")
    if not isinstance(sample_id, str) or not sample_id or not revision:
        raise ValueError("Sample identity and source revision are required")
    return SourceSample(
        dataset=dataset,
        sample_id=sample_id,
        revision=revision,
        parts=tuple(parts),
        source_to_z_up=source_to_z_up.copy(),
        metadata={
            "source_metadata": deepcopy(row.get("source_metadata", {})),
            "dataset_card_url": f"https://huggingface.co/datasets/{dataset}/blob/{revision}/README.md",
        },
        source_splits=deepcopy(row.get("source_splits", [])),
        manual_pages=tuple(deepcopy(row.get("manual_pages", []))),
        manual=deepcopy(row.get("manual", [])),
        steps=tuple(deepcopy(row.get("steps", []))),
        annotations=deepcopy(
            {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "parts",
                    "source_metadata",
                    "source_splits",
                    "manual_pages",
                    "manual",
                    "steps",
                }
            }
        ),
    )
