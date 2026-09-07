from copy import deepcopy

import numpy as np
import pytest


@pytest.fixture
def row():
    # A Y-up upright box and a separate thin planar part. Polygon faces matter.
    vertices = [[x, y, z] for x in (-1.0, 1.0) for y in (0.0, 4.0) for z in (-0.5, 0.5)]
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return {
        "object_id": "Test/object",
        "sample_id": "variant/Test/object/fracture-0",
        "parts_ct": 2,
        "parts": [
            {
                "part_id": "00",
                "vertices": vertices,
                "faces": faces,
                "normals": [[0.0, -1.0, 0.0]],
                "face_normal_indices": [[0] * 4] * 6,
                "annotation_part_id": "0",
            },
            {
                "part_id": "01",
                "vertices": [[3.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 0.1, 0.0], [3.0, 0.1, 0.0]],
                "faces": [[0, 1, 2, 3]],
                "normals": [],
                "face_normal_indices": [[-1] * 4],
            },
        ],
        "steps": [{"step_id": 0, "manual_page_index": None}, {"step_id": 1}],
        "manual_pages": [],
        "source_splits": {"image_pa": ["val", "test"], "dgl": []},
        "source_metadata": {"schema_version": "1", "unit_scale_to_meters": None},
        "assembly_order": ["00", "01"],
    }


@pytest.fixture
def fantastic_row(row):
    value = deepcopy(row)
    value.pop("sample_id")
    value["object_id"] = "00/00002"
    value["source_splits"] = []
    value["steps"] = []
    for part, role in zip(value["parts"], ("broken", "synthetic_repair")):
        part.pop("annotation_part_id", None)
        part.pop("face_normal_indices", None)
        part["part_id"] = "model_b_0" if role == "broken" else "model_r_0"
        part["role"] = role
        part["source_file"] = f"00/00002/{role}.ply"
        part["vertex_colors"] = (
            [[10, 20, 30, 255]] * len(part["vertices"]) if role == "broken" else []
        )
        part["ply_header"] = "ply\nformat binary_little_endian 1.0\nend_header\n"
    value["complete_reference"] = deepcopy(value["parts"][0])
    value["complete_reference"]["role"] = "complete_reference"
    value["complete_reference"]["vertices"] = (
        np.asarray(value["parts"][0]["vertices"]) * 100 + 200
    ).tolist()
    value["annotation"] = {
        "mask": [True, False] * 4,
        "mask_dtype": "bool",
        "mask_shape": [8],
        "mask_mesh_role": "broken",
        "transform": [[0, -1, 0, 99], [1, 0, 0, 88], [0, 0, 1, 77], [0, 0, 0, 1]],
        "transform_dtype": "float64",
        "transform_shape": [4, 4],
    }
    return value


@pytest.fixture
def assemblybench_row(row):
    value = deepcopy(row)
    value["poses"] = {
        "0": {
            "0": {"00": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
            "1": {
                "00": [1.0, 2.0, 3.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                "01": [3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }
    value["motions"] = [{"part_id": "00", "frames": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]}]
    value["manual"] = [{"step_idx": 0, "description": "Insert the part."}]
    return value
