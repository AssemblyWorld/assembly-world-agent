"""Ground-truth reconstruction and validated, headless final-state extraction."""

from xml.etree import ElementTree as ET

import numpy as np

from ..episodes import ENGINE, _render_mesh, _scene
from ..similarity import SimilarityConfig, resolve_equivalence
from ..utils.scale import evaluation_scale as evaluation_scale


def equivalence_groups(sample):
    """Compatibility wrapper; source syntax is interpreted by adapters."""
    result = resolve_equivalence(sample, config=SimilarityConfig(policy="source"))
    ids = sorted(p.part_id for p in sample.parts)
    return [[ids.index(pid) for pid in group] for group in result["groups"]], result[
        "ignored_composite_self_groups"
    ]


def final_poses(episode, sample):
    """Validate original geometry and restore body poses without stepping or rendering."""
    import mujoco as mj

    if mj.__version__ != ENGINE:
        raise ValueError(f"Expected MuJoCo {ENGINE}")
    parts = sorted(sample.parts, key=lambda part: part.part_id)
    mapping = {part.part_id: f"part-{i + 1:04d}" for i, part in enumerate(parts)}
    expected_objects = [
        dict(id=mapping[p.part_id], name=f"Part {p.part_id}", type="rigid", collisionEnabled=False)
        for p in parts
    ]
    manifest, files = episode["manifest"], episode["files"]
    if manifest["objects"] != expected_objects:
        raise ValueError("Episode object catalog differs from reconstructed parts")
    scene, _ = _scene(sample, mapping)
    if files["world/model.xml"] != scene["model.xml"]:
        raise ValueError("Episode model differs from reconstructed task model")
    for part in parts:
        name = mapping[part.part_id]
        text = files[f"world/meshes/{name}.obj"].decode().splitlines()
        vertices = np.array(
            [[float(v) for v in line.split()[1:]] for line in text if line.startswith("v ")]
        )
        faces = np.array(
            [[int(v) - 1 for v in line.split()[1:]] for line in text if line.startswith("f ")]
        )
        expected_vertices, expected_faces = _render_mesh(part)
        if (
            vertices.shape != expected_vertices.shape
            or not np.allclose(vertices, expected_vertices, atol=2e-7, rtol=0)
            or not np.array_equal(faces, expected_faces)
        ):
            raise ValueError(f"Episode geometry differs: {part.part_id}")
    # Validate asset paths before passing XML to the native compiler.
    xml = ET.fromstring(files["world/model.xml"])
    if any("world/" + e.attrib["file"] not in files for e in xml.findall("./asset/mesh")):
        raise ValueError("Missing episode mesh asset")
    model = mj.MjModel.from_xml_string(
        files["world/model.xml"].decode(),
        assets={
            k[6:]: v for k, v in files.items() if k.startswith("world/") and k != "world/model.xml"
        },
    )
    if mj.mj_stateSize(model, manifest["stateSpec"]) != manifest["stateSize"]:
        raise ValueError("Episode state size differs from native model")
    frame = episode["states"][max(episode["states"])]
    data = mj.MjData(model)
    mj.mj_setState(model, data, np.array(frame["integration"]), manifest["stateSpec"])
    mj.mj_forward(model, data)
    poses = []
    for part in parts:
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, mapping[part.part_id])
        if body <= 0:
            raise ValueError("Missing episode body")
        poses.append((data.xmat[body].reshape(3, 3).copy(), data.xpos[body].copy()))
    return poses, frame["index"]
