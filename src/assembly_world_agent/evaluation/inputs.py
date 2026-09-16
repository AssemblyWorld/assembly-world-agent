"""Ground-truth reconstruction and validated, headless final-state extraction."""

from xml.etree import ElementTree as ET

import numpy as np

from ..episode_model import load_model, validate_compiled_model
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


def effective_faces(vertices, faces, *, relative=1e-12):
    """Triangles whose area is not negligible relative to the squared part extent."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return faces
    xyz = vertices[faces]
    area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0))) ** 2 or 1.0
    return faces[area > relative * scale]


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
    if manifest.get("model"):
        model = load_model(episode)
        expected = mj.MjModel.from_xml_string(scene["model.xml"].decode(), assets=scene)
        validate_compiled_model(model, expected)
        del expected
    else:
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
            if vertices.shape != expected_vertices.shape or not np.allclose(
                vertices, expected_vertices, atol=2e-7, rtol=0
            ):
                raise ValueError(f"Episode geometry differs: {part.part_id}")
            # Last-bit coordinate drift between code builds can flip a degenerate
            # triangle's area between exactly zero (omitted) and ~1e-18 (kept), so
            # compare face lists after dropping negligible-area triangles on both sides.
            if not np.array_equal(faces, expected_faces) and not np.array_equal(
                effective_faces(vertices, faces), effective_faces(expected_vertices, expected_faces)
            ):
                raise ValueError(f"Episode geometry differs: {part.part_id}")
        # Validate asset paths before passing XML to the native compiler.
        xml = ET.fromstring(files["world/model.xml"])
        if any("world/" + e.attrib["file"] not in files for e in xml.findall("./asset/mesh")):
            raise ValueError("Missing episode mesh asset")
        model = mj.MjModel.from_xml_string(
            files["world/model.xml"].decode(),
            assets={
                k[6:]: v
                for k, v in files.items()
                if k.startswith("world/") and k != "world/model.xml"
            },
        )
    return _restore_poses(model, episode, [mapping[part.part_id] for part in parts])


def _restore_poses(model, episode, names):
    import mujoco as mj

    manifest = episode["manifest"]
    if mj.mj_stateSize(model, manifest["stateSpec"]) != manifest["stateSize"]:
        raise ValueError("Episode state size differs from native model")
    frame = episode["states"][max(episode["states"])]
    data = mj.MjData(model)
    mj.mj_setState(model, data, np.array(frame["integration"]), manifest["stateSpec"])
    mj.mj_forward(model, data)
    poses = []
    for name in names:
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body <= 0:
            raise ValueError("Missing episode body")
        poses.append((data.xmat[body].reshape(3, 3).copy(), data.xpos[body].copy()))
    return poses, frame["index"]


def final_poses_against_initial(episode, initial, part_ids):
    """Restore body poses of a final episode validated against its initial episode.

    Used when source geometry comes from the evaluation cache: the final episode
    must carry the initial episode's object catalog and world files (byte-identical
    for XML worlds, structurally identical compiled models for MJB worlds).
    """
    import mujoco as mj

    if mj.__version__ != ENGINE:
        raise ValueError(f"Expected MuJoCo {ENGINE}")
    ids = list(part_ids)
    if sorted(ids) != ids or len(set(ids)) != len(ids):
        raise ValueError("Part IDs must be sorted and unique")
    names = [f"part-{i + 1:04d}" for i in range(len(ids))]
    expected_objects = [
        dict(id=name, name=f"Part {pid}", type="rigid", collisionEnabled=False)
        for name, pid in zip(names, ids)
    ]
    manifest, files = episode["manifest"], episode["files"]
    reference, reference_files = initial["manifest"], initial["files"]
    if reference["objects"] != expected_objects:
        raise ValueError("Initial episode object catalog differs from cached parts")
    if manifest["objects"] != expected_objects:
        raise ValueError("Episode object catalog differs from reconstructed parts")
    for key in ("id", "producer", "contract", "engine", "stateSpec", "stateSize"):
        if manifest[key] != reference[key]:
            raise ValueError(f"Episode {key} differs from the initial episode")
    if manifest.get("model") != reference.get("model"):
        raise ValueError("Episode model declaration differs from the initial episode")
    world = {k: v for k, v in files.items() if k.startswith("world/")}
    reference_world = {k: v for k, v in reference_files.items() if k.startswith("world/")}
    if manifest.get("model"):
        if set(world) != set(reference_world):
            raise ValueError("Episode world inventory differs from the initial episode")
        model = load_model(episode)
        expected = load_model(initial)
        validate_compiled_model(model, expected)
        del expected
    else:
        if world != reference_world:
            raise ValueError("Episode world files differ from the initial episode")
        xml = ET.fromstring(files["world/model.xml"])
        if any("world/" + e.attrib["file"] not in files for e in xml.findall("./asset/mesh")):
            raise ValueError("Missing episode mesh asset")
        model = mj.MjModel.from_xml_string(
            files["world/model.xml"].decode(),
            assets={k[6:]: v for k, v in world.items() if k != "world/model.xml"},
        )
    return _restore_poses(model, episode, names)
