"""Independent setup-archive producer for the pinned 3DWebAgent public contract."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .models import AssemblySample
from .utils import apply_pose, mesh_from_record, rotation_matrix, triangulate

CONTRACT = "3dwebagent-runtime-1"
ENGINE = "3.12.0"
IMPLEMENTATION = "assembly-world-agent-episode-v1"
CONTRACT_DIRECTORY = Path(__file__).parent / "contracts"
ENABLED_TOOLS = [
    "list_objects",
    "get_object",
    "get_scene",
    "get_state",
    "translate_objects",
    "rotate_objects",
    "set_object_pose",
    "group_objects",
    "ungroup_objects",
    "capture_scene",
    "move_camera",
    "start_episode",
    "end_episode",
]


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def contract_provenance() -> dict:
    provenance = json.loads((CONTRACT_DIRECTORY / "provenance.json").read_bytes())
    if sha256((CONTRACT_DIRECTORY / "episode.schema.json").read_bytes()) != provenance["sha256"]:
        raise ValueError("Pinned episode schema checksum differs")
    return provenance


@dataclass(frozen=True)
class EpisodeExport:
    path: Path
    episode_id: str
    sha256: str
    part_ids: dict[str, str]


def _obj(part) -> bytes:
    # Revalidate at the API boundary: callers can edit the dataclass's NumPy buffers.
    mesh = mesh_from_record(
        {
            "vertices": part.mesh.vertices,
            "faces": part.mesh.faces,
            "normals": part.mesh.normals,
            "face_normal_indices": part.mesh.face_normal_indices,
        }
    )
    vertices = mesh.vertices
    faces = triangulate(mesh)
    if len(vertices) == 3:
        # MuJoCo requires at least four vertices; preserve the exact source surface.
        vertices = np.vstack([vertices, vertices.mean(axis=0)])
        faces = np.array([[a, b, 3] for face in faces for a, b in zip(face, np.roll(face, -1))])
    lines = ["v " + " ".join(format(float(v), ".17g") for v in point) for point in vertices]
    lines.extend("f " + " ".join(str(int(i) + 1) for i in face) for face in faces)
    return ("\n".join(lines) + "\n").encode()


def _scene(sample: AssemblySample, mapping: dict[str, str]) -> tuple[dict[str, bytes], dict]:
    root = ET.Element("mujoco", model="assembly-task")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    option = ET.SubElement(root, "option", timestep="0.002", gravity="0 0 0")
    ET.SubElement(option, "flag", contact="disable", constraint="disable", autoreset="disable")
    asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    files, bounds = {}, []
    for part in sorted(sample.parts, key=lambda p: p.part_id):
        name = mapping[part.part_id]
        files[f"meshes/{name}.obj"] = _obj(part)
        pose = part.initial_pose
        rotation_matrix(pose.quaternion)
        if pose.position.shape != (3,) or not np.isfinite(pose.position).all():
            raise ValueError(f"Invalid initial position: {part.part_id}")
        ET.SubElement(asset, "mesh", name=name, file=f"meshes/{name}.obj", inertia="shell")
        body = ET.SubElement(
            world,
            "body",
            name=name,
            pos=" ".join(format(float(v), ".17g") for v in pose.position),
            quat=" ".join(format(float(v), ".17g") for v in pose.quaternion),
        )
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="0.1 0.1 0.1")
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=name,
            contype="0",
            conaffinity="0",
            rgba="0.70 0.73 0.79 1",
        )
        bounds.append(apply_pose(part.mesh.vertices, pose))
    points = np.concatenate(bounds)
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low) / 2), 0.01)
    direction = np.array([1.3, -1.8, 1.5])
    direction /= np.linalg.norm(direction)
    # A bounding sphere at 3.5 radii fits both axes of the platform's 4:3 agent camera.
    camera = {"position": (center + direction * radius * 3.5).tolist(), "target": center.tolist()}
    ET.indent(root)
    files["model.xml"] = ET.tostring(root, encoding="utf-8")
    return files, camera


def export_episode(sample: AssemblySample, output: str | Path) -> EpisodeExport:
    """Write a deterministic setup ZIP. Identical existing output is reused.

    Source and prepared geometry are never modified. Ground truth, annotations,
    manuals, and evaluation points are intentionally excluded. Requires [episodes].
    """
    try:
        import jsonschema
        import mujoco
    except ImportError as error:
        raise ImportError("Episode export requires: uv sync --extra episodes") from error
    if mujoco.__version__ != ENGINE:
        raise ValueError(f"Episode export requires MuJoCo {ENGINE}")
    provenance = contract_provenance()
    ids = sorted(p.part_id for p in sample.parts)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Episode requires nonempty unique part IDs")
    mapping = {part_id: f"part-{i + 1:04d}" for i, part_id in enumerate(ids)}
    assets, camera = _scene(sample, mapping)
    with tempfile.TemporaryDirectory(prefix="awa-episode-") as directory:
        for name, payload in assets.items():
            path = Path(directory) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        model = mujoco.MjModel.from_xml_path(str(Path(directory) / "model.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if model.nbody - 1 != len(ids) or model.njnt != len(ids):
            raise ValueError("Compiled object catalog differs")
        for part in sample.parts:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mapping[part.part_id])
            if body < 1 or not np.allclose(
                data.xpos[body], part.initial_pose.position, atol=1e-12, rtol=0
            ):
                raise ValueError("Compiled initial pose differs")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite compiled integration state")
    build = sha256(Path(__file__).read_bytes() + json_bytes(provenance))
    identity = sha256(
        json_bytes(
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "revision": sample.revision,
                "protocol": sample.protocol_version,
                "config": asdict(sample.config),
                "build": build,
                "assets": {k: sha256(v) for k, v in sorted(assets.items())},
                "camera": camera,
            }
        )
    )
    manifest = dict(
        format="3dwebagent-episode",
        version=1,
        id=f"awa-{identity}",
        contract=CONTRACT,
        engine=ENGINE,
        name=f"{sample.dataset.split('/')[-1]} / {sample.sample_id}",
        units="normalized",
        stateSpec=int(spec),
        stateSize=len(state),
        lifecycle="setup",
        originTime=0,
        task="Assemble the supplied parts using the accompanying IKEA manual. "
        "Use explicit object IDs and visual feedback. Physics is disabled. "
        "End the episode when finished; no automatic success score is supplied."
        if sample.dataset == "AssemblyWorld/ikea-manual"
        else "Assemble the supplied parts using the accompanying task resources. Physics is disabled.",
        requiredCapabilities=["state"],
        producer=dict(
            backend="native",
            engine=ENGINE,
            implementation=IMPLEMENTATION,
            language="python",
            build=build,
        ),
        runtime=dict(
            enabledTools=ENABLED_TOOLS.copy(),
            physics=dict(
                enabled=False,
                gravity=[0, 0, -9.81],
                detection=False,
                response=False,
                strategy="fixed",
                duration=0.25,
                maxDuration=5.0,
                quietDuration=0.1,
                linearThreshold=0.01,
                angularThreshold=0.01,
            ),
        ),
        objects=[
            dict(id=mapping[pid], name=f"Part {pid}", type="rigid", collisionEnabled=False)
            for pid in ids
        ],
        hashes={},
    )
    frame = dict(index=0, camera=camera, groups=[], groupCounter=0)
    logical = dict(
        manifest=manifest,
        initial={**frame, "integration": state.tolist()},
        states=[],
        calls=[],
        events=[],
        trajectory=[],
        revision=0,
    )
    schema = json.loads((CONTRACT_DIRECTORY / "episode.schema.json").read_bytes())
    jsonschema.Draft7Validator(schema).validate(logical)
    files = {"world/" + name: payload for name, payload in assets.items()}
    files.update(
        {
            "frames.jsonl": json_bytes({"kind": "initial", **frame}),
            "frames.bin": state.astype("<f8").tobytes(),
            "calls.jsonl": b"",
            "events.jsonl": b"",
        }
    )
    manifest["hashes"] = {k: sha256(v) for k, v in sorted(files.items())}
    files["manifest.json"] = json_bytes(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compresslevel=9)
    payload = buffer.getvalue()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace a different archive: {output}")
    else:
        # Exclusive creation prevents an existing episode from being replaced.
        with output.open("xb") as stream:
            stream.write(payload)
    return EpisodeExport(output, manifest["id"], sha256(payload), mapping)
