"""Optional MJB producer; the legacy XML producer and its identity remain unchanged."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .episodes import CONTRACT, ENABLED_TOOLS, ENGINE, EpisodeExport, _scene, json_bytes, sha256
from .models import AssemblySample

CONTRACT_DIRECTORY = Path(__file__).parent / "contracts" / "mjb"


def mjb_provenance():
    provenance = json.loads((CONTRACT_DIRECTORY / "provenance.json").read_bytes())
    if sha256((CONTRACT_DIRECTORY / "episode.schema.json").read_bytes()) != provenance["sha256"]:
        raise ValueError("Pinned MJB schema checksum differs")
    return provenance


def export_mjb_episode(sample: AssemblySample, output: str | Path) -> EpisodeExport:
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
    provenance = mjb_provenance()
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
        binary = Path(directory) / "model.mjb"
        mujoco.mj_saveModel(model, str(binary))
        assets = {"model.mjb": binary.read_bytes()}
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite compiled integration state")
    package = Path(__file__).parent
    producer_sources = [
        Path(__file__),
        package / "episodes.py",
        package / "preparation.py",
        *sorted((package / "utils").glob("*.py")),
    ]
    build = sha256(
        b"".join(path.read_bytes() for path in producer_sources) + json_bytes(provenance)
    )
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
        model=dict(format="mjb", path="model.mjb"),
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
        "Body poses are transforms of baked initial geometry, not part centers. "
        "Specify an explicit pivot when rotating around a part center. "
        "Episode termination is disabled; no automatic success score is supplied."
        if sample.dataset == "AssemblyWorld/ikea-manual"
        else "Assemble the supplied parts using the accompanying task resources. Physics is disabled.",
        requiredCapabilities=["state"],
        producer=dict(
            backend="native",
            engine=ENGINE,
            implementation="assembly-world-agent-episode-v1-mjb",
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
