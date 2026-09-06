import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from assembly_world_agent import adapt_sample, export_episode, prepare_sample
from assembly_world_agent.artifacts import read_config, write_task
from assembly_world_agent.episodes import CONTRACT_DIRECTORY, contract_provenance, sha256
from assembly_world_agent.utils import rotation_matrix, triangulate

mujoco = pytest.importorskip("mujoco")
jsonschema = pytest.importorskip("jsonschema")


def test_setup_archive_determinism_and_native_state(row, tmp_path):
    task = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    originals = [p.mesh.vertices.copy() for p in task.parts]
    result = export_episode(task, tmp_path / "one.episode.zip")
    other = export_episode(task, tmp_path / "two.episode.zip")
    assert result.sha256 == other.sha256
    assert result.path.read_bytes() == other.path.read_bytes()
    assert export_episode(task, result.path) == result
    with zipfile.ZipFile(result.path) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(files.pop("manifest.json"))
    assert manifest["hashes"] == {k: sha256(v) for k, v in files.items()}
    assert manifest["lifecycle"] == "setup" and manifest["units"] == "normalized"
    assert not manifest["runtime"]["physics"]["enabled"]
    assert not manifest["runtime"]["physics"]["detection"]
    assert not {"apply_force", "advance_simulation"} & set(manifest["runtime"]["enabledTools"])
    assert files["calls.jsonl"] == files["events.jsonl"] == b""
    assert all(
        k.startswith("world/") or k in {"frames.jsonl", "frames.bin", "calls.jsonl", "events.jsonl"}
        for k in files
    )
    assert not any(k.startswith("observations/") or "target" in k or "manual" in k for k in files)
    metadata = json.loads(files["frames.jsonl"])
    assert metadata.pop("kind") == "initial"
    state = np.frombuffer(files["frames.bin"], dtype="<f8")
    assert len(state) == manifest["stateSize"]
    logical = dict(
        manifest=manifest,
        initial={**metadata, "integration": state.tolist()},
        states=[],
        calls=[],
        events=[],
        trajectory=[],
        revision=0,
    )
    jsonschema.validate(
        logical, json.loads((CONTRACT_DIRECTORY / "episode.schema.json").read_bytes())
    )
    model = mujoco.MjModel.from_xml_string(
        files["world/model.xml"].decode(),
        assets={
            k[6:]: v for k, v in files.items() if k.startswith("world/") and not k.endswith("xml")
        },
    )
    data = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, state.copy(), spec)
    mujoco.mj_forward(model, data)
    restored = np.empty_like(state)
    mujoco.mj_getState(model, data, restored, spec)
    np.testing.assert_array_equal(restored, state)
    for part, original in zip(task.parts, originals):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, result.part_ids[part.part_id])
        np.testing.assert_allclose(data.xpos[body], part.initial_pose.position, atol=1e-12)
        np.testing.assert_allclose(
            data.xmat[body].reshape(3, 3), rotation_matrix(part.initial_pose.quaternion), atol=1e-12
        )
        np.testing.assert_array_equal(part.mesh.vertices, original)
    assert contract_provenance()["commit"].startswith("9ce88b6")


def test_resources_stay_in_memory_and_id_mapping(row, tmp_path):
    from PIL import Image

    row["manual_pages"] = [{"image": Image.new("RGBA", (5, 7), (1, 2, 3, 100)), "page_index": 0}]
    task = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    record = write_task(task, tmp_path)
    assert write_task(task, tmp_path) == record
    directory = Path(record["config_directory"])
    config = read_config(directory)
    assert config["samples"][task.sample_id]["episode_id"] == record["episode_id"]
    assert sorted(p.name for p in directory.iterdir()) == [
        "Test--object.episode.zip",
        "config.json",
    ]
    assert task.manual_pages[0]["image"].getpixel((0, 0)) == (1, 2, 3, 100)
    assert len(task.parts[0].points) == 1000
    with zipfile.ZipFile(directory / record["episode"]) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["objects"][0]["id"] == "part-0001"


def test_lone_triangle_remains_planar(row, tmp_path):
    row["parts"] = [
        {
            "part_id": "triangle",
            "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "faces": [[0, 1, 2]],
        }
    ]
    row["parts_ct"] = 1
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    export = export_episode(sample, tmp_path / "triangle.episode.zip")
    with zipfile.ZipFile(export.path) as z:
        obj = z.read("world/meshes/part-0001.obj").decode()
    points = np.array(
        [[float(x) for x in line.split()[1:]] for line in obj.splitlines() if line.startswith("v ")]
    )
    assert len(points) == 4 and np.linalg.matrix_rank(points - points.mean(0)) == 2
    assert len(triangulate(sample.parts[0].mesh)) == 1


def test_no_overwrite_and_invalid_mesh(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    path = tmp_path / "existing.zip"
    path.write_bytes(b"existing run")
    with pytest.raises(FileExistsError):
        export_episode(sample, path)
    assert path.read_bytes() == b"existing run"
    sample.parts[0].mesh.vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        export_episode(sample, tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_export_changes_with_initialization(row, tmp_path):
    from assembly_world_agent import PreparationConfig

    source = adapt_sample("ikea-manual", row, revision="fixture")
    a = export_episode(prepare_sample(source), tmp_path / "a.zip")
    b = export_episode(
        prepare_sample(source, PreparationConfig(initialization_seed=1)), tmp_path / "b.zip"
    )
    assert a.episode_id != b.episode_id
    assert a.part_ids == b.part_ids


def test_part_order_does_not_change_archive(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    original = export_episode(sample, tmp_path / "ordered.zip")
    reordered = export_episode(
        replace(sample, parts=tuple(reversed(sample.parts))), tmp_path / "reordered.zip"
    )
    assert original.path.read_bytes() == reordered.path.read_bytes()


def test_export_rejects_invalid_indices_and_duplicate_ids(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    invalid = replace(sample.parts[0], mesh=replace(sample.parts[0].mesh, faces=((0, 1, 999),)))
    with pytest.raises(ValueError):
        export_episode(replace(sample, parts=(invalid,)), tmp_path / "invalid.zip")
    with pytest.raises(ValueError, match="unique"):
        export_episode(
            replace(sample, parts=(sample.parts[0], sample.parts[0])), tmp_path / "duplicate.zip"
        )
    assert not list(tmp_path.iterdir())
