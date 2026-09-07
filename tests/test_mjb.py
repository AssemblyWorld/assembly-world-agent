from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from assembly_world_agent import PreparationConfig, adapt_sample, cli, prepare_sample
from assembly_world_agent.artifacts import read_config, write_task
from assembly_world_agent.episode_io import read_episode
from assembly_world_agent.episode_model import compiled_parts, load_model, validate_compiled_model
from assembly_world_agent.evaluation.inputs import final_poses
from assembly_world_agent.mjb import export_mjb_episode

mj = pytest.importorskip("mujoco")


def test_optional_format_defaults_and_cli(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(cli, "convert_dataset", mock)
    for command in (["convert", "fantastic-breaks", "--limit", "1"], ["convert-ikea"]):
        assert cli.main(command) == 0
        assert mock.call_args.kwargs["model_format"] == "xml"
    assert cli.main(["convert", "fantastic-breaks", "--limit", "1", "--model-format", "mjb"]) == 0
    assert mock.call_args.kwargs["model_format"] == "mjb"
    with pytest.raises(SystemExit):
        cli.main(["convert", "fantastic-breaks", "--all", "--model-format", "bad"])


def test_mjb_geometry_state_reuse_and_isolation(fantastic_row, tmp_path):
    source = adapt_sample("fantastic-breaks", fantastic_row, revision="a" * 40)
    sample = prepare_sample(source)
    originals = [p.mesh.vertices.copy() for p in sample.parts]
    a = write_task(sample, tmp_path, model_format="mjb")
    assert a == write_task(sample, tmp_path, model_format="mjb")
    xml = write_task(sample, tmp_path)
    assert xml["config_directory"] != a["config_directory"]
    assert "model_format" not in read_config(xml["config_directory"])["identity"]
    assert read_config(a["config_directory"])["identity"]["model_format"] == "mjb"
    ep = read_episode(Path(a["config_directory"]) / a["episode"])
    xml_ep = read_episode(Path(xml["config_directory"]) / xml["episode"])
    assert ep["manifest"]["version"] == 1
    assert ep["manifest"]["model"] == dict(format="mjb", path="model.mjb")
    assert [k for k in ep["files"] if k.startswith("world/")] == ["world/model.mjb"]
    model = load_model(ep)
    validate_compiled_model(model, load_model(xml_ep))
    geometry = compiled_parts(model, ep["manifest"]["objects"])
    for p, part, original in zip(geometry, sample.parts, originals):
        np.testing.assert_allclose(p["geometry"]["vertices"], part.mesh.vertices, atol=2e-7)
        np.testing.assert_array_equal(part.mesh.vertices, original)
    poses, index = final_poses(ep, sample)
    assert index == 0
    for (_, position), part in zip(poses, sample.parts):
        np.testing.assert_allclose(position, part.initial_pose.position, atol=1e-12)
    from assembly_world_agent.vis.results import episode_data

    assert len(episode_data(ep)["parts"]) == 2
    for changed in (replace(source, revision="b" * 40), source):
        config = (
            PreparationConfig(initialization_seed=1) if changed is source else PreparationConfig()
        )
        record = write_task(prepare_sample(changed, config), tmp_path, model_format="mjb")
        assert record["config_directory"] != a["config_directory"]
    model.mesh_vert[0, 0] += 0.05
    binary = tmp_path / "changed.mjb"
    mj.mj_saveModel(model, str(binary))
    ep["files"]["world/model.mjb"] = binary.read_bytes()
    with pytest.raises(ValueError, match="compiled model differs"):
        final_poses(ep, sample)


def test_mjb_no_overwrite_and_bad_descriptor(fantastic_row, tmp_path):
    sample = prepare_sample(adapt_sample("fantastic-breaks", fantastic_row, revision="a" * 40))
    path = tmp_path / "existing.zip"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export_mjb_episode(sample, path)
    assert path.read_bytes() == b"existing"
    with pytest.raises(ValueError, match="declaration"):
        load_model(dict(manifest=dict(model=dict(format="mjb", path="../model.mjb")), files={}))


def test_mjb_reference_annotations_do_not_enter_input(fantastic_row, tmp_path):
    changed = deepcopy(fantastic_row)
    changed["complete_reference"]["vertices"] = (
        np.asarray(changed["complete_reference"]["vertices"]) * 7 + 31
    ).tolist()
    changed["annotation"]["transform"] = (np.eye(4) * 5).tolist()
    a, b = [
        prepare_sample(adapt_sample("fantastic-breaks", row, revision="a" * 40))
        for row in (fantastic_row, changed)
    ]
    assert (
        export_mjb_episode(a, tmp_path / "a.zip").sha256
        == export_mjb_episode(b, tmp_path / "b.zip").sha256
    )
    for p, q in zip(a.parts, b.parts):
        np.testing.assert_array_equal(p.mesh.vertices, q.mesh.vertices)
        np.testing.assert_array_equal(p.gt_pose.position, q.gt_pose.position)
        np.testing.assert_array_equal(p.gt_pose.quaternion, q.gt_pose.quaternion)
