from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from assembly_world_agent import adapt_sample, load_samples, loading, prepare_sample
from assembly_world_agent.adapters import get_adapter
from assembly_world_agent.similarity import SimilarityConfig, resolve_equivalence

REVISION = "654d94e30e3cb246a04f97aab1bc4ca9f0ad0af9"


@pytest.mark.parametrize("name", ["fantastic-breaks", "AssemblyWorld/fantastic-breaks"])
@pytest.mark.parametrize("streaming", [False, True])
def test_loading_identity_cache_and_explicit_streaming(monkeypatch, fantastic_row, name, streaming):
    mock = Mock(return_value=[fantastic_row])
    monkeypatch.setattr(loading, "load_dataset", mock)
    monkeypatch.setattr(loading, "HfApi", Mock(side_effect=AssertionError("Unnecessary Hub query")))
    kwargs = {"streaming": True} if streaming else {}
    sample = next(load_samples(name, sample_ids=["00/00002"], cache_dir="cache", **kwargs))
    assert sample.sample_id == "00/00002"
    assert sample.revision == get_adapter(name).DEFAULT_REVISION == REVISION
    assert sample.dataset == "AssemblyWorld/fantastic-breaks"
    expected = dict(
        split="full", revision=REVISION, streaming=streaming, cache_dir="cache", token=None
    )
    if not streaming:
        expected["writer_batch_size"] = 1
    mock.assert_called_once_with("AssemblyWorld/fantastic-breaks", **expected)


def test_source_geometry_and_annotations_are_preserved(fantastic_row):
    before = deepcopy(fantastic_row)
    source = adapt_sample("fantastic-breaks", fantastic_row, revision=REVISION)
    assert len(source.parts) == 2
    assert source.annotations["complete_reference"] == before["complete_reference"]
    assert source.annotations["annotation"] == before["annotation"]
    assert source.metadata["source_metadata"] == before["source_metadata"]
    assert source.source_splits == [] and source.manual_pages == ()
    np.testing.assert_array_equal(source.source_to_z_up, np.eye(3))
    for part, record in zip(source.parts, before["parts"]):
        np.testing.assert_array_equal(part.mesh.vertices, record["vertices"])
        np.testing.assert_array_equal(
            part.mesh.normals, np.asarray(record["normals"]).reshape(-1, 3)
        )
        assert part.mesh.faces == tuple(tuple(f) for f in record["faces"])
        for key in ("role", "vertex_colors", "source_file", "ply_header"):
            assert part.metadata[key] == record[key]
        np.testing.assert_array_equal(part.assembled_pose.position, np.zeros(3))
        np.testing.assert_array_equal(part.assembled_pose.quaternion, [1, 0, 0, 0])
    source.parts[0].mesh.vertices[0, 0] += 1
    source.parts[0].metadata["vertex_colors"][0][0] = 99
    source.annotations["annotation"]["transform"][0][3] += 1
    assert fantastic_row == before


def test_reference_and_unverified_transform_never_change_task(fantastic_row):
    before = deepcopy(fantastic_row)
    original = prepare_sample(adapt_sample("fantastic-breaks", fantastic_row, revision=REVISION))
    changed = deepcopy(fantastic_row)
    changed["complete_reference"]["vertices"] = [[-1000, -2000, -3000]]
    changed["annotation"]["transform"] = (np.eye(4) * 42).tolist()
    other = prepare_sample(adapt_sample("fantastic-breaks", changed, revision=REVISION))
    assert fantastic_row == before
    np.testing.assert_array_equal(original.source_to_world, other.source_to_world)
    np.testing.assert_array_equal(original.world_to_source, other.world_to_source)
    for a, b in zip(original.parts, other.parts):
        np.testing.assert_array_equal(a.mesh.vertices, b.mesh.vertices)
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.gt_pose.position, b.gt_pose.position)
        np.testing.assert_array_equal(a.gt_pose.quaternion, b.gt_pose.quaternion)
        np.testing.assert_array_equal(a.initial_pose.position, b.initial_pose.position)
        np.testing.assert_array_equal(a.initial_pose.quaternion, b.initial_pose.quaternion)
    groups = resolve_equivalence(original, config=SimilarityConfig(policy="source"))
    assert groups["source_annotation_missing"]
    assert groups["groups"] == [["model_b_0"], ["model_r_0"]]


def test_episode_and_rebuilt_resources_exclude_reference(monkeypatch, fantastic_row, tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("jsonschema")
    from pathlib import Path

    from assembly_world_agent.artifacts import load_prepared, write_task
    from assembly_world_agent.episode_io import read_episode
    from assembly_world_agent.evaluation.inputs import final_poses

    task = prepare_sample(adapt_sample("fantastic-breaks", fantastic_row, revision=REVISION))
    record = write_task(task, tmp_path)
    monkeypatch.setattr(loading, "load_dataset", Mock(return_value=[fantastic_row]))
    _, rebuilt = next(load_prepared(record["config_directory"]))
    episode = read_episode(Path(record["config_directory"]) / record["episode"])
    assert len(episode["manifest"]["objects"]) == 2
    assert not any("reference" in name or "annotation" in name for name in episode["files"])
    poses, frame = final_poses(episode, rebuilt)
    assert frame == 0 and len(poses) == 2
    for a, b in zip(task.parts, rebuilt.parts):
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.gt_pose.position, b.gt_pose.position)
    changed = deepcopy(fantastic_row)
    changed["complete_reference"] = {}
    changed["annotation"] = {}
    other = prepare_sample(adapt_sample("fantastic-breaks", changed, revision=REVISION))
    assert write_task(other, tmp_path) == record
