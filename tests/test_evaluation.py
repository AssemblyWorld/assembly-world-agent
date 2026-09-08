import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from assembly_world_agent import adapt_sample, export_episode, prepare_sample
from assembly_world_agent.episode_io import read_episode
from assembly_world_agent.evaluation.geometry import (
    align,
    chamfer,
    metrics_from_errors,
    score_parts,
    transform,
)
from assembly_world_agent.evaluation.inputs import equivalence_groups, evaluation_scale, final_poses
from assembly_world_agent.evaluation.runner import summarize
from assembly_world_agent.utils import apply_pose, rotation_matrix


def test_chamfer_formula_and_threshold():
    a = np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 1.0, 0]])
    b = np.array([[1.0, 0, 0], [4.0, 1, 0]])
    distances = cdist(a, b, metric="sqeuclidean")
    assert chamfer(a, b) == pytest.approx(distances.min(0).mean() + distances.min(1).mean())
    assert metrics_from_errors([0.01, 0], 0.002) == dict(SCD=2.0, PA=1.0, SR=1)
    assert metrics_from_errors([np.nextafter(0.01, np.inf), 0], 0.002) == dict(
        SCD=2.0, PA=0.5, SR=0
    )


def clouds():
    rng = np.random.default_rng(52)
    return [
        rng.normal(size=(50, 3)) * [0.3, 0.15, 0.07] + center
        for center in ([0, 0, 0], [2, 0, 0], [0.7, 1, 0.2])
    ]


def test_global_alignment_and_local_error():
    target = clouds()
    rotation = Rotation.from_euler("xyz", [73, -41, 127], degrees=True).as_matrix()
    translation = np.array([12.0, -5.0, 3.0])
    prediction = [transform(p, rotation, translation) for p in target]
    r, t, info = align(np.concatenate(prediction), np.concatenate(target))
    assert info["chamfer"] < 1e-20
    assert np.linalg.det(r) == pytest.approx(1)
    score, _ = score_parts(
        [transform(p, r, t) for p in prediction],
        target,
        [[0], [1], [2]],
        ["a", "b", "c"],
        info["chamfer"],
    )
    assert score["PA"] == score["SR"] == 1
    prediction[1] += [0, 1.1, 0]
    r, t, info = align(np.concatenate(prediction), np.concatenate(target))
    score, _ = score_parts(
        [transform(p, r, t) for p in prediction],
        target,
        [[0], [1], [2]],
        ["a", "b", "c"],
        info["chamfer"],
    )
    assert score["SCD"] > 1 and score["PA"] < 1 and score["SR"] == 0


def test_equivalent_exchange_and_unique_exchange():
    a = clouds()[0]
    target = [a, a + [3, 0, 0]]
    prediction = target[::-1]
    matched, records = score_parts(prediction, target, [[0, 1]], ["a", "b"], 0)
    assert matched == dict(SCD=0.0, PA=1.0, SR=1)
    assert records[0]["target_part_id"] == "b"
    unique, _ = score_parts(prediction, target, [[0], [1]], ["a", "b"], 0)
    assert unique["PA"] == unique["SR"] == 0


def test_inspection_candidate_transforms_do_not_change_alignment():
    target = np.concatenate(clouds())
    prediction = target + [3, -2, 1]
    r, t, ordinary = align(prediction, target)
    ri, ti, inspection = align(prediction, target, include_candidate_transforms=True)
    np.testing.assert_array_equal(r, ri)
    np.testing.assert_array_equal(t, ti)
    assert ordinary["seed"] == inspection["seed"]
    for before, after in zip(ordinary["candidates"], inspection["candidates"]):
        assert before == {k: v for k, v in after.items() if k not in {"rotation", "translation"}}
        moved = transform(prediction, np.array(after["rotation"]), np.array(after["translation"]))
        assert chamfer(moved, target) == pytest.approx(after["chamfer"], abs=1e-12)


def test_no_scale_or_reflection_fitting():
    target = np.concatenate(clouds())
    for prediction in (target * 3, target * [-1, 1, 1]):
        r, _, info = align(prediction, target)
        assert np.linalg.det(r) == pytest.approx(1)
        assert info["chamfer"] > 0.001


def test_incorrect_relative_orientation():
    rng = np.random.default_rng(74)
    target = [
        rng.normal(size=(60, 3)) * [0.7, 0.04, 0.03] + center
        for center in ([0, 0, 0], [2, 0, 0], [0, 2, 0])
    ]
    prediction = [p.copy() for p in target]
    prediction[0] = target[0] @ Rotation.from_euler("z", 90, degrees=True).as_matrix().T
    rotation, translation, info = align(np.concatenate(prediction), np.concatenate(target))
    score, _ = score_parts(
        [transform(p, rotation, translation) for p in prediction],
        target,
        [[0], [1], [2]],
        ["0", "1", "2"],
        info["chamfer"],
    )
    assert score["SR"] == 0 and score["PA"] < 1


def test_source_groups_and_composites(row):
    from assembly_world_agent.adapters.equivalence import parse_source_equivalence

    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    parts = tuple(
        replace(p, metadata={"annotation_part_id": str(i)}) for i, p in enumerate(sample.parts)
    )
    for relation, expected, ignored in [
        ({}, [[0], [1]], []),
        ('{"0,1":["0,1"]}', [[0], [1]], ["0,1"]),
        ({"0": ["0", "1"]}, [[0, 1]], []),
    ]:
        normalized = parse_source_equivalence(parts, {"geometric_equivalence_relation": relation})
        assert equivalence_groups(replace(sample, parts=parts, source_equivalence=normalized)) == (
            expected,
            ignored,
        )
    with pytest.raises(ValueError, match="Unknown"):
        parse_source_equivalence(parts, {"geometric_equivalence_relation": {"0": ["99"]}})


def test_final_state_restoration_and_geometry(row, tmp_path, monkeypatch):
    mj = pytest.importorskip("mujoco")
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    episode = read_episode(export_episode(sample, tmp_path / "initial.zip").path)
    monkeypatch.setattr(mj, "Renderer", lambda *_: pytest.fail("Must not render"))
    poses, index = final_poses(episode, sample)
    assert index == 0
    assert all(np.allclose(t, 0) for r, t in poses)
    files = episode["files"]
    model = mj.MjModel.from_xml_string(
        files["world/model.xml"].decode(),
        assets={
            k[6:]: v for k, v in files.items() if k.startswith("world/") and k != "world/model.xml"
        },
    )
    data = mj.MjData(model)
    for i, part in enumerate(sorted(sample.parts, key=lambda p: p.part_id)):
        data.qpos[i * 7 : i * 7 + 7] = np.r_[part.gt_pose.position, part.gt_pose.quaternion]
    state = np.empty(episode["manifest"]["stateSize"])
    mj.mj_getState(model, data, state, episode["manifest"]["stateSpec"])
    episode["states"][5] = dict(
        index=5, integration=state.tolist(), groups=[{"members": ["a", "b"]}]
    )
    episode["traces"].append(dict(index=999, integration=[0] * len(state)))
    poses, index = final_poses(episode, sample)
    assert index == 5
    divisor = evaluation_scale(sample)
    predicted, target, seeds = [], [], []
    for part, (r, t) in zip(sample.parts, poses):
        assert np.allclose(r, rotation_matrix(part.gt_pose.quaternion))
        predicted.append(transform(part.points, r, t) / divisor)
        target.append(apply_pose(part.points, part.gt_pose) / divisor)
        seeds.append((np.eye(3), np.zeros(3)))
    _, _, info = align(np.concatenate(predicted), np.concatenate(target), seeds)
    assert info["chamfer"] < 1e-20
    modified = replace(
        sample,
        parts=(
            replace(
                sample.parts[0],
                mesh=replace(sample.parts[0].mesh, vertices=sample.parts[0].mesh.vertices + 0.1),
            ),
            *sample.parts[1:],
        ),
    )
    with pytest.raises(ValueError, match="geometry differs"):
        final_poses(episode, modified)


def test_summary_macro_average_and_errors():
    rows = [
        dict(sample_id="a", status="scored", SCD=2, PA=0.5, SR=0),
        dict(sample_id="b", status="scored", SCD=0, PA=1, SR=1),
        dict(sample_id="c", status="error", error="missing", SCD=None, PA=None, SR=None),
    ]
    summary = summarize(rows, {})
    assert summary["SCD"] == 1 and summary["PA"] == 0.75 and summary["SR"] == 0.5
    assert summary["denominator"] == 2 and summary["status"] == "incomplete"


@pytest.mark.parametrize("broken_pool", [False, True])
@pytest.mark.parametrize("metadata_name", ["meta.json", "run.json"])
def test_run_missing_final_is_recorded_and_outputs_replaced(
    tmp_path, monkeypatch, broken_pool, metadata_name
):
    from concurrent.futures import ThreadPoolExecutor

    from assembly_world_agent.artifacts import config_id
    from assembly_world_agent.evaluation import runner

    class BrokenPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            raise RuntimeError("Worker pool unavailable")

    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", BrokenPool if broken_pool else ThreadPoolExecutor
    )

    identity = dict(
        protocol="assembly-preparation-v1",
        revision="a" * 40,
        dataset="ikea-manual",
        preparation={"sampling_seed": 0, "initialization_seed": 0},
    )
    config = dict(
        version=1,
        config_id=config_id(identity),
        identity=identity,
        samples={"Test/a": {}, "Test/b": {}},
    )
    (tmp_path / metadata_name).write_text(json.dumps({"config": config}))
    (tmp_path / "metrics.json").write_text("original scheduling results")
    monkeypatch.setattr(
        runner,
        "load_samples",
        lambda *a, **kw: iter(
            [
                type("Source", (), {"sample_id": "Test/a"})(),
                type("Source", (), {"sample_id": "Test/b"})(),
            ]
        ),
    )

    def score(directory, *args):
        if directory.name == "Test--a":
            raise FileNotFoundError("final.episode.zip")
        return dict(sample_id="Test/b", status="scored", SCD=0.0, PA=1.0, SR=1)

    monkeypatch.setattr(runner, "evaluate_sample", score)
    for _ in range(2):
        summary = runner.evaluate_run(tmp_path)
        assert summary["error_samples"] == (2 if broken_pool else 1)
        assert summary["scored_samples"] == (0 if broken_pool else 1)
        if broken_pool:
            assert summary["SCD"] is summary["PA"] is summary["SR"] is None
        assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "metrics.json").read_text() == "original scheduling results"
