import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from assembly_world_agent.evaluation.garf import point_counts, score_poses


def test_budget_and_anchor():
    assert point_counts([1, 3]).tolist() == [1260, 3740]
    assert point_counts([1, 1, 1]).sum() == 5000
    with pytest.raises(ValueError):
        point_counts([0, 1])


def test_shared_rigid_transform_is_removed():
    rng = np.random.default_rng(3)
    points = [rng.normal(size=(20, 3)), rng.normal(size=(30, 3))]
    target = [
        (np.eye(3), np.zeros(3)),
        (Rotation.from_euler("XYZ", [20, 30, 40], degrees=True).as_matrix(), np.ones(3)),
    ]
    q = Rotation.from_euler("XYZ", [60, 10, 30], degrees=True).as_matrix()
    t = np.array([4, -8, 5])
    prediction = [(q @ r, q @ p + t) for r, p in target]
    metrics, _ = score_poses(points, prediction, target, 0)
    assert metrics["RMSE_R"] < 1e-12
    assert metrics["RMSE_T"] < 1e-12
    assert metrics["CD"] < 1e-24
    assert metrics["PA"] == 1


def test_rms_before_part_average_and_strict_threshold():
    points = [np.zeros((1, 3)), np.zeros((3, 3))]
    gt = [(np.eye(3), np.zeros(3)), (np.eye(3), np.ones(3))]
    pred = [(np.eye(3), np.zeros(3)), (np.eye(3), np.ones(3) + [0.3, 0, 0])]
    metrics, rows = score_poses(points, pred, gt, 0)
    assert metrics["RMSE_T"] == pytest.approx(0.3 / np.sqrt(3) / 2)
    assert metrics["PA"] == 0.5
    assert metrics["CD"] == pytest.approx(0.09)
    assert rows[1]["part_cd"] == pytest.approx(0.18)


def test_euler_wrap_not_geodesic():
    def rotation(angle):
        return Rotation.from_euler("XYZ", [0, 0, angle], degrees=True).as_matrix()

    points = [np.zeros((1, 3)), np.zeros((1, 3))]
    gt = [(np.eye(3), np.zeros(3)), (rotation(179), np.ones(3))]
    pred = [(np.eye(3), np.zeros(3)), (rotation(-179), np.ones(3))]
    metrics, _ = score_poses(points, pred, gt, 0)
    assert metrics["RMSE_R"] == pytest.approx(1 / np.sqrt(3))


def test_threshold_is_strict(monkeypatch):
    monkeypatch.setattr("assembly_world_agent.evaluation.garf.chamfer", lambda p, g: 0.01)
    poses = [(np.eye(3), np.zeros(3))]
    metrics, _ = score_poses([np.zeros((1, 3))], poses, poses, 0)
    assert metrics["PA"] == 0


def test_intrinsic_xyz_and_part_weighted_global_cd():
    x, y, z = np.deg2rad([20, 30, 40])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    points = [np.zeros((1, 3)), np.zeros((9, 3))]
    target = [(np.eye(3), np.zeros(3)), (np.eye(3), np.array([10, 0, 0]))]
    prediction = [target[0], (rx @ ry @ rz, np.array([10.1, 0, 0]))]
    metrics, _ = score_poses(points, prediction, target, 0)
    assert metrics["RMSE_R"] == pytest.approx(np.sqrt((20**2 + 30**2 + 40**2) / 3) / 2)
    # Nine points on the moved part must not give it 90% of the shape CD weight.
    assert metrics["CD"] == pytest.approx(0.01)


def test_resume_union_and_missing_archive(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from assembly_world_agent.evaluation import garf

    identity = {"dataset": "example", "revision": "a" * 40}
    monkeypatch.setattr(garf, "config_id", lambda identity: "config")
    runs = []
    for i, samples in enumerate(({"a": {}, "b": {}, "c": {}}, {"b": {}, "c": {}})):
        run = tmp_path / str(i)
        run.mkdir()
        (run / "run.json").write_text(
            json.dumps(
                {"config": {"config_id": "config", "identity": identity, "samples": samples}}
            )
        )
        directory = run / "samples" / ("a" if i == 0 else "b")
        directory.mkdir(parents=True)
        (directory / "final.episode.zip").write_bytes(b"test")
        runs.append(run)
    monkeypatch.setattr(
        garf,
        "load_samples",
        lambda *args, **kwargs: iter(
            SimpleNamespace(sample_id=sid) for sid in kwargs["sample_ids"]
        ),
    )
    monkeypatch.setattr(
        garf,
        "evaluate_sample",
        lambda d, e, i, s, seed: dict(
            sample_id=s.sample_id, status="scored", RMSE_R=1, RMSE_T=2, PA=1, CD=0
        ),
    )
    summary = garf.evaluate_runs(runs, tmp_path / "output", workers=1)
    assert summary["expected_samples"] == 3
    assert summary["scored_samples"] == 2
    assert summary["status"] == "incomplete"
    assert summary["errors"][0]["sample_id"] == "c"
    assert len((tmp_path / "output" / "metrics.jsonl").read_text().splitlines()) == 3
    (runs[1] / "samples" / "a").mkdir()
    (runs[1] / "samples" / "a" / "final.episode.zip").write_bytes(b"duplicate")
    config = json.loads((runs[1] / "run.json").read_text())
    config["config"]["samples"]["a"] = {}
    (runs[1] / "run.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Multiple final"):
        garf.evaluate_runs(runs, tmp_path / "duplicate-output")
