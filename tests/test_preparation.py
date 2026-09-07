from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from assembly_world_agent import PreparationConfig, adapt_sample, prepare_sample
from assembly_world_agent.utils import apply_pose, rotation_matrix, separated, transform_points


@pytest.mark.parametrize(
    "dataset",
    [
        "ikea-manual",
        "partnet-manualpa",
        "breaking-bad-volume-constrained",
        "assemblybench",
        "fantastic-breaks",
    ],
)
def test_world_and_inverse_reconstruction(dataset, row, assemblybench_row):
    raw = assemblybench_row if dataset == "assemblybench" else row
    before = deepcopy(raw)
    source = adapt_sample(dataset, raw, revision="fixture")
    task = prepare_sample(source)
    assert raw == before
    assert task.revision == "fixture"
    assert task.source_splits == raw["source_splits"]
    assembled, boxes = [], []
    for original, part in zip(source.parts, task.parts):
        gt = apply_pose(part.mesh.vertices, part.gt_pose)
        expected = apply_pose(original.mesh.vertices, original.assembled_pose)
        np.testing.assert_allclose(transform_points(gt, task.world_to_source), expected, atol=1e-12)
        np.testing.assert_allclose(gt, transform_points(expected, task.source_to_world), atol=1e-12)
        np.testing.assert_array_equal(
            original.mesh.vertices, raw["parts"][len(assembled)]["vertices"]
        )
        assert part.mesh.faces == original.mesh.faces
        assert part.points.shape == (1000, 3)
        assert np.isfinite(part.points).all()
        for pose in (part.gt_pose, part.initial_pose):
            np.testing.assert_allclose(np.linalg.norm(pose.quaternion), 1, atol=1e-12)
        if len(original.mesh.normals):
            expected_normals = (
                original.mesh.normals
                @ rotation_matrix(original.assembled_pose.quaternion).T
                @ source.source_to_z_up.T
            )
            np.testing.assert_allclose(
                part.mesh.normals @ rotation_matrix(part.gt_pose.quaternion).T,
                expected_normals,
                atol=1e-12,
            )
        initial = apply_pose(part.mesh.vertices, part.initial_pose)
        assert initial[:, 2].min() == pytest.approx(0, abs=1e-12)
        boxes.append(np.array([initial[:, :2].min(0), initial[:, :2].max(0)]))
        assembled.append(gt)
    points = np.concatenate(assembled)
    np.testing.assert_allclose((points.max(0) + points.min(0))[:2] / 2, 0, atol=1e-12)
    assert points[:, 2].min() == pytest.approx(0, abs=1e-12)
    assert all(
        separated(box, other, task.config.min_gap - 1e-12)
        for i, box in enumerate(boxes)
        for other in boxes[:i]
    )


def test_seeds_and_part_iteration_order_are_independent(row):
    source = adapt_sample("ikea-manual", row, revision="fixture")
    first = prepare_sample(source)
    repeat = prepare_sample(source)
    changed_initial = prepare_sample(source, PreparationConfig(initialization_seed=7))
    changed_sampling = prepare_sample(source, PreparationConfig(sampling_seed=7))
    reordered = prepare_sample(replace(source, parts=tuple(reversed(source.parts))))
    reversed_parts = {p.part_id: p for p in reordered.parts}
    for a, b, c, d in zip(first.parts, repeat.parts, changed_initial.parts, changed_sampling.parts):
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_allclose(
            apply_pose(a.points, a.gt_pose), apply_pose(c.points, c.gt_pose), atol=1e-12
        )
        np.testing.assert_array_equal(a.initial_pose.position, b.initial_pose.position)
        np.testing.assert_array_equal(a.initial_pose.position, d.initial_pose.position)
        np.testing.assert_array_equal(a.points, reversed_parts[a.part_id].points)
        np.testing.assert_array_equal(
            a.initial_pose.position, reversed_parts[a.part_id].initial_pose.position
        )
        assert not np.array_equal(a.points, d.points)
        assert not np.array_equal(a.mesh.vertices, c.mesh.vertices)
        np.testing.assert_array_equal(a.initial_pose.position, np.zeros(3))
        np.testing.assert_array_equal(a.initial_pose.quaternion, [1, 0, 0, 0])


@pytest.mark.parametrize(
    "dataset,axis",
    [
        ("ikea-manual", 1),
        ("partnet-manualpa", 1),
        ("breaking-bad-volume-constrained", 2),
        ("assemblybench", 2),
    ],
)
def test_source_up_maps_to_positive_z(dataset, axis, row, assemblybench_row):
    source = adapt_sample(
        dataset, assemblybench_row if dataset == "assemblybench" else row, revision="fixture"
    )
    np.testing.assert_array_equal(source.source_to_z_up @ np.eye(3)[axis], [0, 0, 1])
    assert np.linalg.det(source.source_to_z_up) == 1


def test_resource_and_fracture_identity_preservation(row, assemblybench_row):
    broken = adapt_sample("breaking-bad-volume-constrained", row, revision="fixture")
    assert broken.sample_id == row["sample_id"]
    assert broken.manual == [] and broken.manual_pages == ()
    bench = adapt_sample("assemblybench", assemblybench_row, revision="fixture")
    assert bench.manual == assemblybench_row["manual"]
    assert bench.annotations["motions"] == assemblybench_row["motions"]
    assert bench.steps[0]["manual_page_index"] is None
    # Pose is from the last assembly step, not the first motion frame.
    np.testing.assert_array_equal(bench.parts[0].assembled_pose.position, [1, 2, 3])


@pytest.mark.parametrize(
    "mutation", ["missing_part", "missing_step", "missing_view", "zero_quaternion"]
)
def test_missing_or_invalid_assemblybench_pose_is_rejected(assemblybench_row, mutation):
    if mutation == "missing_part":
        del assemblybench_row["poses"]["0"]["1"]["01"]
    elif mutation == "missing_step":
        del assemblybench_row["poses"]["0"]["1"]
    elif mutation == "missing_view":
        assemblybench_row["poses"] = {}
    else:
        assemblybench_row["poses"]["0"]["1"]["01"][3:] = [0] * 4
    with pytest.raises(ValueError):
        adapt_sample("assemblybench", assemblybench_row, revision="fixture")


@pytest.mark.parametrize(
    "mutation", ["nan", "bad_index", "float_index", "duplicate", "empty", "line"]
)
def test_invalid_mesh_is_rejected(row, mutation):
    if mutation == "nan":
        row["parts"][0]["vertices"][0][0] = float("nan")
    elif mutation == "bad_index":
        row["parts"][0]["faces"][0][0] = 1000
    elif mutation == "float_index":
        row["parts"][0]["faces"][0][0] = 0.5
    elif mutation == "duplicate":
        row["parts"][1]["part_id"] = "00"
    elif mutation == "empty":
        row["parts"][0]["faces"] = []
    else:
        row["parts"][0]["vertices"] = [[float(i), 0.0, 0.0] for i in range(8)]
    with pytest.raises(ValueError):
        prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps_points": 0},
        {"fps_points": 5000},
        {"surface_points": 4.5},
        {"sampling_seed": -1},
        {"min_gap": 0},
        {"min_gap": float("nan")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PreparationConfig(**kwargs)


def test_zero_assembled_scale_is_rejected(row):
    for part in row["parts"]:
        part["vertices"] = [[0.0, 0.0, 0.0] for _ in part["vertices"]]
    with pytest.raises(ValueError, match="Degenerate assembled"):
        prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))


def test_initial_geometry_is_independent_of_target_poses(row):
    from scipy.spatial.transform import Rotation

    from assembly_world_agent.utils import make_pose

    source = adapt_sample("ikea-manual", row, revision="fixture")
    changed = replace(
        source,
        parts=tuple(
            replace(
                part,
                assembled_pose=make_pose(
                    np.array([i * 4.0, -3.0, 7.0]),
                    Rotation.from_rotvec([0.4 + i, -0.7, 0.2]).as_matrix(),
                ),
            )
            for i, part in enumerate(source.parts)
        ),
    )
    first, second = prepare_sample(source), prepare_sample(changed)
    for a, b in zip(first.parts, second.parts):
        np.testing.assert_array_equal(a.mesh.vertices, b.mesh.vertices)
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.mesh.normals, b.mesh.normals)
        np.testing.assert_array_equal(a.initial_pose.position, [0, 0, 0])
        np.testing.assert_array_equal(a.initial_pose.quaternion, [1, 0, 0, 0])


def test_source_coordinate_reexpression_preserves_initial_scene(row):
    from scipy.spatial.transform import Rotation

    from assembly_world_agent.utils import make_pose

    source = adapt_sample("ikea-manual", row, revision="fixture")
    parts = []
    for i, part in enumerate(source.parts):
        rotation = Rotation.from_rotvec([0.3, -0.5 - i, 0.8]).as_matrix()
        offset = np.array([3.0, -2.0, 5.0])
        target_rotation = rotation_matrix(part.assembled_pose.quaternion) @ rotation.T
        parts.append(
            replace(
                part,
                mesh=replace(
                    part.mesh,
                    vertices=part.mesh.vertices @ rotation.T + offset,
                    normals=part.mesh.normals @ rotation.T,
                ),
                assembled_pose=make_pose(
                    part.assembled_pose.position - target_rotation @ offset, target_rotation
                ),
            )
        )
    first, second = prepare_sample(source), prepare_sample(replace(source, parts=tuple(parts)))
    for a, b in zip(first.parts, second.parts):
        np.testing.assert_allclose(a.mesh.vertices, b.mesh.vertices, atol=1e-11)
        np.testing.assert_allclose(
            apply_pose(a.mesh.vertices, a.gt_pose),
            apply_pose(b.mesh.vertices, b.gt_pose),
            atol=1e-11,
        )


def test_geometry_only_preparation_preserves_exact_task(row, monkeypatch):
    from assembly_world_agent import adapt_sample, preparation, prepare_sample
    from assembly_world_agent.episodes import _obj

    source = adapt_sample("ikea-manual", row, revision="fixture")
    full = prepare_sample(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Result export must not sample point clouds")

    monkeypatch.setattr(preparation, "sample_surface", forbidden)
    monkeypatch.setattr(preparation, "farthest_point_sample", forbidden)
    geometry = prepare_sample(source, sample_points=False)
    for a, b in zip(full.parts, geometry.parts):
        assert _obj(a) == _obj(b)
        assert np.array_equal(a.gt_pose.position, b.gt_pose.position)
        assert np.array_equal(a.gt_pose.quaternion, b.gt_pose.quaternion)
        assert b.points.shape == (0, 3)
