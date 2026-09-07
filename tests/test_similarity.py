from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from assembly_world_agent import adapt_sample, prepare_sample
from assembly_world_agent.adapters.equivalence import parse_source_equivalence
from assembly_world_agent.similarity import (
    SimilarityConfig,
    groups_from_distances,
    inspect_similarity_pair,
    resolve_equivalence,
)
from assembly_world_agent.utils import make_pose


def test_transitive_groups_and_inclusive_boundary():
    distances = [[0, 0.1, 0.25], [0.1, 0, 0.1], [0.25, 0.1, 0]]
    groups, diagnostics = groups_from_distances(["a", "b", "c"], distances, 0.1)
    assert groups == [["a", "b", "c"]]
    assert diagnostics[0]["max_chamfer"] == 0.25
    assert diagnostics[0]["above_threshold_pairs"] == [{"part_ids": ["a", "c"], "chamfer": 0.25}]
    assert groups_from_distances(["a", "b", "c"], distances, np.nextafter(0.1, 0))[0] == [
        ["a"],
        ["b"],
        ["c"],
    ]


def test_source_annotation_mapping_missing_and_composites(row):
    row["parts"][0]["annotation_part_id"] = "0"
    row["parts"][1]["annotation_part_id"] = "1"
    row["geometric_equivalence_relation"] = {"0": ["1"], "0,1": ["0,1"]}
    source = adapt_sample("ikea-manual", row, revision="fixture")
    result = resolve_equivalence(prepare_sample(source), config=SimilarityConfig("source"))
    assert result["groups"] == [sorted(p.part_id for p in source.parts)]
    assert not result["source_annotation_missing"]
    assert result["ignored_composite_self_groups"] == ["0,1"]
    assert (
        source.annotations["geometric_equivalence_relation"]
        == row["geometric_equivalence_relation"]
    )
    missing = parse_source_equivalence(source.parts, {})
    assert missing["missing"] and len(missing["groups"]) == 2
    for relation in [{"0": ["99"]}, {"99": None}, {"0,1": ["0"]}, {"0": "1"}]:
        with pytest.raises(ValueError):
            parse_source_equivalence(source.parts, {"geometric_equivalence_relation": relation})


def test_geometry_independent_of_pose_metadata_order_and_rigid_placement(row):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    a = sample.parts[0]
    rotation = Rotation.from_euler("xyz", [41, 19, -123], degrees=True).as_matrix()
    offset = np.array([7, -5, 4])
    b = replace(
        a,
        part_id="copy",
        points=a.points @ rotation.T + offset,
        mesh=replace(a.mesh, vertices=a.mesh.vertices @ rotation.T + offset),
    )
    sample = replace(sample, parts=(a, b), source_equivalence={})
    original = resolve_equivalence(sample)
    assert original["groups"] == [sorted([a.part_id, b.part_id])]
    assert max(map(max, original["distances"])) < 1e-20
    changed = replace(
        sample,
        parts=tuple(
            replace(
                p,
                gt_pose=make_pose(np.ones(3) * 999, np.eye(3)),
                initial_pose=make_pose(np.ones(3) * -999, np.eye(3)),
                metadata={},
            )
            for p in sample.parts[::-1]
        ),
        annotations={"geometric_equivalence_relation": "irrelevant"},
    )
    assert resolve_equivalence(changed) == original
    assert resolve_equivalence(sample, config=SimilarityConfig("source"))["groups"] == [
        [pid] for pid in sorted([a.part_id, b.part_id])
    ]
    pair = inspect_similarity_pair(sample, a.part_id, b.part_id)
    assert pair["alignment"]["chamfer"] == pytest.approx(original["distances"][0][1], abs=1e-20)
    scaled = replace(b, points=b.points * 2, mesh=replace(b.mesh, vertices=b.mesh.vertices * 2))
    different = resolve_equivalence(replace(sample, parts=(a, scaled)))
    assert len(different["groups"]) == 2


def test_grouping_uses_registration_not_reflections(row):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    rng = np.random.default_rng(132)
    # Four asymmetric lobes produce a chiral shape that cannot rigidly match its mirror.
    cloud = np.concatenate(
        [
            rng.normal(size=(250, 3)) * 0.015 + c
            for c in [[0, 0, 0], [0.8, 0, 0], [0, 0.45, 0], [0.1, 0.15, 0.3]]
        ]
    )
    a = replace(sample.parts[0], points=cloud, mesh=replace(sample.parts[0].mesh, vertices=cloud))
    b = replace(
        a,
        part_id="mirror",
        points=cloud * [-1, 1, 1],
        mesh=replace(a.mesh, vertices=cloud * [-1, 1, 1]),
    )
    result = resolve_equivalence(replace(sample, parts=(a, b)))
    assert len(result["groups"]) == 2
    pair = inspect_similarity_pair(replace(sample, parts=(a, b)), a.part_id, b.part_id)
    assert np.linalg.det(pair["alignment"]["rotation"]) == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [{"policy": "identity"}, {"threshold": float("nan")}, {"threshold": -1}, {"threshold": True}],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SimilarityConfig(**kwargs)


def test_saved_comparison_preserves_error_rows(tmp_path):
    import json

    from assembly_world_agent.evaluation.inspection import compare_run

    (tmp_path / "meta.json").write_text(
        json.dumps({"config": {"identity": {}, "samples": {"a": {}}}})
    )
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"sample_id": "a", "status": "error", "error": "missing episode"}) + "\n"
    )
    rows = compare_run(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "error"
    assert "missing episode" in rows[0]["error"]
    (tmp_path / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="every expected"):
        compare_run(tmp_path)


def test_shape_equivalence_does_not_realign_displaced_parts(row):
    from assembly_world_agent.evaluation.geometry import score_parts

    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    a = sample.parts[0]
    b = replace(a, part_id="copy")
    result = resolve_equivalence(replace(sample, parts=(a, b)))
    assert len(result["groups"]) == 1
    cloud = a.points - a.points.mean(0)
    target = [cloud, cloud + [3, 0, 0]]
    prediction = [cloud, cloud + [3, 10, 0]]
    values, _ = score_parts(prediction, target, [[0, 1]], [a.part_id, "copy"], 1)
    assert values["PA"] == 0.5 and values["SR"] == 0


def test_repreparation_seed_and_ground_truth_do_not_change_geometry_groups(row):
    source = adapt_sample("ikea-manual", row, revision="fixture")
    first = prepare_sample(source)
    altered = replace(
        source,
        parts=tuple(
            replace(
                p,
                assembled_pose=make_pose(
                    np.array([i * 15, -20, 31]),
                    Rotation.from_euler("x", 73, degrees=True).as_matrix(),
                ),
            )
            for i, p in enumerate(source.parts)
        ),
    )
    second = prepare_sample(altered, replace(first.config, initialization_seed=879))
    a, b = resolve_equivalence(first), resolve_equivalence(second)
    assert a["groups"] == b["groups"]
    np.testing.assert_allclose(a["distances"], b["distances"], rtol=0, atol=1e-9)
