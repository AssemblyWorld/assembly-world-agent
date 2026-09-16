import base64
import gzip
import json

import numpy as np
import pytest

from assembly_world_agent import adapt_sample, prepare_sample
from assembly_world_agent.vis import benchmark as preview
from assembly_world_agent.vis.results import packed


@pytest.fixture
def sample(row):
    return prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"), sample_points=False)


def _sphere_mesh(n=60):
    """A dense UV sphere with 2 n^2 triangles."""
    theta = np.linspace(0, np.pi, n + 1)
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vertices = np.array(
        [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in theta for p in phi]
    )
    triangles = []
    for i in range(n):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            triangles.append([a, c, d])
            triangles.append([a, d, b])
    return vertices, np.array(triangles)


def test_decimate_mesh_respects_budget_and_indices():
    vertices, triangles = _sphere_mesh()
    assert len(triangles) == 7200
    same_v, same_t, decimated = preview.decimate_mesh(vertices, triangles, limit=10_000)
    assert not decimated and np.array_equal(same_t, triangles)
    v, t, decimated = preview.decimate_mesh(vertices, triangles, limit=1000)
    assert decimated and 0 < len(t) <= 1000
    assert t.min() >= 0 and t.max() < len(v)
    assert not np.any((t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 0] == t[:, 2]))
    assert len(np.unique(np.sort(t, axis=1), axis=0)) == len(t)
    assert np.linalg.norm(v, axis=1).max() <= 1.0 + 1e-9
    assert np.all(np.linalg.norm(v, axis=1) > 0.5)


def test_part_geometry_marks_decimation(sample):
    part = sorted(sample.parts, key=lambda p: p.part_id)[0]
    mesh = preview.part_geometry(part)
    assert mesh["kind"] == "mesh" and mesh["decimated"] is False
    assert mesh["source_triangles"] == len(mesh["triangles"]) // 3 > 0
    small = preview.part_geometry(part, limit=1)
    assert small["kind"] == "mesh" and small["decimated"] is True
    assert small["source_triangles"] == mesh["source_triangles"]
    assert len(small["triangles"]) // 3 <= mesh["source_triangles"]
    assert max(small["triangles"], default=-1) < len(small["vertices"])


def test_shape_row_orders_parts_and_records_both_poses(sample):
    meta = dict(sample_id="Test/object", parts=len(sample.parts))
    row = preview.shape_row(meta, sample)
    assert [p["id"] for p in row["parts"]] == sorted(p.part_id for p in sample.parts)
    assert len(row["initial"]) == len(row["gt"]) == len(sample.parts)
    assert all(len(pose) == 7 for pose in row["initial"] + row["gt"])
    for pose in row["initial"] + row["gt"]:
        assert np.linalg.norm(pose[3:]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="benchmark says"):
        preview.shape_row(dict(meta, parts=99), sample)
    raw = gzip.decompress(base64.b64decode(packed(row)))
    assert json.loads(raw) == json.loads(json.dumps(row))


def _block(name, source, mode, samples, **extra):
    return dict(
        name=name,
        source=source,
        dataset=source,
        repo_id=f"AssemblyWorld/{source}",
        revision="a" * 40,
        config_id="prep",
        data=name,
        reference_mode=mode,
        prompt_file=f"{name}/task.txt",
        prompt_sha256="x",
        samples=samples,
        **extra,
    )


def test_shape_metadata_merges_paired_blocks_and_statistics():
    pn = {"1": dict(category="chair", parts=5, band="low", sha256="x")}
    fb = {"00/1": dict(category="00", parts=2, band=None, sha256="y")}
    benchmark = dict(
        sources=["partnet-manualpa", "fantastic-breaks"],
        blocks=[
            _block("pn-none", "partnet-manualpa", "none", pn),
            _block("pn-image", "partnet-manualpa", "final-image", pn),
            _block("fb", "fantastic-breaks", "none", fb),
        ],
    )
    shapes = preview.shape_metadata(benchmark)
    assert [s["sample_id"] for s in shapes] == ["1", "00/1"]
    assert shapes[0]["blocks"] == ["pn-none", "pn-image"]
    assert shapes[0]["reference_modes"] == ["none", "final-image"]
    assert shapes[1]["blocks"] == ["fb"] and shapes[1]["band"] is None
    stats = preview.statistics(benchmark, shapes)
    assert stats["evaluations"] == 3 and stats["distinct_shapes"] == 2
    assert stats["sources"]["partnet-manualpa"]["band_counts"] == {"low": 1}
    assert [b["samples"] for b in stats["blocks"]] == [1, 1, 1]


def test_write_page_embeds_manifest_payloads_and_bundle(tmp_path):
    page = dict(benchmark="demo", statistics={}, shapes=[])
    output = preview.write_page(
        tmp_path / "out" / "preview.html", page, [packed({"a": 1}), packed({"b": 2})]
    )
    text = output.read_text()
    assert text.startswith("<!doctype html>") and text.rstrip().endswith("</body></html>")
    assert 'id="manifest"' in text and 'id="shape-0"' in text and 'id="shape-1"' in text
    assert "window.ResultThree" in text and "DecompressionStream" in text
    assert "AssemblyWorldBench — demo" in text
    assert not list((tmp_path / "out").glob("*.html.*"))


def test_compact_mesh_reindexes_shared_pools(sample):
    part = sorted(sample.parts, key=lambda p: p.part_id)[0]
    geometry = preview.part_geometry(part)
    vertices = np.asarray(geometry["vertices"])
    triangles = np.asarray(geometry["triangles"])
    assert triangles.min() >= 0 and triangles.max() < len(vertices)
    assert len(vertices) == len(np.unique(triangles))
    assert len(vertices) <= len(part.mesh.vertices)
    pool = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]]
    assert preview.compact_mesh(pool, [0, 1, 2]) == ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 1, 2])
    compact_vertices, compact_triangles = preview.compact_mesh(pool, [3, 1, 2])
    assert compact_vertices == [[1, 0, 0], [0, 1, 0], [9, 9, 9]] and compact_triangles == [2, 0, 1]
    assert (
        preview.compact_mesh([[0.12345678, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 1, 2])[0][0][0]
        == 0.123457
    )
