import base64
import copy
import gzip
import json

import numpy as np
import pytest

from assembly_world_agent import adapt_sample, export_episode, prepare_sample
from assembly_world_agent.vis import results
from assembly_world_agent.vis.replay import read_episode

pytest.importorskip("mujoco")


def test_select_final_checkpoint_and_corruption(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    good = export_episode(sample, checkpoints / "calls-000001.episode.zip").path
    final = tmp_path / "final.episode.zip"
    final.write_bytes(good.read_bytes())
    assert results.select_episode(tmp_path)[1]["path"] == final.name
    final.write_bytes(b"broken")
    ep, provenance, errors = results.select_episode(tmp_path)
    assert ep and provenance["path"] == "checkpoints/" + good.name and errors
    good.write_bytes(b"broken")
    assert results.select_episode(tmp_path)[0] is None


def test_body_pose_and_camera_filter(row, tmp_path):
    import mujoco as mj

    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    ep = read_episode(export_episode(sample, tmp_path / "initial.zip").path)
    initial = ep["states"][0]
    camera = copy.deepcopy(initial)
    camera["camera"]["position"][0] += 1
    ep["states"][1] = camera
    moved = copy.deepcopy(camera)
    files = ep["files"]
    model = mj.MjModel.from_xml_string(
        files["world/model.xml"].decode(),
        assets={
            k[6:]: v for k, v in files.items() if k.startswith("world/") and k != "world/model.xml"
        },
    )
    data = mj.MjData(model)
    mj.mj_setState(model, data, np.array(camera["integration"]), ep["manifest"]["stateSpec"])
    data.qpos[:3] = [1, 2, 3]
    data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    state = np.empty(ep["manifest"]["stateSize"])
    mj.mj_getState(model, data, state, ep["manifest"]["stateSpec"])
    moved["integration"] = state.tolist()
    ep["states"][2] = moved
    ep["calls"] = [
        dict(
            index=i, name=name, arguments={}, status=status, before_index=before, state_index=after
        )
        for i, (name, status, before, after) in enumerate(
            [
                ("get_scene", "completed", 0, 0),
                ("move_camera", "completed", 0, 1),
                ("set_object_pose", "completed", 1, 2),
                ("bad", "failed", 2, 2),
            ]
        )
    ]
    payload = results.episode_data(ep)
    assert [c["changed"] for c in payload["calls"]] == [False, True, True, False]
    assert np.allclose(
        payload["frames"]["2"]["poses"][0], [1, 2, 3, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    )
    assert np.allclose(payload["parts"][0]["geometry"]["vertices"], sample.parts[0].mesh.vertices)
    assert payload["calls"][-1]["status"] == "failed"


def test_missing_resources_isolated_and_html_safe(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    meta = {
        "config": {
            "identity": {"dataset": "ikea-manual", "revision": "fixture", "preparation": {}},
            "samples": {"a": {}, "b": {}},
        }
    }
    (run / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(results, "load_result_samples", lambda *a, **kw: iter(()))
    output = tmp_path / "out.html"
    result = results.export_results(run, output)
    assert result["samples"] == 2 and result["replays"] == result["ground_truths"] == 0
    assert set(result["errors"]) == {"a", "b"}
    content = output.read_text()
    assert 'id="sample-1"' in content and 'src="http' not in content
    dangerous = {"arguments": "</script><img onerror=alert(1)>"}
    assert json.loads(gzip.decompress(base64.b64decode(results.packed(dangerous)))) == dangerous


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_lossless_manual_pixels(mode):
    import io

    from PIL import Image

    image = Image.new("RGBA", (17, 23), (12, 53, 170, 0))
    image.putpixel((3, 5), (237, 78, 5, 128))
    image = image.convert(mode)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    encoded = results.encode_manual_image(raw)
    payload = base64.b64decode(encoded["data"].split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as restored:
        assert restored.size == image.size
        assert restored.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()
    assert "embedded_sha256" not in encoded


def test_parallel_manual_order_and_isolated_errors(tmp_path):
    import io

    from PIL import Image

    manual = tmp_path / "manualbook"
    manual.mkdir()
    stream = io.BytesIO()
    Image.new("RGB", (5, 7), "red").save(stream, format="PNG")
    raw = stream.getvalue()
    (manual / "page.png").write_bytes(raw)
    pages = [
        {"file": "page.png", "sha256": "not-verified"},
        {"file": "missing.png", "sha256": "missing"},
        {"file": "page.png", "sha256": "wrong"},
        {"file": "page.png", "sha256": "not-verified"},
    ]
    (manual / "pages.json").write_text(json.dumps({"pages": pages}))
    errors = []
    encoded = results.manual_pages(tmp_path, errors)
    assert [p["file"] for p in encoded] == [p["file"] for p in pages]
    assert encoded[0] == encoded[3]
    assert "error" in encoded[1]
    assert encoded[2]["data"] == encoded[0]["data"]
    assert "error" not in encoded[2]
    assert len(errors) == 1
    assert list(results.parallel_map(results.packed, encoded)) == [
        results.packed(page) for page in encoded
    ]


def test_result_selection_skips_resource_hashes(row, tmp_path):
    import zipfile

    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    path = export_episode(sample, tmp_path / "initial.zip").path
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files["manifest.json"])
    manifest["hashes"] = {name: "0" * 64 for name in manifest["hashes"]}
    files["manifest.json"] = json.dumps(manifest).encode()
    final = tmp_path / "final.episode.zip"
    with zipfile.ZipFile(final, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_episode(final)
    episode, provenance, errors = results.select_episode(tmp_path)
    assert episode and not errors
    assert provenance == {"path": "final.episode.zip"}


@pytest.mark.parametrize("triangle", [False, True])
def test_direct_mesh_matches_obj_roundtrip(row, triangle):
    from types import SimpleNamespace

    from assembly_world_agent.episodes import _obj, _render_mesh
    from assembly_world_agent.models import Mesh

    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    parts = sample.parts
    if triangle:
        parts = [SimpleNamespace(mesh=Mesh(np.eye(3), ((0, 1, 2),)))]
    for part in parts:
        vertices, faces = _render_mesh(part)
        assert {"vertices": vertices.tolist(), "triangles": faces.ravel().tolist()} == (
            results.obj_mesh(_obj(part))
        )
        assert results.render_geometry(part) == results.obj_mesh(_obj(part))


def test_light_ikea_loader_preserves_geometry_without_adapter(row, monkeypatch):
    import datasets

    revision = "a" * 40
    expected = prepare_sample(
        adapt_sample("ikea-manual", row, revision=revision), sample_points=False
    )
    cached = datasets.Dataset.from_list([row])
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return cached

    def forbidden(*args, **kwargs):
        raise AssertionError("IKEA result export must bypass the full adapter")

    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr(results, "load_samples", forbidden)
    sources = list(
        results.load_result_samples("ikea-manual", revision=revision, sample_ids=[row["object_id"]])
    )
    assert len(sources) == 1
    assert calls[0]["streaming"] is False and calls[0]["revision"] == revision
    assert sources[0].manual_pages == () and sources[0].annotations == {}
    actual = prepare_sample(sources[0], sample_points=False)
    for a, b in zip(actual.parts, expected.parts):
        assert results.render_geometry(a) == results.render_geometry(b)
        assert np.array_equal(a.gt_pose.position, b.gt_pose.position)
        assert np.array_equal(a.gt_pose.quaternion, b.gt_pose.quaternion)
