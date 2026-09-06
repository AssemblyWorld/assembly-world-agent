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
    monkeypatch.setattr(results, "load_samples", lambda *a, **kw: iter(()))
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
    assert encoded["embedded_sha256"] == results.sha256(payload)
