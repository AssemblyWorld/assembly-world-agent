import io
import json
import os
import shutil
import zipfile

import numpy as np
import pytest
from PIL import Image

from assembly_world_agent import adapt_sample, export_episode, prepare_sample
from assembly_world_agent.episodes import sha256
from assembly_world_agent.vis import render_episode
from assembly_world_agent.vis.replay import observation, read_episode, text_pages

pytest.importorskip("mujoco")


def test_archive_read_and_checksum_rejection(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    path = export_episode(sample, tmp_path / "initial.zip").path
    ep = read_episode(path)
    assert ep["calls"] == [] and list(ep["states"]) == [0]
    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    files["frames.bin"] += b"bad"
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="checksum"):
        read_episode(tmp_path / "bad.zip")


def test_bad_binary_size_with_matching_hash(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    path = export_episode(sample, tmp_path / "initial.zip").path
    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    files["frames.bin"] += b"extra"
    manifest = json.loads(files["manifest.json"])
    manifest["hashes"]["frames.bin"] = sha256(files["frames.bin"])
    files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="binary state size"):
        read_episode(tmp_path / "bad.zip")


def test_json_pagination_preserves_long_returns():
    call = {
        "arguments": {"id": "part-0001"},
        "result": {"items": list(range(250)), "last": "END_MARKER"},
    }
    pages = text_pages(call)
    assert len(pages) > 5 and all(len(page) <= 29 for page in pages)
    assert "END_MARKER" in "\n".join(pages[-1])
    assert "part-0001" in "\n".join(pages[0])


def test_recorded_observation_and_missing_reference():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(stream, format="PNG")
    call = {"result": {"content": [{"type": "image", "observation": "image.png"}]}}
    image = observation({"files": {"observations/image.png": stream.getvalue()}}, call)
    assert image.size == (8, 6) and image.getpixel((0, 0)) == (255, 0, 0)
    with pytest.raises(ValueError, match="Missing observation"):
        observation({"files": {}}, call)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg is an optional system dependency"
)
def test_video_encoding_without_gl(row, tmp_path, monkeypatch):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    path = export_episode(sample, tmp_path / "initial.zip").path
    before = path.read_bytes()
    ep = read_episode(path)
    ep["calls"] = [
        {
            "index": 0,
            "name": "capture_scene",
            "arguments": {},
            "status": "success",
            "actor": "agent",
            "timestamp": "fixture",
            "state_index": 0,
            "trace_start": 0,
            "trace_end": 0,
            "result": {},
        }
    ]
    monkeypatch.setattr("assembly_world_agent.vis.replay.read_episode", lambda _: ep)
    monkeypatch.setattr(
        "assembly_world_agent.vis.replay.observation",
        lambda *_: Image.new("RGB", (1024, 768), "red"),
    )
    from assembly_world_agent.vis.replay import compose

    scene_colors = []

    def track_scene(scene, *args):
        scene_colors.append(scene.getpixel((0, 0)))
        return compose(scene, *args)

    monkeypatch.setattr("assembly_world_agent.vis.replay.compose", track_scene)

    class Renderer:
        def __init__(self, episode):
            pass

        def render(self, frame):
            return Image.new("RGB", (1024, 768), "#b0c0d0")

        def close(self):
            pass

    monkeypatch.setattr("assembly_world_agent.vis.replay.StateRenderer", Renderer)
    result = render_episode(path, tmp_path / "replay", fps=4)
    assert result["video_frames"] == 8 and result["duration_seconds"] == 2
    assert scene_colors == [(176, 192, 208)] * 2
    assert result["original_observations_displayed"] == 0
    assert result["calls"] == 1 and path.read_bytes() == before
    assert b"ftyp" in (tmp_path / "replay.mp4").read_bytes()[:32]
    with Image.open(tmp_path / "replay.gif") as image:
        assert image.format == "GIF" and image.size == (1280, 720)
        image.verify()
    with pytest.raises(FileExistsError):
        render_episode(path, tmp_path / "replay")


def test_reject_invalid_render_options(tmp_path):
    with pytest.raises(ValueError, match="Formats"):
        render_episode(tmp_path / "missing.zip", tmp_path / "out", formats=["avi"])
    with pytest.raises(ValueError, match="Positive"):
        render_episode(tmp_path / "missing.zip", tmp_path / "out", fps=0)


@pytest.mark.skipif(os.environ.get("AWA_TEST_GL") != "1", reason="Needs an OpenGL context")
def test_native_projection_preserves_square_aspect():
    import mujoco

    from assembly_world_agent.vis.replay import StateRenderer

    xml = b'<mujoco><worldbody><geom type="box" size=".5 .01 .5" rgba="1 1 1 1"/></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml.decode())
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, mujoco.MjData(model), state, spec)
    renderer = StateRenderer(
        {
            "files": {"world/model.xml": xml},
            "manifest": {"stateSpec": spec, "stateSize": len(state)},
        }
    )
    try:
        pixels = np.array(
            renderer.render(
                {
                    "integration": state,
                    "camera": {"position": [0.0, -3.0, 0.0], "target": [0.0, 0.0, 0.0]},
                }
            )
        )
    finally:
        renderer.close()
    y, x = np.where(pixels.min(axis=2) > 100)
    assert len(x) > 1000
    assert abs((x.max() - x.min()) / (y.max() - y.min()) - 1) < 0.01
