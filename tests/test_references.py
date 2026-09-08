"""Reference isolation, source ordering and actual multimodal subprocess inputs."""

import asyncio
import base64
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from assembly_world_agent.adapters import reference_pages
from assembly_world_agent.experiments import agents, runner


def bench_row():
    def page(step, view="000", kind="diagram"):
        return {
            "step_id": step,
            "view_id": view,
            "kind": kind,
            "source_file": f"{kind}/{view}_{step}.png",
            "image": {"bytes": b"image"},
        }

    return {
        "object_id": "test",
        "steps": [{"step_id": i} for i in (0, 2, 10)],
        "manual_pages": [
            page(10),
            page(2),
            page(0),
            page(11),
            page(10, "001"),
            page(10, kind="arrow"),
        ],
    }


def test_bench_numeric_order_and_reference_isolation():
    row = bench_row()
    assert reference_pages("assemblybench", row, "none") == []
    assert [p["step_id"] for p in reference_pages("assemblybench", row, "final-image")] == [10]
    pages = reference_pages("assemblybench", row, "manualbook")
    assert [p["step_id"] for p in pages] == [0, 2, 10]
    assert all(p["kind"] == "diagram" and p["view_id"] == "000" for p in pages)


@pytest.mark.parametrize("duplicate", [False, True])
def test_bench_missing_or_ambiguous_final_image(duplicate):
    row = bench_row()
    if duplicate:
        row["manual_pages"].append(row["manual_pages"][0])
    else:
        row["manual_pages"].pop(0)
    with pytest.raises(ValueError, match="expected one view-0"):
        reference_pages("assemblybench", row, "final-image")


def test_ikea_cover_and_original_order():
    pages = [{"manual_id": m, "page_index": p} for m, p in [("0", 0), ("0", 1), ("1", 0)]]
    row = {"manual_pages": pages}
    assert reference_pages("ikea-manual", row, "final-image") == pages[:1]
    assert reference_pages("ikea-manual", row, "manualbook") == pages
    with pytest.raises(ValueError):
        reference_pages("ikea-manual", {"manual_pages": pages[1:]}, "final-image")
    with pytest.raises(ValueError, match="missing reference"):
        reference_pages("ikea-manual", {}, "manualbook")


def test_undefined_dataset_mode_is_explicit():
    with pytest.raises(ValueError, match="does not define"):
        reference_pages("fantastic-breaks", {}, "final-image")
    assert reference_pages("fantastic-breaks", {}, "none") == []


@pytest.mark.parametrize(
    "mode,count", [("none", 0), ("final-image", 1), ("manualbook", 2), (None, 2)]
)
def test_explicit_pages_bytes_order_and_legacy(tmp_path, mode, count):
    source = tmp_path / "source"
    source.mkdir()
    for name, color in [("02.webp", "blue"), ("01.jpg", "red")]:
        Image.new("RGB", (8, 8), color).save(source / name)
    output = tmp_path / "output"
    output.mkdir()
    inputs = {}
    options = {"manual": str(source)}
    if mode is not None:
        options["reference_mode"] = mode
    pages = runner.manual_pages(inputs, {"options": options}, output)
    assert len(pages) == count
    if mode in {"final-image", "manualbook"}:
        for path, original in zip(pages, sorted(source.iterdir())):
            assert Path(path).read_bytes() == original.read_bytes()
        assert inputs["manual"]["reference_mode"] == mode
        assert inputs["manual"]["pages"][0]["source_file"] == "01.jpg"
    if mode is None:
        assert "reference_mode" not in inputs["manual"]
        assert all(Path(p).suffix == ".png" for p in pages)


def test_none_never_opens_manual_or_dataset(tmp_path, monkeypatch):
    import datasets

    def forbidden(*args, **kwargs):
        pytest.fail("No-reference mode accessed the dataset")

    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    inputs = {}
    assert (
        runner.manual_pages(
            inputs,
            {
                "options": {"reference_mode": "none", "manual": "/missing"},
                "config": {"identity": {}},
            },
            tmp_path,
        )
        == []
    )


def test_missing_provenance_fails_for_new_modes(tmp_path):
    with pytest.raises(ValueError, match="require pinned"):
        runner.manual_pages({}, {"options": {"reference_mode": "final-image"}}, tmp_path)
    assert runner.manual_pages({}, {"options": {}}, tmp_path) == []


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_cli_image_transport_and_unchanged_permissions(tmp_path, monkeypatch, provider, count):
    paths = []
    for i in range(count):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(path)
        paths.append(str(path))
    options = {"agent": provider, "model": "test"}
    args = agents.command(options, tmp_path, tmp_path / "bridge", paths)
    base = agents.command(options, tmp_path, tmp_path / "bridge")
    if provider == "codex":
        attached = [args[i + 1] for i, arg in enumerate(args) if arg == "--image"]
        assert attached == paths
        stripped = list(args)
        for path in paths:
            index = stripped.index("--image")
            del stripped[index : index + 2]
        assert stripped == base
    else:
        assert args == base + (["--input-format", "stream-json"] if count else [])
    script = tmp_path / "fake.py"
    captured = tmp_path / "stdin.txt"
    script.write_text(
        "import sys,json\n"
        f"open({str(captured)!r},'w').write(sys.stdin.read())\n"
        "print(json.dumps({'type':'result','result':'done'}))\n"
    )
    monkeypatch.setattr(agents, "command", lambda *args: [sys.executable, str(script)])
    result = asyncio.run(
        agents.run_agent(options, tmp_path, None, "same task", tmp_path / "log", images=paths)
    )
    assert result["status"] == "completed"
    if provider == "claude" and count:
        message = json.loads(captured.read_text())
        content = message["message"]["content"]
        assert content == agents.prompt_content("same task", paths)
        assert content[0] == {"type": "text", "text": "same task"}
        assert [base64.b64decode(b["source"]["data"]) for b in content[1:]] == [
            Path(p).read_bytes() for p in paths
        ]
    else:
        assert captured.read_text() == "same task"


@pytest.mark.parametrize("mode", [None, "none", "final-image", "manualbook"])
def test_resume_preserves_reference_mode(tmp_path, monkeypatch, mode):
    options = {"logs": str(tmp_path), "concurrency": 1}
    if mode is not None:
        options["reference_mode"] = mode
    source = runner.create_run(options, None, {"a": {"sample_id": "a"}})
    observed = []

    async def doctor(options):
        assert options.get("reference_mode") == mode
        return {}

    async def schedule(path):
        observed.append(runner.read_json(path / "run.json"))
        return 0

    monkeypatch.setattr(runner, "doctor", doctor)
    monkeypatch.setattr(runner, "schedule", schedule)
    assert asyncio.run(runner.launch({"concurrency": 2}, source=source)) == 0
    assert observed[0]["options"].get("reference_mode") == mode


@pytest.mark.parametrize("mode,count", [("none", 0), ("final-image", 1), ("manualbook", 2)])
def test_runner_forwards_and_records_only_selected_images(tmp_path, mode, count):

    manual = tmp_path / "manual"
    manual.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(manual / f"{i}.png")
    initial = tmp_path / "initial.zip"
    initial.write_bytes(b"initial")
    options = {"logs": str(tmp_path), "reference_mode": mode, "manual": str(manual)}
    run = runner.create_run(
        options, None, {"a": {"initial_path": str(initial), "sha256": runner.sha256(b"initial")}}
    )
    meta = runner.read_json(run / "run.json")
    received = []

    class Browser:
        loaded = False

        def __init__(self, *args):
            pass

        async def start(self, url):
            pass

        def mcp_command(self):
            return ["unused"]

        async def close(self):
            pass

    async def agent(options, root, bridge, prompt, conversation, *, images=()):
        transport = runner.read_json(bridge)
        assert transport["manual"] == list(images)
        assert len(images) == count
        received.append(prompt)
        return {"status": "completed", "final_answer": "done"}

    asyncio.run(
        runner.execute_sample(
            run, "a", meta, episode_url="unused", browser_factory=Browser, agent=agent
        )
    )
    assert received == [runner.DEFAULT_TASK + "\n" + runner.REFERENCE_PROTOCOL]
    record = json.loads((run / "samples/a/conversation.jsonl").read_text().splitlines()[0])
    if count:
        assert len(record["content"]) == count + 1
        assert all(block["type"] == "image" for block in record["content"][1:])
    else:
        assert record["content"] == received[0]
    inputs = runner.read_json(run / "samples/a/input.json")
    assert inputs["manual"]["reference_mode"] == mode
    assert len(inputs["manual"]["pages"]) == count


def test_hf_selection_uses_adapter_and_preserves_provenance(tmp_path, monkeypatch):
    import io

    import datasets

    stream = io.BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="PNG")
    row = bench_row()
    for page in row["manual_pages"]:
        page["image"] = {"bytes": stream.getvalue(), "path": None}
    data = datasets.Dataset.from_list([row])
    features = data.features.copy()
    features["manual_pages"].feature["image"] = datasets.Image(decode=False)
    data = data.cast(features)
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: data)
    inputs = {"sample_id": "test"}
    meta = {
        "options": {"reference_mode": "final-image"},
        "config": {"identity": {"dataset": "AssemblyWorld/assemblybench", "revision": "a" * 40}},
    }
    pages = runner.manual_pages(inputs, meta, tmp_path)
    assert len(pages) == 1
    record = inputs["manual"]["pages"][0]
    assert record["step_id"] == 10 and record["view_id"] == "000"
    assert record["kind"] == "diagram"
    assert inputs["manual"]["revision"] == "a" * 40
    assert record["source_sha256"] == record["image_sha256"]
