"""Native, WASM and isolated browser checks against an explicit 3DWebAgent checkout."""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright

from assembly_world_agent.artifacts import experiment, load_prepared, read_config, sample_name
from assembly_world_agent.episodes import contract_provenance
from assembly_world_agent.utils import apply_pose, separated

# This is upstream's registration adapter, not an external MCP client connection.
REGISTER = """() => {
  const tools = new Map(); globalThis.__episodeTools = tools;
  Object.defineProperty(document, 'modelContext', {configurable: true, value: {
    registerTool(tool, options) {
      tools.set(tool.name, tool);
      options.signal.addEventListener('abort', () => tools.delete(tool.name));
    },
    unregisterTool(name) { tools.delete(name); },
    getTools: async () => [...tools.values()].map(t => ({name: t.name})),
  }});
}"""


def invoke(page, name, arguments=None):
    return page.evaluate(
        """async ({name, args}) =>
        JSON.parse(await globalThis.__episodeTools.get(name).execute(args))""",
        {"name": name, "args": arguments or {}},
    )


def archive(path):
    with zipfile.ZipFile(path) as z:
        files = {name: z.read(name) for name in z.namelist()}
    return dict(
        manifest=json.loads(files["manifest.json"]),
        files=files,
        calls=[json.loads(line) for line in files["calls.jsonl"].splitlines()],
        events=[json.loads(line) for line in files["events.jsonl"].splitlines()],
    )


def command_json(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def native(checkout, operation, path):
    return command_json(
        [
            str(checkout / "python/.venv/bin/python"),
            str(checkout / "python/replay.py"),
            operation,
            str(path.resolve()),
            *(["--atol", "1e-9", "--rtol", "1e-7"] if operation == "verify" else []),
        ],
        checkout,
    )


def wasm(checkout, directory):
    return command_json(
        [
            "node",
            "--experimental-strip-types",
            str(checkout / "scripts/validate-episodes.ts"),
            str(directory.resolve()),
        ],
        checkout,
    )


def open_episode(browser, args, path, count, errors):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console", lambda message: errors.append(message.text) if message.type == "error" else None
    )
    page.on("dialog", lambda dialog: dialog.accept())
    page.add_init_script("(" + REGISTER + ")()")
    page.goto(args.base_url)
    expect(page.get_by_role("button", name="Import OBJ", exact=True)).to_be_enabled()
    page.get_by_label("Episode file", exact=True).set_input_files(path)
    expect(page.locator("[data-object-id]")).to_have_count(count)
    page.wait_for_function("globalThis.__episodeTools.has('get_state')")
    page.get_by_role("tab", name="Timeline", exact=True).click()
    assert not page.evaluate(
        "globalThis.__episodeTools.has('apply_force') || globalThis.__episodeTools.has('advance_simulation')"
    )
    return context, page


def export_run(page, path):
    with page.expect_download() as download:
        page.get_by_text("File", exact=True).click()
        page.get_by_role("button", name="Export full episode…", exact=True).click()
    download.value.save_as(path)
    return archive(path)


def capture(page, path):
    result = invoke(page, "capture_scene")
    assert result["content"][0]["type"] == "image"
    data = base64.b64decode(result["content"][0]["data"])
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG" and image.size == (1024, 768)
        image.verify()
    path.write_bytes(data)
    return data


def check_poses(state, sample, field):
    parts = sorted(sample.parts, key=lambda p: p.part_id)
    assert len(state["objects"]) == len(parts)
    for i, part in enumerate(parts):
        obj = next(o for o in state["objects"] if o["id"] == f"part-{i + 1:04d}")
        assert obj["movable"] and not obj["collisionEnabled"]
        pose = getattr(part, field)
        np.testing.assert_allclose(obj["position"], pose.position, atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(obj["quaternion"], pose.quaternion, atol=1e-9, rtol=1e-7)


def check_initial(initial, sample):
    assert initial["manifest"]["lifecycle"] == "setup"
    assert initial["calls"] == initial["events"] == []
    frames = [json.loads(line) for line in initial["files"]["frames.jsonl"].splitlines()]
    assert len(frames) == 1 and frames[0]["kind"] == "initial"
    assert frames[0]["groups"] == [] and frames[0]["groupCounter"] == 0
    assert len(initial["files"]["frames.bin"]) == initial["manifest"]["stateSize"] * 8
    frame = frames[0]
    camera = np.array(frame["camera"]["position"])
    target = np.array(frame["camera"]["target"])
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tangent = np.tan(np.deg2rad(38 / 2))
    boxes = []
    for part in sample.parts:
        vertices = apply_pose(part.mesh.vertices, part.initial_pose)
        assert abs(vertices[:, 2].min()) < 1e-9
        boxes.append(np.array([vertices[:, :2].min(0), vertices[:, :2].max(0)]))
        delta = vertices - camera
        depth = delta @ forward
        assert np.all((depth > 0.01) & (depth < 1000))
        assert np.all(np.abs(delta @ up) < depth * tangent)
        assert np.all(np.abs(delta @ right) < depth * tangent * 4 / 3)
        assert part.points.shape == (sample.config.fps_points, 3)
    for i, box in enumerate(boxes):
        assert all(separated(box, other, sample.config.min_gap - 1e-9) for other in boxes[:i])


def verify_sample(browser, args, sample, entry, run, gt, result):
    stem = sample_name(sample.sample_id)
    path = (args.config / entry["episode"]).resolve()
    initial = archive(path)
    check_initial(initial, sample)
    result["native_initial"] = native(args.webagent, "inspect", path)
    assert result["native_initial"]["max_restore_error"] == 0
    errors = result["browser_errors"] = []
    count = len(sample.parts)
    ids = [f"part-{i + 1:04d}" for i in range(count)]
    context, page = open_episode(browser, args, path, count, errors)
    expect(page.get_by_text("Episode · setup", exact=True)).to_be_visible()
    page.screenshot(path=run / "screenshots" / f"{stem}--initial-browser.png")
    state = invoke(page, "get_state")
    check_poses(state, sample, "initial_pose")
    expect(page.get_by_text("Episode · active", exact=True)).to_be_visible()
    first = capture(page, run / "screenshots" / f"{stem}--initial-agent.png")
    calls = page.get_by_role("list", name="Recorded calls").get_by_role("listitem")
    before = calls.count()
    page.get_by_role("button", name="View from +X", exact=True).click()
    expect(calls).to_have_count(before)
    assert capture(page, run / "screenshots" / f"{stem}--free-view-agent.png") == first
    assert invoke(page, "get_state")["camera"] == state["camera"]
    invoke(page, "list_objects")
    invoke(page, "get_object", {"id": ids[0]})
    invoke(page, "get_scene")
    move = {"ids": [ids[0]], "delta": [0.07, -0.03, 0.02], "space": "world"}
    invoke(page, "translate_objects", move)
    invoke(page, "rotate_objects", {"ids": [ids[0]], "angles": [0, 0, 23], "space": "world"})
    if count > 1:
        invoke(page, "group_objects", {"ids": ids[:2], "name": "Round-trip fixture"})
        current = invoke(page, "get_state")
        invoke(page, "ungroup_objects", {"ids": [current["groups"][0]["id"]]})
        invoke(page, "group_objects", {"ids": ids[:2], "name": "Persisted pair"})
    invoke(page, "move_camera", {"yaw": 12, "pitch": -3, "zoom": 1.08})
    current = invoke(page, "get_state")
    runtime_path = run / f"{stem}--runtime.episode.zip"
    runtime = export_run(page, runtime_path)
    assert runtime["manifest"]["lifecycle"] == "active"
    assert len(runtime["calls"]) == calls.count()
    for call in runtime["calls"]:
        assert call["status"] == "completed" and call["actor"] == "agent"
        assert (
            "result" in call and "arguments" in call and call["before_index"] <= call["state_index"]
        )
    assert (
        next(c for c in runtime["calls"] if c["name"] == "translate_objects")["arguments"] == move
    )
    context.close()
    context, page = open_episode(browser, args, runtime_path, count, errors)
    expect(page.get_by_text("Episode · active", exact=True)).to_be_visible()
    expect(page.get_by_role("list", name="Recorded calls").get_by_role("listitem")).to_have_count(
        len(runtime["calls"])
    )
    restored = invoke(page, "get_state")
    assert restored == current
    corrupt = run / f"{stem}--bad-hash.zip"
    with zipfile.ZipFile(corrupt, "w") as z:
        for name, data in initial["files"].items():
            z.writestr(name, data + b"\n" if name == "world/model.xml" else data)
    page.get_by_label("Episode file", exact=True).set_input_files(corrupt)
    expect(page.get_by_role("alert").filter(has_text="Archive checksum mismatch")).to_be_visible()
    assert invoke(page, "get_state") == restored
    invoke(page, "end_episode", {"reason": "Deterministic integration fixture completed"})
    expect(page.get_by_text("Episode · ended", exact=True)).to_be_visible()
    ended_path = run / f"{stem}--ended.episode.zip"
    ended = export_run(page, ended_path)
    assert ended["calls"][: len(runtime["calls"])] == runtime["calls"]
    assert ended["events"][: len(runtime["events"])] == runtime["events"]
    for name in ("frames.jsonl", "frames.bin"):
        assert ended["files"][name] == runtime["files"][name]
    context.close()
    context, page = open_episode(browser, args, ended_path, count, errors)
    expect(page.get_by_text("Episode · ended", exact=True)).to_be_visible()
    context.close()
    result["native_runtime"] = native(args.webagent, "verify", runtime_path)
    result["native_ended"] = native(args.webagent, "verify", ended_path)
    assert result["native_runtime"]["status"] == result["native_ended"]["status"] == "passed"
    # Ground truth is rebuilt from HF and consumed only in this separate fixture run.
    context, page = open_episode(browser, args, path, count, errors)
    for i, part in enumerate(sorted(sample.parts, key=lambda p: p.part_id)):
        invoke(
            page,
            "set_object_pose",
            {
                "id": ids[i],
                "position": part.gt_pose.position.tolist(),
                "quaternion": part.gt_pose.quaternion.tolist(),
            },
        )
    check_poses(invoke(page, "get_state"), sample, "gt_pose")
    capture(page, gt / "screenshots" / f"{stem}--gt-agent.png")
    page.screenshot(path=gt / "screenshots" / f"{stem}--gt-browser.png")
    gtpath = gt / f"{stem}--gt-fixture.episode.zip"
    export_run(page, gtpath)
    context.close()
    gt_result = native(args.webagent, "verify", gtpath)
    assert gt_result["status"] == "passed" and not errors
    result.update(
        status="passed",
        exact_roundtrip_records=True,
        free_view_independent=True,
        corrupted_import_preserves_scene=True,
        initial_camera_covers_all_vertices=True,
    )
    return dict(sample_id=sample.sample_id, status="passed", native=gt_result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--webagent", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:5184/")
    args = parser.parse_args()
    args.webagent = args.webagent.resolve()
    configuration = read_config(args.config)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.webagent, text=True
    ).strip()
    assert commit == contract_provenance()["commit"], "Use the pinned 3DWebAgent checkout"
    with (
        experiment("browser-acceptance", logs=args.logs, inputs=configuration) as (
            run,
            meta,
            metrics,
        ),
        experiment("gt-regression", logs=args.logs, inputs=configuration) as (gt, gm, gmetrics),
    ):
        for directory, m in ((run, meta), (gt, gm)):
            (directory / "screenshots").mkdir()
            m.update(
                upstream_commit=commit,
                engine="3.12.0",
                external_mcp_client_tested=False,
                model_assembly_evaluated=False,
            )
        gm["fixture"] = "GT-driven, not agent assembly success"
        metrics["tolerances"] = {"atol": 1e-9, "rtol": 1e-7}
        metrics["wasm_initial"] = wasm(args.webagent, args.config)
        metrics["samples"] = []
        gmetrics["samples"] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                for _, sample in load_prepared(args.config, cache_dir=args.cache_dir):
                    result = {"sample_id": sample.sample_id}
                    metrics["samples"].append(result)
                    gmetrics["samples"].append(
                        verify_sample(
                            browser,
                            args,
                            sample,
                            configuration["samples"][sample.sample_id],
                            run,
                            gt,
                            result,
                        )
                    )
                    print(f"{sample.sample_id}: browser passed", flush=True)
            finally:
                browser.close()
        metrics["wasm_runtime"] = wasm(args.webagent, run)
        gmetrics["wasm_gt"] = wasm(args.webagent, gt)
    print(run)
    print(gt)


if __name__ == "__main__":
    main()
