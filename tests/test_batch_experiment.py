import hashlib
import json
import zipfile

import pytest

from assembly_world_agent.batch_experiment import (
    archive_episode,
    collect_exports,
    export_trace,
    finalize_ready,
    render_sample,
)


def sample_directory(tmp_path):
    directory = tmp_path / "samples" / "Bench--applaro"
    (directory / "trace").mkdir(parents=True)
    (directory / "checkpoints").mkdir()
    (directory / "input.json").write_text(json.dumps({"episode_id": "expected"}))
    return directory


def episode(path, identity="expected", calls=b"{}\n"):
    manifest = dict(
        id=identity, lifecycle="active", hashes={"calls.jsonl": hashlib.sha256(calls).hexdigest()}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("calls.jsonl", calls)


def test_archive_rejects_wrong_sample_and_conflicting_final(tmp_path):
    directory = sample_directory(tmp_path)
    source = tmp_path / "download.zip"
    episode(source, identity="another")
    with pytest.raises(ValueError, match="different sample"):
        archive_episode(tmp_path, "Bench/applaro", source)
    assert not (directory / "final.episode.zip").exists()
    episode(source)
    result = archive_episode(tmp_path, "Bench/applaro", source)
    original = result.read_bytes()
    episode(source, calls=b"{}\n{}\n")
    with pytest.raises(FileExistsError):
        archive_episode(tmp_path, "Bench/applaro", source)
    assert result.read_bytes() == original


def test_export_trace_keeps_calls_without_private_content(tmp_path):
    directory = sample_directory(tmp_path)
    source = tmp_path / "rollout.jsonl"
    payloads = [
        {"type": "message", "role": "system", "content": "private instructions"},
        {
            "type": "message",
            "role": "assistant",
            "channel": "analysis",
            "content": "private thought",
        },
        {"type": "reasoning", "content": "private reasoning"},
        {"type": "function_call", "name": "capture_scene", "arguments": "{}"},
        {"type": "function_call_output", "output": "original result"},
        {"type": "message", "role": "assistant", "channel": "final", "content": "done"},
    ]
    source.write_text(
        "".join(json.dumps({"type": "response_item", "payload": p}) + "\n" for p in payloads)
    )
    assert export_trace(tmp_path, "Bench/applaro", source) == 3
    text = (directory / "trace" / "records.jsonl").read_text()
    assert "private" not in text
    assert "capture_scene" in text and "original result" in text


def test_download_collector_ignores_unrelated_files_and_deduplicates(tmp_path):
    directory = sample_directory(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"started_at": "2026-01-01T00:00:00+00:00"}))
    (tmp_path / "progress.json").write_text(json.dumps({"Bench/applaro": {"status": "running"}}))
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "unrelated.zip").write_bytes(b"unrelated user file")
    episode(downloads / "ikea-manual___Bench_applaro.episode.zip")
    assert len(collect_exports(tmp_path, downloads)) == 1
    assert not (downloads / "ikea-manual___Bench_applaro.episode.zip").exists()
    first = directory / "exports" / "ikea-manual___Bench_applaro.episode.zip"
    original = first.read_bytes()
    assert (downloads / "unrelated.zip").read_bytes() == b"unrelated user file"
    assert collect_exports(tmp_path, downloads) == []
    episode(downloads / "ikea-manual___Bench_applaro.episode.zip", calls=b"{}\n{}\n")
    assert len(collect_exports(tmp_path, downloads)) == 1
    assert first.read_bytes() == original
    assert len(list((directory / "exports").glob("*.zip"))) == 2


def test_run_can_stop_video_generation_without_blocking_complete_artifacts(tmp_path):
    directory = sample_directory(tmp_path)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"generate_mp4": True}))
    (tmp_path / "progress.json").write_text(json.dumps({"Bench/applaro": {"status": "archiving"}}))
    source = tmp_path / "download.zip"
    calls = json.dumps({"index": 0, "name": "capture_scene", "state_index": 0, "timestamp": "now"})
    episode(source, calls=(calls + "\n").encode())
    archive_episode(tmp_path, "Bench/applaro", source)
    (directory / "report.md").write_text("Checked the connection.")
    (directory / "result.json").write_text(
        json.dumps({"status": "completed", "checked_connections": [{"capture_calls": [1]}]})
    )
    (directory / "trace/manifest.json").write_text(json.dumps({"records": 1}))
    (directory / "trace/records.jsonl").write_text("{}\n")
    assert finalize_ready(tmp_path) == []
    meta.write_text(json.dumps({"generate_mp4": False}))
    with pytest.raises(RuntimeError, match="disabled"):
        render_sample(tmp_path, "Bench/applaro")
    assert finalize_ready(tmp_path) == ["Bench/applaro"]
    assert not (directory / "replay.mp4").exists()
    assert json.loads((directory / "evidence.json").read_text())["status"] == "passed"
