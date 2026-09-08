"""Offline orchestration tests; real browser/CLI smoke runs are explicit."""

import asyncio
import json
import os
import sys

import pytest

from assembly_world_agent.artifacts import sample_name, write_json
from assembly_world_agent.cli import main
from assembly_world_agent.experiments import agents, runner
from assembly_world_agent.experiments.transport import images


def make_run(tmp_path, count=3):
    inputs = {
        f"sample-{i}": {"sample_id": f"sample-{i}", "initial_path": str(tmp_path / f"{i}.zip")}
        for i in range(count)
    }
    return runner.create_run({"logs": str(tmp_path), "concurrency": 2}, None, inputs)


def test_cli_validates_selection():
    with pytest.raises(SystemExit):
        main(["run", "--dataset", "ikea-manual", "--agent", "codex", "--model", "test"])
    with pytest.raises(SystemExit):
        main(["run", "--episode", "x.zip", "--limit", "2", "--agent", "codex", "--model", "test"])


def test_compact_layout(tmp_path):
    run = make_run(tmp_path, 1)
    assert {p.name for p in run.iterdir()} == {"run.json", "samples"}
    assert {p.name for p in (run / "samples/sample-0").iterdir()} == {
        "input.json",
        "prompt.txt",
        "conversation.jsonl",
        "result.json",
    }
    assert runner.status(run)["counts"] == {"pending": 1}


def test_bounded_workers_and_no_early_stop_on_failure(tmp_path):
    run = make_run(tmp_path, 5)
    active = peak = 0
    visited = []

    urls = []

    async def execute(directory, sid, meta, *, episode_url):
        nonlocal active, peak
        urls.append(episode_url)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        visited.append(sid)
        result = {"status": "failed" if sid == "sample-0" else "completed"}
        write_json(directory / "samples" / sample_name(sid) / "result.json", result)
        return result

    assert asyncio.run(runner.schedule(run, execute=execute)) == 1
    assert peak == 2
    assert len(visited) == 5
    from urllib.parse import urlsplit

    assert len({urlsplit(url).netloc for url in urls}) == 1
    assert len(set(urls)) == 5
    import urllib.error
    import urllib.request

    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(urls[0], timeout=1)
    assert runner.status(run)["counts"] == {"completed": 4, "failed": 1}


def test_resume_preserves_old_runs_and_skips_partial(tmp_path):
    run = make_run(tmp_path, 4)
    states = [
        {"status": "completed", "agent_outcome": {"status": "partial"}},
        {"status": "running"},
        {"status": "failed", "execution": {"status": "interrupted"}},
        {"status": "failed", "execution": {"status": "timeout"}},
    ]
    for i, result in enumerate(states):
        if i == 0:
            (run / "samples/sample-0/final.episode.zip").write_bytes(b"final")
            result["archive"] = {"status": "saved", "sha256": runner.sha256(b"final")}
        write_json(run / "samples" / f"sample-{i}" / "result.json", result)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    meta, config, inputs = runner.resume_inputs(run)
    assert set(inputs) == {"sample-1", "sample-2"}
    new = runner.create_run(meta["options"], config, inputs, source_run=str(run), task=meta["task"])
    assert new != run
    assert all(p.read_bytes() == value for p, value in before.items())
    assert set(runner.resume_inputs(run, True)[2]) == {"sample-1", "sample-2", "sample-3"}


def test_conversation_privacy_and_unknown_events():
    assert (
        agents.normalize(
            {"type": "item.completed", "item": {"type": "reasoning", "text": "private"}}
        )
        is None
    )
    assert agents.normalize({"type": "system", "session_id": "abc", "instructions": "secret"}) == {
        "type": "session",
        "session_id": "abc",
    }
    record = agents.normalize(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {"type": "text", "text": "visible"},
                ]
            },
        }
    )
    assert record["content"] == [{"type": "text", "text": "visible"}]
    assert agents.normalize({"type": "new_vendor_event", "value": 3})["event"]["value"] == 3


def test_serialized_webmcp_images():
    picture = {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}
    value = {
        "content": [
            {"type": "text", "text": json.dumps({"output": json.dumps({"content": [picture]})})}
        ]
    }
    assert list(images(value)) == [picture]


def test_outcome_keeps_unstructured_final_answer_separate():
    assert runner.outcome("No JSON answer") is None
    assert runner.outcome('```json\n{"status":"partial"}\n```') == {"status": "partial"}


@pytest.mark.parametrize("failure", ["agent", "export", "timeout", "cancel", None])
def test_failure_salvage_and_cleanup(tmp_path, monkeypatch, failure):
    initial = tmp_path / "input.zip"
    initial.write_bytes(b"input")
    run = make_run(tmp_path, 1)
    sample = run / "samples/sample-0"
    write_json(
        sample / "input.json", {"initial_path": str(initial), "sha256": runner.sha256(b"input")}
    )
    meta = runner.read_json(run / "run.json")
    meta["options"]["timeout_seconds"] = 0.01 if failure == "timeout" else None
    monkeypatch.setattr(runner, "manual_pages", lambda *args: [])
    closed = []

    class Browser:
        loaded = False

        def __init__(self, *args):
            pass

        async def start(self, *args):
            assert args == ("http://127.0.0.1/episode.zip",)
            self.loaded = True

        def mcp_command(self):
            return ["unused"]

        async def export(self, path, inputs):
            if failure == "export":
                raise RuntimeError("export failed")
            path.write_bytes(b"final")
            return {"status": "saved", "sha256": runner.sha256(b"final")}

        async def close(self):
            closed.append(True)

    async def agent(*args):
        if failure == "agent":
            raise RuntimeError("agent failed")
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "timeout":
            await asyncio.sleep(1)
        return {"status": "completed", "final_answer": '{"status":"completed"}'}

    result = asyncio.run(
        runner.execute_sample(
            run,
            "sample-0",
            meta,
            episode_url="http://127.0.0.1/episode.zip",
            browser_factory=Browser,
            agent=agent,
        )
    )
    assert closed == [True]
    assert result["status"] == ("completed" if failure is None else "failed")
    assert (sample / "final.episode.zip").exists() == (failure != "export")
    assert initial.read_bytes() == b"input"
    assert len(list(sample.iterdir())) == (4 if failure == "export" else 5)


def test_changed_input_is_not_run(tmp_path):
    run = make_run(tmp_path, 1)
    write_json(
        run / "samples/sample-0/input.json",
        {"initial_path": str(tmp_path / "missing.zip"), "sha256": "wrong"},
    )
    result = asyncio.run(
        runner.execute_sample(
            run,
            "sample-0",
            runner.read_json(run / "run.json"),
            episode_url="http://127.0.0.1/missing",
        )
    )
    assert result["status"] == "failed"
    assert result["archive"]["status"] == "not_saved"


def test_commands_isolate_mcp_and_do_not_bypass_permissions(tmp_path):
    for name in ("codex", "claude"):
        args = agents.command({"agent": name, "model": "test"}, tmp_path, tmp_path / "bridge.json")
        assert not any("dangerously" in arg for arg in args)
        assert "--ignore-user-config" in args if name == "codex" else "--strict-mcp-config" in args


def test_evaluation_reads_new_metadata_without_provenance(tmp_path):
    from assembly_world_agent.evaluation.runner import evaluate_run

    write_json(tmp_path / "run.json", {"config": None})
    with pytest.raises(ValueError, match="no dataset provenance"):
        evaluate_run(tmp_path)


@pytest.mark.parametrize("invalid", [False, True])
def test_actual_subprocess_streaming_and_errors(tmp_path, monkeypatch, invalid):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        + ("print('invalid event')\n" if invalid else "")
        + "print(json.dumps({'type':'item.completed','item':"
        "{'type':'agent_message','text':'final reply'}}))\n"
        "print('diagnostic', file=sys.stderr)\n"
    )
    monkeypatch.setattr(agents, "command", lambda *args: [sys.executable, str(script)])
    result = asyncio.run(
        agents.run_agent({}, tmp_path, None, "test", tmp_path / "conversation.jsonl")
    )
    assert result["status"] == ("failed" if invalid else "completed")
    assert result["final_answer"] == "final reply"
    assert result["usage"] is None
    assert ("error" in result) == invalid


def test_cancel_terminates_agent_process(tmp_path, monkeypatch):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import os, time\n"
        f"open({str(tmp_path / 'pid')!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(agents, "command", lambda *args: [sys.executable, str(script)])

    async def run():
        task = asyncio.create_task(agents.run_agent({}, tmp_path, None, "", tmp_path / "log"))
        async with asyncio.timeout(5):
            while not (tmp_path / "pid").exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


def test_resume_detects_missing_archive(tmp_path):
    run = make_run(tmp_path, 1)
    write_json(
        run / "samples/sample-0/result.json",
        {"status": "completed", "archive": {"sha256": "missing"}},
    )
    assert list(runner.resume_inputs(run)[2]) == ["sample-0"]


@pytest.mark.parametrize("subcommand", ["run", "doctor"])
@pytest.mark.parametrize("headless", [False, True])
def test_cli_browser_mode(subcommand, headless, monkeypatch):
    from assembly_world_agent.experiments import browser

    received = []

    async def inspect(options):
        received.append(options)
        return {} if subcommand == "doctor" else 0

    monkeypatch.setattr(browser, "doctor", inspect)
    monkeypatch.setattr(runner, "launch", inspect)
    args = [subcommand, "--agent", "claude"]
    if subcommand == "run":
        args += ["--episode", "/tmp/example.episode.zip", "--model", "test"]
    if headless:
        args += ["--headless"]
    assert main(args) == 0
    assert received[0]["headless"] is headless


@pytest.mark.parametrize("headless", [None, False, True])
def test_resume_preserves_browser_mode(tmp_path, monkeypatch, headless):
    source = make_run(tmp_path, 1)
    meta = runner.read_json(source / "run.json")
    if headless is not None:
        meta["options"]["headless"] = headless
    write_json(source / "run.json", meta)
    observed = []

    async def doctor(options):
        assert options.get("headless", False) is bool(headless)
        return {}

    async def schedule(directory):
        observed.append(runner.read_json(directory / "run.json"))
        return 0

    monkeypatch.setattr(runner, "doctor", doctor)
    monkeypatch.setattr(runner, "schedule", schedule)
    assert asyncio.run(runner.launch({"concurrency": 1}, source=source)) == 0
    assert observed[0]["options"].get("headless", False) is bool(headless)
    assert observed[0]["source_run"] == str(source)
