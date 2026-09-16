"""Batch dispatch safety and recovered transport classification."""

import asyncio

import pytest

from assembly_world_agent.experiments import batch, runner


def test_recovered_transport_requires_successful_exit_and_export():
    result = {
        "archive": {"status": "saved"},
        "execution": {
            "status": "failed",
            "exit_code": 0,
            "final_answer": "partial",
            "usage": {},
            "agent_error": {"type": "error", "message": "Reconnecting... waiting for network"},
        },
    }
    assert batch.outcome(result) == "completed_with_transport_warning"
    result["execution"]["exit_code"] = 1
    assert batch.outcome(result) == "failed"
    result["execution"]["exit_code"] = 0
    result["archive"]["status"] = "not_saved"
    assert batch.outcome(result) == "failed"


def test_failure_guard_drains_active_samples(monkeypatch, tmp_path):
    async def scenario():
        started = []
        active_started = asyncio.Event()
        release = asyncio.Event()

        async def execute(directory, sid, meta, *, episode_url):
            started.append(sid)
            if sid == "active":
                active_started.set()
                await release.wait()
            return {"archive": {"status": "not_saved"}}

        monkeypatch.setattr(runner, "execute_sample", execute)
        guard = batch.DrainGuard(tmp_path / "STOP")
        active = asyncio.create_task(guard.execute(tmp_path, "active", {}, episode_url="test"))
        await active_started.wait()
        for sid in ["one", "two", "three"]:
            await guard.execute(tmp_path, sid, {}, episode_url="test")
        with pytest.raises(RuntimeError, match="consecutive"):
            await guard.execute(tmp_path, "not-started", {}, episode_url="test")
        assert not active.done()
        release.set()
        await active
        assert "not-started" not in started
        assert "active" in guard.results

    asyncio.run(scenario())


@pytest.mark.parametrize("pause", [False, True])
def test_groups_run_in_order_and_pause_prevents_next_group(monkeypatch, tmp_path, pause):
    from assembly_world_agent.artifacts import write_json

    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "options": {},
            "task": "same task",
            "groups": [
                {"name": name, "expected_samples": 1, "options": {"name": name}}
                for name in ["first", "second"]
            ],
        },
    )
    visited = []

    async def doctor(options):
        return {}

    async def schedule(directory, *, execute):
        visited.append(directory.name)
        if pause:
            execute.__self__.reason = "Systemic failure"
        return 1 if pause else 0

    monkeypatch.setattr(runner, "doctor", doctor)
    monkeypatch.setattr(runner, "select_inputs", lambda options: (None, {"one": {}}))
    monkeypatch.setattr(runner, "create_run", lambda options, *a, **k: tmp_path / options["name"])
    monkeypatch.setattr(runner, "schedule", schedule)
    monkeypatch.setattr(runner, "status", lambda directory: {"counts": {"completed": 1}})
    result = asyncio.run(batch.launch_batch(plan))
    assert visited == (["first"] if pause else ["first", "second"])
    assert result["status"] == ("paused" if pause else "finished")


@pytest.mark.parametrize("stop_after", [None, "image"])
def test_reorder_retains_attempts_and_selects_only_pending(monkeypatch, tmp_path, stop_after):
    from assembly_world_agent.artifacts import write_json

    source = tmp_path / "batch-plan.json"
    write_json(
        source,
        {
            "options": {},
            "task": "fixed",
            "groups": [
                {
                    "name": name,
                    "expected_samples": 3,
                    "options": {"sample_id": ["done", "failed", "pending"]},
                }
                for name in ["none", "image"]
            ],
        },
    )
    run = tmp_path / "run"
    write_json(run / "run.json", {"samples": ["done", "failed", "pending"]})
    for sid, status in [("done", "completed"), ("failed", "failed"), ("pending", "pending")]:
        write_json(run / "samples" / sid / "result.json", {"status": status})
    write_json(
        tmp_path / "batch_state.json",
        {
            "status": "paused",
            "pause_reason": "Operator requested drain",
            "groups": [{"name": "none", "run": str(run)}],
        },
    )

    async def launch(path):
        return runner.read_json(path)

    monkeypatch.setattr(batch, "launch_batch", launch)
    result = asyncio.run(
        batch.continue_reordered(source, tmp_path / "new", ["image", "none"], stop_after=stop_after)
    )
    assert [g["name"] for g in result["groups"]] == (["image"] if stop_after else ["image", "none"])
    if stop_after:
        assert [g["name"] for g in result["deferred_groups"]] == ["none"]
    else:
        assert result["groups"][1]["options"]["sample_id"] == ["pending"]
    assert result["retained_runs"] == [str(run)]


def test_plan_without_task_uses_each_groups_prompt_file(monkeypatch, tmp_path):
    from assembly_world_agent.artifacts import write_json

    prompts = {}
    for name in ("none", "image"):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"task for {name}")
        prompts[name] = str(path)
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "options": {},
            "groups": [
                {"name": name, "expected_samples": 1, "options": {"prompt_file": prompts[name]}}
                for name in prompts
            ],
        },
    )
    tasks = []

    async def doctor(options):
        return {}

    async def schedule(directory, *, execute):
        return 0

    original = runner.create_run
    monkeypatch.setattr(runner, "doctor", doctor)
    monkeypatch.setattr(runner, "select_inputs", lambda options: (None, {"one": {}}))
    monkeypatch.setattr(runner, "schedule", schedule)
    monkeypatch.setattr(runner, "status", lambda directory: {"counts": {"completed": 1}})

    def record(options, config, inputs, *, task=None, **kwargs):
        options = {**options, "logs": str(tmp_path / "logs")}
        directory = original(options, config, inputs, task=task, **kwargs)
        tasks.append(runner.read_json(directory / "run.json")["task"])
        return directory

    monkeypatch.setattr(runner, "create_run", record)
    result = asyncio.run(batch.launch_batch(plan))
    assert result["status"] == "finished"
    assert tasks == ["task for none", "task for image"]
