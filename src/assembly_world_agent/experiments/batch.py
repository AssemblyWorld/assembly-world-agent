"""Run ordered experiment groups with a draining failure circuit breaker."""

import asyncio
from pathlib import Path

from ..artifacts import now, sample_name, write_json
from . import runner


def outcome(result):
    """Classify recovered transport warnings without changing original results."""
    execution = result.get("execution", {})
    if result.get("archive", {}).get("status") != "saved":
        return "failed"
    if execution.get("status") == "completed":
        return "completed"
    error = execution.get("agent_error") or {}
    if (
        execution.get("exit_code") == 0
        and execution.get("final_answer")
        and execution.get("usage") is not None
        and not execution.get("parse_errors")
        and isinstance(error, dict)
        and error.get("type") == "error"
        and error.get("message", "").startswith("Reconnecting...")
    ):
        return "completed_with_transport_warning"
    return "failed"


class DrainGuard:
    """Stop new dispatches after consecutive failures, preserving active tasks."""

    def __init__(self, stop_file, failure_limit=3):
        self.stop_file = Path(stop_file)
        self.failure_limit = failure_limit
        self.failures = 0
        self.reason = None
        self.results = {}

    async def execute(self, directory, sid, meta, *, episode_url):
        if self.stop_file.exists():
            self.reason = "Operator requested drain"
        if self.reason:
            raise RuntimeError(self.reason)
        result = await runner.execute_sample(directory, sid, meta, episode_url=episode_url)
        classification = outcome(result)
        self.results[sid] = classification
        self.failures = self.failures + 1 if classification == "failed" else 0
        if self.failures >= self.failure_limit:
            self.reason = f"{self.failure_limit} consecutive execution/archive failures"
        write_json(
            Path(directory) / "batch_outcomes.json",
            {
                "updated_at": now(),
                "pause_reason": self.reason,
                "samples": self.results,
            },
        )
        return result


async def launch_batch(plan_path):
    """Launch each declared group once; never retry or silently skip failures."""
    plan_path = Path(plan_path).resolve()
    plan = runner.read_json(plan_path)
    state_path = plan_path.with_name("batch_state.json")
    if state_path.exists():
        raise ValueError("Batch state already exists; inspect it before creating a continuation")
    state = {"started_at": now(), "status": "preflight", "groups": []}
    write_json(state_path, state)
    try:
        versions = await runner.doctor(plan["options"])
        for group in plan["groups"]:
            if plan_path.with_name("STOP").exists():
                state.update(status="paused", pause_reason="Operator requested drain")
                break
            options = {**plan["options"], **group["options"]}
            config, inputs = await asyncio.to_thread(runner.select_inputs, options)
            if len(inputs) != group["expected_samples"]:
                raise ValueError(f"Unexpected sample count for {group['name']}")
            # A plan without "task" lets each group's prompt_file supply its own text.
            directory = runner.create_run(
                options, config, inputs, task=plan.get("task"), versions=versions
            )
            entry = {"name": group["name"], "run": str(directory), "status": "running"}
            state["groups"].append(entry)
            state.update(status="running", current_group=group["name"], updated_at=now())
            write_json(state_path, state)
            print(f"Group {group['name']}: {directory}", flush=True)
            guard = DrainGuard(plan_path.with_name("STOP"))
            await runner.schedule(directory, execute=guard.execute)
            counts = runner.status(directory)["counts"]
            unfinished = counts.get("pending", 0) + counts.get("running", 0)
            entry.update(
                status="paused" if guard.reason or unfinished else "finished", counts=counts
            )
            if guard.reason or unfinished:
                state.update(status="paused", pause_reason=guard.reason or "Unfinished samples")
                break
            write_json(state_path, state)
        else:
            state["status"] = "finished"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["updated_at"] = now()
        write_json(state_path, state)
    return state


async def continue_reordered(source_plan, destination, order, *, stop_after=None):
    """Wait for an operator drain, then continue only unattempted sample IDs."""
    source_plan = Path(source_plan).resolve()
    destination = Path(destination).resolve()
    plan = runner.read_json(source_plan)
    groups = {group["name"]: group for group in plan["groups"]}
    if len(order) != len(groups) or set(order) != set(groups):
        raise ValueError("Order must contain each existing group exactly once")
    if stop_after is not None and stop_after not in order:
        raise ValueError("Stop group must belong to the declared order")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "handoff.json",
        {
            "status": "waiting_for_drain",
            "source_plan": str(source_plan),
            "order": order,
            "started_at": now(),
        },
    )
    while True:
        state = runner.read_json(source_plan.with_name("batch_state.json"))
        if state["status"] not in {"running", "preflight"}:
            break
        await asyncio.sleep(10)
    if state["status"] != "paused" or state.get("pause_reason") != "Operator requested drain":
        raise ValueError(f"Source batch did not complete an operator drain: {state}")
    retained = list(plan.get("retained_runs", []))
    attempted = {name: set() for name in groups}
    for entry in state["groups"]:
        run = Path(entry["run"])
        retained.append(str(run))
        outcomes_path = run / "batch_outcomes.json"
        if outcomes_path.exists():
            classifications = list(runner.read_json(outcomes_path)["samples"].values())
            if len(classifications) >= 3 and classifications[-3:] == ["failed"] * 3:
                raise ValueError("Consecutive infrastructure failures require inspection")
        for sid in runner.read_json(run / "run.json")["samples"]:
            result = runner.read_json(run / "samples" / sample_name(sid) / "result.json")
            if result["status"] == "running":
                raise ValueError(f"Sample still running: {sid}")
            if result["status"] != "pending":
                attempted[entry["name"]].add(sid)
    selected = []
    active_order = order if stop_after is None else order[: order.index(stop_after) + 1]
    for name in active_order:
        group = groups[name]
        ids = group["options"]["sample_id"]
        remaining = [sid for sid in ids if sid not in attempted[name]]
        if remaining:
            selected.append(
                {
                    **group,
                    "expected_samples": len(remaining),
                    "options": {**group["options"], "sample_id": remaining},
                }
            )
    new_plan = {
        **plan,
        "groups": selected,
        "retained_runs": retained,
        "deferred_groups": [groups[name] for name in order[len(active_order) :]],
        "stop_after": stop_after,
    }
    target = destination / "batch-plan.json"
    write_json(target, new_plan)
    write_json(
        destination / "handoff.json",
        {
            "status": "continuing",
            "source_plan": str(source_plan),
            "order": order,
            "updated_at": now(),
            "plan": str(target),
        },
    )
    return await launch_batch(target)
