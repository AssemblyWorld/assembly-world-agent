"""AssemblyWorldBench: benchmark definition, run matching and SR aggregation.

A benchmark is five ordinary prepared configuration directories plus one
``benchmark.json`` that names them. Runs are produced by the normal ``run``
command and scored by the normal evaluation; this module only decides which
block a run belongs to and folds block scores into the frozen Overall rule.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .artifacts import now, sample_name, write_json
from .episodes import sha256

BENCHMARK_VERSION = 1
BLOCK_KEYS = (
    "name",
    "source",
    "dataset",
    "repo_id",
    "revision",
    "config_id",
    "data",
    "reference_mode",
    "prompt_file",
    "prompt_sha256",
    "samples",
)


def read_benchmark(path):
    """Load and validate a benchmark definition; paths inside are relative to its directory."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("version") != BENCHMARK_VERSION:
        raise ValueError("Unsupported benchmark version")
    names, keys = set(), set()
    for block in value["blocks"]:
        missing = [k for k in BLOCK_KEYS if k not in block]
        if missing:
            raise ValueError(f"Block is missing {missing}")
        if block["name"] in names:
            raise ValueError(f"Duplicate block name: {block['name']}")
        names.add(block["name"])
        key = (block["repo_id"], block["config_id"], block["reference_mode"])
        if key in keys:
            raise ValueError(f"Blocks share dataset, configuration and reference mode: {key}")
        keys.add(key)
        if not block["samples"]:
            raise ValueError(f"Block has no samples: {block['name']}")
        for sid in block["samples"]:
            sample_name(sid)
        if block["source"] not in value["sources"]:
            raise ValueError(f"Block source is not an aggregation source: {block['source']}")
        prompt = path.parent / block["prompt_file"]
        if sha256(prompt.read_bytes()) != block["prompt_sha256"]:
            raise ValueError(f"Task text differs from the benchmark record: {prompt}")
    value["_directory"] = str(path.parent)
    return value


def _run_meta(run):
    run = Path(run).resolve()
    metadata = run / "run.json" if (run / "run.json").exists() else run / "meta.json"
    return run, json.loads(metadata.read_text())


def block_for_run(benchmark, run, *, accept_task_mismatch=False):
    """The unique block a run belongs to, or an error naming what differs.

    With ``accept_task_mismatch`` a run whose recorded task text differs from the
    block's ``task.txt`` (for example an imported historical run) is matched anyway;
    the caller records the mismatch instead of hiding it.
    """
    run, meta = _run_meta(run)
    config = meta.get("config") or {}
    identity = config.get("identity") or {}
    mode = (meta.get("options") or {}).get("reference_mode")
    key = (identity.get("dataset"), config.get("config_id"), mode)
    matches = [
        b for b in benchmark["blocks"] if (b["repo_id"], b["config_id"], b["reference_mode"]) == key
    ]
    if len(matches) != 1:
        raise ValueError(f"{run.name}: no benchmark block for {key}")
    block = matches[0]
    if identity.get("revision") != block["revision"]:
        raise ValueError(f"{run.name}: revision differs from block {block['name']}")
    unknown = sorted(set(config["samples"]) - set(block["samples"]))
    if unknown:
        raise ValueError(f"{run.name}: samples outside block {block['name']}: {unknown}")
    for sid, item in config["samples"].items():
        if item.get("sha256") != block["samples"][sid]["sha256"]:
            raise ValueError(f"{run.name}: initial episode differs for {sid}")
    if sha256(meta.get("task", "").encode()) != block["prompt_sha256"] and not accept_task_mismatch:
        raise ValueError(f"{run.name}: task text differs from block {block['name']}")
    return block


def task_matches(benchmark, run):
    """Whether a run's recorded task text is the block's ``task.txt``."""
    block = block_for_run(benchmark, run, accept_task_mismatch=True)
    _, meta = _run_meta(run)
    return sha256(meta.get("task", "").encode()) == block["prompt_sha256"]


def match_runs_to_blocks(benchmark, runs, *, accept_task_mismatch=False):
    """Group run directories by block; every run must belong to exactly one block."""
    grouped = {}
    for run in runs:
        block = block_for_run(benchmark, run, accept_task_mismatch=accept_task_mismatch)
        grouped.setdefault(block["name"], []).append(Path(run).resolve())
    return grouped


def _is_attempt(result):
    """A sample record that produced a final episode; interrupted or unstarted ones are not.

    A drained batch leaves undispatched samples behind as pending records with no archive,
    so they must not be counted as evaluations when a continuation run is joined with it.
    """
    if result.get("status") in {"pending", "running"}:
        return False
    archive = result.get("archive") or {}
    return archive.get("status", "saved") == "saved"


def agent_outcomes(runs, sample_ids):
    """Count recorded agent outcome statuses over the runs' samples (refusals show as unable)."""
    counter = Counter()
    for run in runs:
        run, meta = _run_meta(run)
        for sid in meta.get("samples", []):
            if sid not in sample_ids:
                continue
            path = run / "samples" / sample_name(sid) / "result.json"
            # A run that lists a sample it holds no record for contributes nothing: the
            # sample was either never dispatched or its record was set aside, and another
            # run in the join supplies it.
            if not path.is_file():
                continue
            result = json.loads(path.read_text())
            if not _is_attempt(result):
                continue
            outcome = result.get("agent_outcome") or {}
            counter[str(outcome.get("status") or "none")] += 1
    return dict(sorted(counter.items()))


# Published list prices in USD per million tokens, standard tier, short context, used only
# when an agent CLI reports token usage but no cost (Codex). Recorded with every estimate.
LIST_PRICES = {
    "gpt-6-astra": dict(
        input=10.0,
        cached_input=1.0,
        output=50.0,
        source="https://developers.openai.com/api/docs/pricing",
        retrieved="2026-09-16",
    ),
    "gpt-5.6-sol": dict(
        input=4.0,
        cached_input=0.4,
        output=20.0,
        source="https://developers.openai.com/api/docs/pricing",
        retrieved="2026-09-20",
    ),
    "gpt-5.6-terra": dict(
        input=2.0,
        cached_input=0.2,
        output=12.0,
        source="https://developers.openai.com/api/docs/pricing",
        retrieved="2026-09-20",
    ),
    # Served locally with vLLM; priced at the OpenRouter list price for the same model so
    # the cost column stays comparable with hosted systems.
    "qwen3.8-27b": dict(
        input=0.214,
        cached_input=0.15,
        output=2.55,
        source="https://openrouter.ai/qwen/qwen3.8-27b",
        retrieved="2026-09-16",
    ),
    # DeepSeek V4.1 Flash called directly on api.deepseek.com. DeepSeek charges twice these
    # rates during peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays); the off-peak
    # rates are recorded here and the provenance notes when each block ran.
    "deepseek-flash": dict(
        input=0.15,
        cached_input=0.003,
        output=0.60,
        source="https://api-docs.deepseek.com/quick_start/pricing",
        retrieved="2026-09-19",
    ),
    # Alibaba-hosted Qwen3.8-Flash reached through OpenRouter (its only provider).
    "qwen/qwen3.8-flash": dict(
        input=0.15,
        cached_input=0.016,
        output=0.47,
        source="https://openrouter.ai/qwen/qwen3.8-flash",
        retrieved="2026-09-17",
    ),
    # Same model called directly on Alibaba Model Studio (DashScope), same list price.
    "qwen3.8-flash": dict(
        input=0.15,
        cached_input=0.016,
        output=0.47,
        source="https://www.qwencloud.com/pricing/api",
        retrieved="2026-09-18",
    ),
    "qwen3.8-max": dict(
        input=2.0,
        cached_input=0.25,
        output=6.0,
        source="https://www.qwencloud.com/pricing/api",
        retrieved="2026-09-18",
    ),
}


def estimate_cost(model, usage):
    """USD from list prices for CLIs that report tokens only; None for unknown models."""
    prices = LIST_PRICES.get(model)
    if not prices or not usage:
        return None
    cached = usage.get("cache_read_input_tokens", usage.get("cached_input_tokens", 0))
    if "cache_read_input_tokens" in usage:
        uncached = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    else:
        uncached = usage.get("input_tokens", 0) - cached
    return (uncached * prices["input"] + cached * prices["cached_input"]) / 1e6 + usage.get(
        "output_tokens", 0
    ) * prices["output"] / 1e6


def run_budget(runs, sample_ids):
    """Per-evaluation means of wall time, cost, tokens and tool calls over the runs' samples.

    Wall time is the harness measurement from browser start to episode export, so it is
    comparable across agents. Cost is the agent CLI's own figure when it reports one
    (Claude Code does; Codex does not). Tokens are split into total input (including
    cache reads and writes), output, and cached input, using each CLI's usage fields.
    """
    totals = Counter()
    counted = Counter()
    cost_sources = set()
    for run in runs:
        run, meta = _run_meta(run)
        model = (meta.get("options") or {}).get("model")
        for sid in meta.get("samples", []):
            if sid not in sample_ids:
                continue
            sample = run / "samples" / sample_name(sid)
            path = sample / "result.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text())
            if not _is_attempt(result):
                continue
            execution = result.get("execution") or {}
            usage = execution.get("usage") or {}
            counted["samples"] += 1
            if result.get("duration_seconds") is not None:
                totals["seconds"] += float(result["duration_seconds"])
                counted["seconds"] += 1
            if execution.get("cost_usd") is not None:
                totals["cost_usd"] += float(execution["cost_usd"])
                counted["cost_usd"] += 1
                cost_sources.add("reported by the agent CLI")
            elif (estimate := estimate_cost(model, usage)) is not None:
                totals["cost_usd"] += estimate
                counted["cost_usd"] += 1
                cost_sources.add(
                    f"estimated from list prices ({model}, {LIST_PRICES[model]['retrieved']})"
                )
            if usage:
                cached = usage.get("cache_read_input_tokens", usage.get("cached_input_tokens", 0))
                if "cache_read_input_tokens" in usage:  # Claude Code: input excludes cache
                    total_input = (
                        usage.get("input_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0)
                        + cached
                    )
                else:  # Codex: input_tokens already includes cached input
                    total_input = usage.get("input_tokens", 0)
                totals["input_tokens"] += total_input
                totals["cached_input_tokens"] += cached
                totals["output_tokens"] += usage.get("output_tokens", 0)
                counted["tokens"] += 1
            conversation = sample / "conversation.jsonl"
            if not conversation.is_file() and result.get("native_webmcp_call_count") is not None:
                totals["tool_calls"] += int(result["native_webmcp_call_count"])
                counted["tool_calls"] += 1
            if conversation.is_file():
                calls = sum(
                    1
                    for line in conversation.read_text().splitlines()
                    if line.startswith('{"') and '"type": "tool_call"' in line
                )
                totals["tool_calls"] += calls
                counted["tool_calls"] += 1

    def mean(total_key, count_key):
        return totals[total_key] / counted[count_key] if counted[count_key] else None

    return dict(
        samples=counted["samples"],
        mean_seconds=mean("seconds", "seconds"),
        mean_cost_usd=mean("cost_usd", "cost_usd"),
        total_cost_usd=totals["cost_usd"] if counted["cost_usd"] else None,
        cost_source=sorted(cost_sources),
        mean_input_tokens=mean("input_tokens", "tokens"),
        mean_cached_input_tokens=mean("cached_input_tokens", "tokens"),
        mean_output_tokens=mean("output_tokens", "tokens"),
        mean_tool_calls=mean("tool_calls", "tool_calls"),
        reported=dict(counted),
    )


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def read_rows(directory):
    rows = {}
    for line in (Path(directory) / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sample_id"]] = row
    return rows


def block_scores(benchmark, block, rows):
    """Score one block under the frozen rules: a sample without a scored row is SR=0, PA=0."""
    protocol = benchmark["evaluation"]["protocol"]
    errors, sr, pa, scd = [], [], [], []
    by_band, by_category = {}, {}
    singleton_ok = True
    for sid, meta in block["samples"].items():
        row = rows.get(sid)
        if row is None or row.get("status") != "scored":
            errors.append(dict(sample_id=sid, error=(row or {}).get("error", "missing row")))
            success, accuracy = 0, 0.0
        else:
            if row.get("protocol_version") != protocol:
                raise ValueError(f"{sid}: protocol {row.get('protocol_version')} != {protocol}")
            if row.get("dataset") != block["repo_id"] or row.get("revision") != block["revision"]:
                raise ValueError(f"{sid}: dataset/revision differs from the benchmark")
            success, accuracy = int(row["SR"]), float(row["PA"])
            scd.append(float(row["SCD"]))
            if any(len(group) != 1 for group in row.get("equivalence_groups") or []):
                singleton_ok = False
        sr.append(success)
        pa.append(accuracy)
        by_band.setdefault(str(meta["band"]), []).append(success)
        by_category.setdefault(str(meta["category"]), []).append(success)
    return dict(
        block=block["name"],
        source=block["source"],
        reference_mode=block["reference_mode"],
        expected=len(block["samples"]),
        scored=len(block["samples"]) - len(errors),
        errors=errors,
        SR=_mean(sr),
        PA=_mean(pa),
        SCD=_mean(scd),
        SR_wilson95=wilson_interval(sum(sr), len(sr)),
        by_band={k: _mean(v) for k, v in sorted(by_band.items())},
        by_category={k: _mean(v) for k, v in sorted(by_category.items())},
        singleton_groups_ok=singleton_ok,
    )


def summarize_benchmark(benchmark, block_rows, outcomes=None, budgets=None, task_match=None):
    """Two-level Overall: mean over sources of the mean over that source's blocks."""
    blocks = {block["name"]: block for block in benchmark["blocks"]}
    unknown = set(block_rows) - set(blocks)
    if unknown:
        raise ValueError(f"Unknown blocks: {sorted(unknown)}")
    scores = {}
    for name in blocks:
        scores[name] = block_scores(benchmark, blocks[name], block_rows.get(name, {}))
        scores[name]["agent_outcomes"] = (outcomes or {}).get(name, {})
        scores[name]["budget"] = (budgets or {}).get(name)
        scores[name]["task_matches"] = (task_match or {}).get(name)
        scores[name]["evaluated"] = name in block_rows
    # Blocks without any run are listed but stay out of the means; missing samples
    # inside an evaluated block already count as SR=0. Status reports completeness.
    sources = {}
    for source in benchmark["sources"]:
        names = [b["name"] for b in benchmark["blocks"] if b["source"] == source]
        sources[source] = _mean(scores[n]["SR"] for n in names if scores[n]["evaluated"])
    overall = _mean(v for v in sources.values() if v is not None)
    complete = all(s["evaluated"] and s["scored"] == s["expected"] for s in scores.values())
    return dict(
        benchmark=benchmark["benchmark"],
        status="complete" if complete else "incomplete",
        rule=benchmark["aggregation"]["rule"],
        missing_sample_rule=benchmark["aggregation"]["missing_sample_rule"],
        blocks=scores,
        sources=sources,
        overall_SR=overall,
    )


def format_summary(summary):
    """Compact SR report: one line per block, per source, then Overall."""

    def pct(value):
        return "  n/a " if value is None else f"{100 * value:5.1f}%"

    lines = [f"{summary['benchmark']}: {summary['status']}"]
    for name, block in summary["blocks"].items():
        interval = block["SR_wilson95"]
        span = (
            "" if interval is None else f"  [95% {100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"
        )
        note = "" if block["evaluated"] else "  (no run)"
        if block.get("task_matches") is False:
            note += "  (legacy task text)"
        lines.append(
            f"  {name:28} SR {pct(block['SR'])}{span}"
            f"  ({block['scored']}/{block['expected']} scored){note}"
        )
        budget = block.get("budget")
        if budget and budget["samples"]:
            cost = budget["mean_cost_usd"]
            lines.append(
                f"  {'':28} per eval: "
                f"{(budget['mean_seconds'] or 0) / 60:.1f} min, "
                + ("cost n/a" if cost is None else f"${cost:.2f}")
                + f", in {(budget['mean_input_tokens'] or 0) / 1000:.0f}k"
                f" / out {(budget['mean_output_tokens'] or 0) / 1000:.1f}k tokens, "
                f"{budget['mean_tool_calls'] or 0:.0f} tool calls"
            )
    for source, value in summary["sources"].items():
        lines.append(f"  {source + ' (source)':28} SR {pct(value)}")
    lines.append(f"  {'Overall':28} SR {pct(summary['overall_SR'])}")
    return "\n".join(lines)


def _reusable(directory, runs):
    """True when ``directory`` already holds a complete evaluation of exactly these runs."""
    meta = directory / "meta.json"
    if not meta.is_file() or not (directory / "metrics.jsonl").is_file():
        return False
    recorded = json.loads(meta.read_text()).get("runs")
    return recorded == [str(Path(run).resolve()) for run in runs]


def evaluate_benchmark(
    benchmark_path,
    runs,
    *,
    output,
    cache_dir=None,
    similarity=None,
    workers=4,
    accept_task_mismatch=False,
):
    """Score benchmark runs block by block into ``output`` and aggregate; runs stay untouched."""
    from .evaluation import evaluate_runs
    from .similarity import SimilarityConfig

    benchmark = read_benchmark(benchmark_path)
    evaluation = benchmark["evaluation"]
    expected = SimilarityConfig(evaluation["similarity_policy"], evaluation["similarity_threshold"])
    similarity = similarity or expected
    if similarity != expected:
        raise ValueError("Similarity configuration differs from the benchmark protocol")
    grouped = match_runs_to_blocks(benchmark, runs, accept_task_mismatch=accept_task_mismatch)
    task_match = {
        name: all(task_matches(benchmark, run) for run in block_runs)
        for name, block_runs in grouped.items()
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "meta.json",
        dict(
            started_at=now(),
            benchmark=str(Path(benchmark_path).resolve()),
            runs={name: [str(r) for r in block_runs] for name, block_runs in grouped.items()},
        ),
    )
    block_rows, outcomes, budgets = {}, {}, {}
    for name, block_runs in grouped.items():
        # An existing block evaluation over the same runs is reused, so a growing set
        # of runs only scores the blocks that changed since the previous call.
        if not _reusable(output / name, block_runs):
            if (output / name).exists():
                raise ValueError(f"{output / name} holds an evaluation of different runs")
            evaluate_runs(
                block_runs,
                output / name,
                cache_dir=cache_dir,
                similarity=similarity,
                workers=workers,
            )
        block_rows[name] = read_rows(output / name)
        block = next(b for b in benchmark["blocks"] if b["name"] == name)
        outcomes[name] = agent_outcomes(block_runs, set(block["samples"]))
        budgets[name] = run_budget(block_runs, set(block["samples"]))
    summary = summarize_benchmark(benchmark, block_rows, outcomes, budgets, task_match)
    summary["output"] = str(output)
    write_json(output / "benchmark_summary.json", summary)
    return summary
