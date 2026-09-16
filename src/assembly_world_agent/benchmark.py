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


def block_for_run(benchmark, run):
    """The unique block a run belongs to, or an error naming what differs."""
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
    if sha256(meta.get("task", "").encode()) != block["prompt_sha256"]:
        raise ValueError(f"{run.name}: task text differs from block {block['name']}")
    return block


def match_runs_to_blocks(benchmark, runs):
    """Group run directories by block; every run must belong to exactly one block."""
    grouped = {}
    for run in runs:
        block = block_for_run(benchmark, run)
        grouped.setdefault(block["name"], []).append(Path(run).resolve())
    return grouped


def agent_outcomes(runs, sample_ids):
    """Count recorded agent outcome statuses over the runs' samples (refusals show as unable)."""
    counter = Counter()
    for run in runs:
        run, meta = _run_meta(run)
        for sid in meta.get("samples", []):
            if sid not in sample_ids:
                continue
            path = run / "samples" / sample_name(sid) / "result.json"
            result = json.loads(path.read_text()) if path.is_file() else {}
            outcome = result.get("agent_outcome") or {}
            counter[str(outcome.get("status") or "none")] += 1
    return dict(sorted(counter.items()))


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


def summarize_benchmark(benchmark, block_rows, outcomes=None):
    """Two-level Overall: mean over sources of the mean over that source's blocks."""
    blocks = {block["name"]: block for block in benchmark["blocks"]}
    unknown = set(block_rows) - set(blocks)
    if unknown:
        raise ValueError(f"Unknown blocks: {sorted(unknown)}")
    scores = {}
    for name in blocks:
        scores[name] = block_scores(benchmark, blocks[name], block_rows.get(name, {}))
        scores[name]["agent_outcomes"] = (outcomes or {}).get(name, {})
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
        lines.append(
            f"  {name:28} SR {pct(block['SR'])}{span}"
            f"  ({block['scored']}/{block['expected']} scored){note}"
        )
    for source, value in summary["sources"].items():
        lines.append(f"  {source + ' (source)':28} SR {pct(value)}")
    lines.append(f"  {'Overall':28} SR {pct(summary['overall_SR'])}")
    return "\n".join(lines)


def evaluate_benchmark(benchmark_path, runs, *, output, cache_dir=None, similarity=None, workers=4):
    """Score benchmark runs block by block into ``output`` and aggregate; runs stay untouched."""
    from .evaluation import evaluate_runs
    from .similarity import SimilarityConfig

    benchmark = read_benchmark(benchmark_path)
    evaluation = benchmark["evaluation"]
    expected = SimilarityConfig(evaluation["similarity_policy"], evaluation["similarity_threshold"])
    similarity = similarity or expected
    if similarity != expected:
        raise ValueError("Similarity configuration differs from the benchmark protocol")
    grouped = match_runs_to_blocks(benchmark, runs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "meta.json",
        dict(
            started_at=now(),
            benchmark=str(Path(benchmark_path).resolve()),
            runs={name: [str(r) for r in block_runs] for name, block_runs in grouped.items()},
        ),
    )
    block_rows, outcomes = {}, {}
    for name, block_runs in grouped.items():
        evaluate_runs(
            block_runs, output / name, cache_dir=cache_dir, similarity=similarity, workers=workers
        )
        block_rows[name] = read_rows(output / name)
        block = next(b for b in benchmark["blocks"] if b["name"] == name)
        outcomes[name] = agent_outcomes(block_runs, set(block["samples"]))
    summary = summarize_benchmark(benchmark, block_rows, outcomes)
    summary["output"] = str(output)
    write_json(output / "benchmark_summary.json", summary)
    return summary
