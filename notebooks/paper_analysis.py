"""Reproducible paper analysis of the immutable results package.

Notebook entry points own orchestration. Experiment records are read only from
results; evaluator inputs are resolved through the existing keyed data cache.
Derived statistics stay under notebooks/.cache, and paper figures are PDFs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "notebooks/.cache/paper-analysis"
PAPER = ROOT.parent / "AssemblyWorldBench"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SYSTEMS = {
    "gpt-6-astra": "GPT-6 Astra",
    "claude-fable-5-1": "Claude Fable",
    "claude-opus-5": "Claude Opus",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "claude-sonnet-5": "Claude Sonnet",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "qwen3.8-max-litellm": "Qwen Max",
    "deepseek-v4.1-flash": "DeepSeek Flash",
}
SHORT_SYSTEMS = dict(
    zip(SYSTEMS, ["Astra", "Fable", "Opus", "Sol", "Sonnet", "Terra", "Qwen", "DeepSeek"])
)
REPRESENTATIVES = ["gpt-6-astra", "claude-fable-5-1", "gpt-5.6-sol", "claude-sonnet-5"]
BLOCKS = [
    "partnet-none",
    "partnet-final-image",
    "ikea-manualbook",
    "assemblybench-manualbook",
    "fantastic-breaks-none",
]
BLOCK_LABELS = [
    "PartNet / none",
    "PartNet / image",
    "IKEA / manual",
    "Industrial / manual",
    "Fractures / none",
]
WEIGHTS = dict(zip(BLOCKS, [0.125, 0.125, 0.25, 0.25, 0.25]))
COLORS = dict(
    zip(
        SYSTEMS,
        ["#1769aa", "#de7722", "#b74767", "#38a6a5", "#8466aa", "#7f8c48", "#936c4e", "#777777"],
    )
)
BUDGETS = [1, 2, 5, 10, 20, 30, 45, 60]
VERSION = "paper-analysis-v2"


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)
        )
        + "\n"
    )


def configure(results=RESULTS, paper=PAPER, cache=CACHE):
    global RESULTS, PAPER, CACHE
    RESULTS, PAPER, CACHE = Path(results).resolve(), Path(paper).resolve(), Path(cache).resolve()
    if (
        RESULTS.name != "results"
        or not (RESULTS / "assemblyworldbench/benchmark/benchmark.json").exists()
    ):
        raise ValueError("Expected the exported results package, not an experiment log directory")
    if CACHE.is_relative_to(RESULTS) or PAPER.is_relative_to(RESULTS):
        raise ValueError("Analysis outputs must stay outside the immutable results package")
    CACHE.mkdir(parents=True, exist_ok=True)
    snapshot = CACHE / "results_file_snapshot.json"
    if not snapshot.exists():
        save_json(
            snapshot,
            {
                str(p.relative_to(RESULTS)): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in RESULTS.rglob("*")
                if p.is_file()
            },
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig, group, name=None):
    path = PAPER / "fig" / group / f"{name or group}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": VERSION})
    plt.close(fig)
    return path


def load_benchmark():
    benchmark = read_json(RESULTS / "assemblyworldbench/benchmark/benchmark.json")
    meta = {b["name"]: b for b in benchmark["blocks"]}
    rows, inputs = [], {}
    for system in SYSTEMS:
        for block in BLOCKS:
            directory = RESULTS / "assemblyworldbench" / system / block
            run = read_json(directory / "run.json")
            metrics_path = directory / "evaluation/metrics.jsonl"
            metrics = {r["sample_id"]: r for r in read_jsonl(metrics_path)}
            inputs[str(metrics_path.relative_to(ROOT))] = digest(metrics_path)
            for sid in run["samples"]:
                sd = directory / "samples" / sid.replace("/", "--")
                result = read_json(sd / "result.json")
                inp = read_json(sd / "input.json")
                m = metrics[sid]
                bm = meta[block]["samples"][sid]
                rows.append(
                    dict(
                        system=system,
                        label=SYSTEMS[system],
                        block=block,
                        source=meta[block]["source"],
                        sample_id=sid,
                        parts=inp["parts"],
                        category=bm["category"],
                        band=bm["band"],
                        PA=float(m.get("PA", 0)),
                        SR=int(m.get("SR", 0)),
                        SCD=m.get("SCD", np.nan),
                        scored=m.get("status") == "scored",
                        status=result.get("status"),
                        execution_status=(result.get("execution") or {}).get("status"),
                        outcome=(result.get("agent_outcome") or {}).get("status"),
                        duration=result.get("duration_seconds", np.nan),
                        has_episode=(sd / "final.episode.zip").exists(),
                        sample_dir=str(sd),
                        metric=m,
                        input=inp,
                        result=result,
                        expected=run["config"]["samples"][sid],
                        identity=run["config"]["identity"],
                    )
                )
    frame = pd.DataFrame(rows)
    assert len(frame) == 800 and not frame.duplicated(["system", "block", "sample_id"]).any()
    save_json(CACHE / "input_manifest.json", inputs)
    return frame


def overall(frame, metric):
    return float(sum(frame[frame.block == b][metric].mean() * WEIGHTS[b] for b in BLOCKS))


def source_weighted_median(frame, metric):
    ordered = frame.sort_values(metric)
    counts = frame.block.value_counts()
    weights = np.array([WEIGHTS[b] / counts[b] for b in ordered.block])
    return float(ordered[metric].iloc[np.searchsorted(np.cumsum(weights), 0.5 - 1e-12)])


def shape_values(frame, metric):
    """Four sources x twenty shapes; PartNet conditions share each bootstrap draw."""
    chunks = []
    pn = frame[frame.block.str.startswith("partnet")].pivot(
        index="sample_id", columns="block", values=metric
    )
    chunks.append(pn[BLOCKS[:2]].mean(axis=1).sort_index().to_numpy())
    for block in BLOCKS[2:]:
        chunks.append(frame[frame.block == block].sort_values("sample_id")[metric].to_numpy())
    assert all(len(c) == 20 for c in chunks)
    return np.asarray(chunks)


def bootstrap(values, seed=20260921, draws=10000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.shape[1], size=(draws, values.shape[0], values.shape[1]))
    samples = values[np.arange(values.shape[0])[None, :, None], idx].mean(axis=(1, 2))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def paired_randomization(values, seed=20260921, draws=100000):
    """Two-sided sign randomization of shape-paired differences.

    Each PartNet shape's two conditions have already been averaged, so their
    system labels are swapped together. The null assumes within-shape label
    exchangeability. All four sources have twenty shapes and equal weight.
    """
    differences = np.asarray(values).ravel()
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, draws, 5000):
        signs = 2 * rng.integers(0, 2, size=(min(5000, draws - start), len(differences))) - 1
        exceed += int((np.abs((signs * differences).mean(axis=1)) >= observed - 1e-12).sum())
    return (exceed + 1) / (draws + 1)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1))
    )
    return adjusted.tolist()


def results_analysis(frame):
    initializations = {}
    for row in frame.to_dict("records"):
        family = "partnet" if row["block"].startswith("partnet") else row["block"]
        key = (family, row["sample_id"])
        initializations.setdefault(key, set()).add(
            (row["input"]["episode_id"], row["input"]["sha256"])
        )
    if len(initializations) != 80 or any(len(values) != 1 for values in initializations.values()):
        raise ValueError(
            "Benchmark systems or paired PartNet conditions do not share initial episodes"
        )
    save_json(
        CACHE / "initialization_audit.json",
        dict(
            evaluations=len(frame),
            unique_shapes=len(initializations),
            same_initial_episode_across_systems_and_partnet_conditions=True,
            identities=[
                dict(
                    source=source,
                    sample_id=sid,
                    episode_id=next(iter(values))[0],
                    initial_sha256=next(iter(values))[1],
                )
                for (source, sid), values in sorted(initializations.items())
            ],
        ),
    )
    rows, paired, reference = [], [], []
    for system in SYSTEMS:
        f = frame[frame.system == system]
        summary = read_json(
            RESULTS / "assemblyworldbench" / system / "evaluation/benchmark_summary.json"
        )
        sr = overall(f, "SR")
        assert abs(sr - summary["overall_SR"]) < 1e-12
        budget = [summary["blocks"][b]["budget"] for b in BLOCKS]

        def budget_mean(key, count):
            den = sum(x["reported"][count] for x in budget)
            return (
                sum((x.get(key) or 0) * x["reported"][count] for x in budget) / den if den else None
            )

        rows.append(
            dict(
                system=system,
                SR=sr,
                SR_ci=bootstrap(shape_values(f, "SR")),
                PA=overall(f, "PA"),
                PA_ci=bootstrap(shape_values(f, "PA")),
                mean_minutes=budget_mean("mean_seconds", "seconds") / 60,
                cost=budget_mean("mean_cost_usd", "cost_usd"),
                cost_n=sum(x["reported"]["cost_usd"] for x in budget),
                calls=budget_mean("mean_tool_calls", "tool_calls"),
                tokens=(
                    budget_mean("mean_input_tokens", "tokens")
                    + budget_mean("mean_output_tokens", "tokens")
                )
                / 1e6,
                scored=int(f.scored.sum()),
                archives=int(f.has_episode.sum()),
                runtime_failed=int((f.status == "failed").sum()),
                timeouts=int((f.execution_status == "timeout").sum()),
                other_failed=int(
                    ((f.status == "failed") & (f.execution_status != "timeout")).sum()
                ),
            )
        )
        pn = f[f.block.str.startswith("partnet")]
        for metric in ["PA", "SR"]:
            p = pn.pivot(index="sample_id", columns="block", values=metric).sort_index()
            delta = (p[BLOCKS[1]] - p[BLOCKS[0]]).to_numpy()
            reference.append(
                dict(
                    system=system,
                    metric=metric,
                    delta=float(delta.mean()),
                    ci=bootstrap(delta[None, :]),
                    positive=int((delta > 0).sum()),
                    negative=int((delta < 0).sum()),
                    n=len(delta),
                )
            )
    from itertools import combinations

    for a, b in combinations(SYSTEMS, 2):
        for metric in ["SR", "PA"]:
            delta = shape_values(frame[frame.system == a], metric) - shape_values(
                frame[frame.system == b], metric
            )
            paired.append(
                dict(
                    a=a,
                    b=b,
                    metric=metric,
                    delta=float(delta.mean()),
                    ci=bootstrap(delta),
                    p_randomization=paired_randomization(delta),
                )
            )
    for row, adjusted in zip(paired, holm_adjust([r["p_randomization"] for r in paired])):
        row["p_holm"] = adjusted
    save_json(
        CACHE / "results_summary.json",
        dict(
            systems=rows,
            paired=paired,
            reference=reference,
            bootstrap="10000 within-source paired shape resamples; seed=20260921; pointwise percentile intervals",
            multiplicity="Two-sided 100000-draw paired label randomization; seed=20260921; Holm correction across 28 system pairs x 2 metrics (56 tests)",
        ),
    )
    pd.DataFrame(rows).to_csv(CACHE / "systems.csv", index=False)
    pd.DataFrame(paired).to_csv(CACHE / "paired.csv", index=False)
    pd.DataFrame(reference).to_csv(CACHE / "reference.csv", index=False)
    return rows, paired, reference


def audit_source_results():
    """Reproduce the source-level aggregates without consulting legacy run paths."""
    verified = []
    for summary_path in sorted(RESULTS.rglob("metrics_summary.json")):
        if "assemblyworldbench" in summary_path.parts:
            continue
        summary = read_json(summary_path)
        records_path = summary_path.with_name("metrics.jsonl")
        records = read_jsonl(records_path)
        excluded = set(summary.get("excluded_ids", []))
        included = [
            r for r in records if r.get("status") == "scored" and r["sample_id"] not in excluded
        ]
        metrics = [m for m in ["SCD", "PA", "SR", "RMSE_R", "RMSE_T", "CD"] if m in summary]
        means = {m: float(np.mean([r[m] for r in included])) for m in metrics}
        for metric, value in means.items():
            if not np.isclose(value, summary[metric], atol=1e-9, rtol=1e-9):
                raise ValueError(f"Source aggregate mismatch: {summary_path}: {metric}")
        block = next(p for p in summary_path.parents if (p / "samples").exists())
        checked = 0
        for row in records:
            if row.get("episode_sha256"):
                episode = (
                    block / "samples" / row["sample_id"].replace("/", "--") / "final.episode.zip"
                )
                if digest(episode) != row["episode_sha256"]:
                    raise ValueError(f"Source archive checksum mismatch: {episode}")
                checked += 1
        verified.append(
            dict(
                source=str(summary_path.relative_to(RESULTS)),
                summary_sha256=digest(summary_path),
                records_sha256=digest(records_path),
                configured=len(records),
                reported=len(included),
                excluded=sorted(excluded),
                episode_checksums_verified=checked,
                means=means,
            )
        )
    missing = []
    missing.append(
        dict(
            analysis="Historical dagger control in Table 1",
            status="missing",
            detail="The export contains eight current systems, not the earlier Astra control. Its retained manuscript entries are excluded from all new analyses.",
        )
    )
    table_none = RESULTS / "partnet/gpt-6-astra/table-none"
    if not table_none.exists():
        missing.append(
            dict(
                analysis="PartNet Table, no-reference, legacy Table 2 cells",
                status="missing",
                detail="The existing manuscript values are retained but cannot be reproduced from this results package; no benchmark subset or historical log is substituted.",
            )
        )
    report = dict(verified=verified, missing=missing)
    save_json(CACHE / "source_results_audit.json", report)
    return report


def supplementary_results(frame, rows, paired, reference):
    # Paired reference conditions and source-stratified complexity.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), constrained_layout=True)
    for i, system in enumerate(SYSTEMS):
        r = next(r for r in reference if r["system"] == system and r["metric"] == "PA")
        axes[0].errorbar(
            r["delta"],
            i,
            xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]],
            fmt="o",
            color=COLORS[system],
            capsize=2,
        )
        f = frame[(frame.system == system) & frame.block.str.startswith("partnet")]
        vals = f.groupby("block").SR.mean()
        axes[1].plot([0, 1], [vals[BLOCKS[0]], vals[BLOCKS[1]]], "o-", color=COLORS[system], lw=1)
    axes[0].set_yticks(range(8), SYSTEMS.values())
    axes[0].invert_yaxis()
    axes[0].axvline(0, color=".6", lw=0.7)
    axes[0].set_xlabel("Paired PA difference (image - none)")
    axes[1].set_xticks([0, 1], ["No reference", "Final image"])
    axes[1].set_ylabel("Shape success rate")
    save_figure(fig, "reference_analysis")
    fig, axes = plt.subplots(1, 5, figsize=(7, 2.0), sharey=True, constrained_layout=True)
    complexity = []
    for ax, b, label in zip(axes, BLOCKS, BLOCK_LABELS):
        for s in REPRESENTATIVES:
            f = frame[(frame.system == s) & (frame.block == b)]
            for band, g in f.groupby("band"):
                complexity.append(
                    dict(
                        system=s,
                        block=b,
                        band=band,
                        n=len(g),
                        parts=float(g.parts.mean()),
                        SR=float(g.SR.mean()),
                        PA=float(g.PA.mean()),
                    )
                )
            a = pd.DataFrame(
                [x for x in complexity if x["system"] == s and x["block"] == b]
            ).sort_values("parts")
            ax.plot(a.parts, a.SR, "o-", color=COLORS[s], ms=3, lw=1, label=SYSTEMS[s])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Mean part count")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("SR")
    axes[-1].legend(loc="upper right", fontsize=5)
    save_figure(fig, "appendix_results", "complexity")
    save_json(CACHE / "complexity.json", complexity)
    # Sensitivity uses the saved matched errors without changing alignment or matching.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), constrained_layout=True)
    thresholds = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]
    sensitivity = []
    for s in SYSTEMS:
        f = frame[frame.system == s].copy()
        for t in thresholds:
            f["tau_PA"] = f.metric.map(
                lambda m: (
                    np.mean([p["chamfer"] <= t for p in m.get("parts", [])])
                    if m.get("parts")
                    else 0
                )
            )
            f["tau_SR"] = f.metric.map(
                lambda m: int(all(p["chamfer"] <= t for p in m["parts"])) if m.get("parts") else 0
            )
            sensitivity.append(
                dict(system=s, threshold=t, PA=overall(f, "tau_PA"), SR=overall(f, "tau_SR"))
            )
        a = pd.DataFrame([x for x in sensitivity if x["system"] == s])
        for ax, metric in zip(axes, ["PA", "SR"]):
            ax.plot(a.threshold, a[metric], "o-", label=SYSTEMS[s], color=COLORS[s], ms=3)
            ax.set_xscale("log")
            ax.set_xlabel("Part CD threshold")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.03)
    axes[1].legend(fontsize=6, ncol=2)
    save_figure(fig, "appendix_results", "thresholds")
    save_json(CACHE / "thresholds.json", sensitivity)
    calibration = []
    for s in SYSTEMS:
        f = frame[frame.system == s]
        valid = f[f.outcome.notna()]
        completed = valid[valid.outcome == "completed"]
        calibration.append(
            dict(
                system=s,
                reports=len(valid),
                completed=len(completed),
                successes=int(completed.SR.sum()),
                precision=float(completed.SR.mean()) if len(completed) else None,
            )
        )
    save_json(CACHE / "calibration.json", calibration)
    # Whole-shape distances for every system, with the paper's display multiplier.
    fig, axes = plt.subplots(4, 2, figsize=(7, 7), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(20260921)
    for ax, system in zip(axes.flat, SYSTEMS):
        for j, block in enumerate(BLOCKS):
            f = frame[(frame.system == system) & (frame.block == block)]
            for outcome in [0, 1]:
                vals = 1000 * f.loc[f.SR == outcome, "SCD"].dropna().to_numpy()
                if np.any(vals <= 0):
                    raise ValueError("Nonpositive SCD needs an explicit display convention")
                center = j + (outcome - 0.5) * 0.3
                ax.scatter(
                    center + rng.uniform(-0.045, 0.045, len(vals)),
                    vals,
                    s=10,
                    alpha=0.65,
                    color=["#bd5d4c", "#1769aa"][outcome],
                    label=["Failed assembly", "Successful assembly"][outcome] if j == 0 else None,
                )
                if len(vals):
                    ax.plot([center - 0.1, center + 0.1], [np.median(vals)] * 2, color="k", lw=1)
        ax.set_xticks(range(5), ["PN / NR", "PN / image", "IKEA", "AB", "FB"], fontsize=9)
        ax.set_title(SHORT_SYSTEMS[system], fontsize=9)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("SCD (×1000)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    save_figure(fig, "appendix_results", "shape_distance")


def audit_available():
    files = sorted(str(p.relative_to(RESULTS)) for p in RESULTS.rglob("*") if p.is_file())
    missing = {
        "robustness_pilots": {
            "status": "missing",
            "required": "Per-attempt records for repeated/layout/missing/distractor/observation conditions",
        },
        "garf_hybrid": {
            "status": "missing",
            "required": "Standard and agent-initialized GARF per-object outputs with checkpoint, seeds and shared protocol",
        },
    }
    # Discovery is reported, never silently interpreted as a complete experimental protocol.
    candidates = [
        p for p in files if any(k in p.lower() for k in ("pilot", "hybrid", "refinement"))
    ]
    result = dict(
        version=VERSION,
        input_root="results",
        files=len(files),
        candidate_files=candidates,
        missing=missing,
        garf_diagnostic="fantastic-breaks/gpt-6-astra/fantastic-breaks-none/evaluation/garf",
    )
    save_json(CACHE / "coverage.json", result)
    return result


configure()
