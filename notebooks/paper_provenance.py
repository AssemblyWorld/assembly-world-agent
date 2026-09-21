"""Write compact manuscript provenance after executing the paper notebooks."""

import platform
from collections import Counter

import numpy as np
import paper_analysis as pa
import paper_figures as figures
import scipy


def write():
    analyses, missing = figures.load_analyses(pa.load_benchmark())
    if missing or any(not d["scores"] for d in analyses):
        raise ValueError("Finalize provenance only after all 800 trajectory analyses are validated")
    inputs = {}
    for pattern in [
        "metrics.jsonl",
        "metrics_summary.json",
        "benchmark_summary.json",
        "run.json",
        "experiment.json",
    ]:
        for path in pa.RESULTS.rglob(pattern):
            inputs[str(path.relative_to(pa.RESULTS))] = pa.digest(path)
    for pattern in ["input.json", "result.json"]:
        for path in (pa.RESULTS / "assemblyworldbench").rglob(pattern):
            inputs[str(path.relative_to(pa.RESULTS))] = pa.digest(path)
    notebook_dir = pa.ROOT / "notebooks"
    implementation = {}
    for pattern in [
        "paper_*.py",
        "paper_*.ipynb",
        "tool_call_timeline.ipynb",
        "run_paper_notebooks.py",
    ]:
        for path in notebook_dir.glob(pattern):
            implementation[path.name] = pa.digest(path)
    coverage = Counter()
    states = 0
    for path in (pa.CACHE / "trajectories").rglob("*.json"):
        d = pa.read_json(path)
        if d["scores"]:
            coverage[d["system"]] += 1
            states += len(d["scores"])
    template = pa.read_json(pa.PAPER / "provenance/template.json")
    for filename, checksum in template["files"].items():
        if pa.digest(pa.PAPER / filename) != checksum:
            raise ValueError(f"Official template file changed: {filename}")
    original = pa.read_json(pa.CACHE / "results_file_snapshot.json")
    current = {
        str(p.relative_to(pa.RESULTS)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in pa.RESULTS.rglob("*")
        if p.is_file()
    }
    if original != current:
        raise ValueError("The input results package was modified during analysis")
    artifacts = {}
    groups = [
        "cross_domain",
        "efficiency",
        "reference_analysis",
        "behavior",
        "qualitative",
        "failures",
        "system_comparison",
        "appendix_behavior",
        "appendix_results",
    ]
    for group in groups:
        for path in (pa.PAPER / "fig" / group).glob("*.pdf"):
            artifacts[str(path.relative_to(pa.PAPER))] = pa.digest(path)
    for path in (pa.PAPER / "tab").glob("*.tex"):
        artifacts[str(path.relative_to(pa.PAPER))] = pa.digest(path)
    for name in ["main.tex", "preamble.tex"]:
        artifacts[name] = pa.digest(pa.PAPER / name)
    reports = {}
    for name in [
        "coverage",
        "source_results_audit",
        "timestamp_audit",
        "results_summary",
        "qualitative_selection",
        "reference_example",
        "trajectory_example",
        "failure_examples",
        "quality_declines",
        "regression_audit",
        "failure_sensitivity",
        "state_score_protocol",
        "retained_table_audit",
        "budget_terminal_audit",
        "refinement_audit",
        "initialization_audit",
    ]:
        path = pa.CACHE / f"{name}.json"
        if path.exists():
            reports[name] = dict(sha256=pa.digest(path), cache_file=path.name)
    output = dict(
        analysis_version=pa.VERSION,
        input_root="assembly-world-agent/results",
        derived_cache="assembly-world-agent/notebooks/.cache/paper-analysis",
        experimental_inputs="Exported results only; no legacy logs or scratchpad inputs",
        evaluator_inputs="Existing identity-keyed dataset cache resolved from each exported input.json",
        results_files_unchanged=len(current),
        input_hashes=inputs,
        implementation_hashes=implementation,
        artifact_hashes=artifacts,
        local_runtime=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__
        ),
        scored_episodes=dict(coverage),
        scored_recorded_state_indices=states,
        unique_nonterminal_pose_cache_entries=len(list((pa.CACHE / "state_scores").glob("*.json"))),
        reports=reports,
        official_style_unchanged=True,
        review_status="Internal content draft; evidence gaps and submission layout remain explicit. Visual inspection is agent-assisted, not a human annotation study.",
    )
    output["remote_scoring"] = []
    for filename in [
        "remote_scoring_first_pass.json",
        "remote_scoring.json",
        "remote_scoring_tail.json",
    ]:
        remote = pa.CACHE / filename
        if remote.exists():
            r = pa.read_json(remote)
            output["remote_scoring"].append(
                dict(
                    cache_file=filename,
                    source_hashes=r["source_hashes"],
                    returned_groups=len(r["reports"]),
                    report_sha256=pa.digest(remote),
                )
            )
    path = pa.PAPER / "provenance/paper_analysis.json"
    pa.save_json(path, output)
    return path


if __name__ == "__main__":
    print(write())
