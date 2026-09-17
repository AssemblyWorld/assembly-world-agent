"""Evaluate recorded final episodes with free-space SCD, PA and SR.

One run directory without --output keeps the historical behaviour and writes
metrics next to the run. Several run directories, an explicit --output, or a
--sample-id filter join the runs by sample identity into a new output directory
and leave the archived runs untouched.

With --benchmark BENCHMARK.json the runs are matched to benchmark blocks, scored
block by block into a new output directory, and aggregated with the benchmark's
frozen rules. Only SR lines are printed unless --json is given.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from assembly_world_agent.benchmark import evaluate_benchmark, format_summary
from assembly_world_agent.evaluation import evaluate_run, evaluate_runs
from assembly_world_agent.similarity import SimilarityConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="New directory for joined metrics")
    parser.add_argument("--sample-id", action="append", help="Restrict scoring to these IDs")
    parser.add_argument("--benchmark", type=Path, help="benchmark.json to score and aggregate")
    parser.add_argument("--json", action="store_true", help="Print the full benchmark summary")
    parser.add_argument(
        "--accept-task-mismatch",
        action="store_true",
        help="Match runs whose task text differs from the block's task.txt; recorded per block",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--similarity-policy", choices=("source", "geometry"))
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Geometry similarity raw squared CD threshold (default: 0.0001; before x1000)",
    )
    args = parser.parse_args()
    if args.similarity_policy == "source" and args.similarity_threshold is not None:
        parser.error("--similarity-threshold applies only to geometry")
    explicit = args.similarity_policy is not None or args.similarity_threshold is not None
    similarity = SimilarityConfig(
        args.similarity_policy or "geometry",
        args.similarity_threshold if args.similarity_threshold is not None else 1e-4,
    )
    if args.benchmark is not None:
        if args.sample_id is not None:
            parser.error("--sample-id cannot be combined with --benchmark")
        output = args.output or Path("logs/assemblyworldbench/evaluation") / datetime.now(
            UTC
        ).strftime("%Y%m%dT%H%M%SZ")
        summary = evaluate_benchmark(
            args.benchmark,
            args.runs,
            output=output,
            cache_dir=args.cache_dir,
            similarity=similarity if explicit else None,
            workers=args.workers,
            accept_task_mismatch=args.accept_task_mismatch,
        )
        print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
        return 0 if summary["status"] == "complete" else 1
    joined = len(args.runs) > 1 or args.output is not None or args.sample_id is not None
    if joined and args.output is None:
        parser.error("--output is required when joining runs or filtering samples")
    if joined:
        summary = evaluate_runs(
            args.runs,
            args.output,
            cache_dir=args.cache_dir,
            similarity=similarity,
            workers=args.workers,
            sample_ids=args.sample_id,
        )
    else:
        summary = evaluate_run(
            args.runs[0], cache_dir=args.cache_dir, similarity=similarity, workers=args.workers
        )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
