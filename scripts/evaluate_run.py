"""Evaluate recorded final episodes with free-space SCD, PA and SR."""

import argparse
import json
from pathlib import Path

from assembly_world_agent.evaluation import evaluate_run
from assembly_world_agent.similarity import SimilarityConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--similarity-policy", choices=("source", "geometry"), default="geometry")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Geometry similarity raw squared CD threshold (default: 0.0001; before x1000)",
    )
    args = parser.parse_args()
    if args.similarity_policy == "source" and args.similarity_threshold is not None:
        parser.error("--similarity-threshold applies only to geometry")
    similarity = SimilarityConfig(
        args.similarity_policy,
        args.similarity_threshold if args.similarity_threshold is not None else 1e-4,
    )
    summary = evaluate_run(args.run, cache_dir=args.cache_dir, similarity=similarity)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
