"""Evaluate saved runs with a documented GARF-style protocol."""

import argparse
import json
from pathlib import Path

from assembly_world_agent.evaluation.garf import evaluate_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    summary = evaluate_runs(
        args.runs, args.output, cache_dir=args.cache_dir, seed=args.seed, limit=args.limit
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
