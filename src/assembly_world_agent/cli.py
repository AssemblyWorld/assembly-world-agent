"""Bounded IKEA initial episode conversion."""

import argparse
import json
from pathlib import Path

from .artifacts import experiment, read_config, write_task
from .loading import load_samples
from .models import PreparationConfig
from .preparation import prepare_sample

DEFAULT_SAMPLES = ("Bench/applaro", "Chair/reidar", "Table/vittsjo_2")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-ikea", help="Convert the three pilot samples by default")
    selection = convert.add_mutually_exclusive_group()
    selection.add_argument("--sample-id", action="append")
    selection.add_argument(
        "--all", action="store_true", help="Explicitly convert the entire IKEA release"
    )
    convert.add_argument("--sampling-seed", type=int, default=0)
    convert.add_argument("--initialization-seed", type=int, default=0)
    convert.add_argument("--revision")
    convert.add_argument("--cache-dir", type=Path)
    convert.add_argument("--output", type=Path, default=Path("data"))
    convert.add_argument("--logs", type=Path, default=Path("logs"))
    args = parser.parse_args(argv)
    config = PreparationConfig(
        sampling_seed=args.sampling_seed, initialization_seed=args.initialization_seed
    )
    selected = None if args.all else args.sample_id or DEFAULT_SAMPLES
    with experiment(
        "conversion",
        logs=args.logs,
        inputs=dict(dataset="ikea-manual", revision=args.revision, sample_ids=selected),
    ) as (directory, meta, metrics):
        metrics["samples"] = []
        for source in load_samples(
            "ikea-manual", revision=args.revision, sample_ids=selected, cache_dir=args.cache_dir
        ):
            task = write_task(prepare_sample(source, config), args.output)
            metrics["samples"].append(task)
            meta.setdefault("configurations", {})[task["config_directory"]] = read_config(
                task["config_directory"]
            )["identity"]
            print(json.dumps(task), flush=True)
        meta["config_directories"] = sorted({row["config_directory"] for row in metrics["samples"]})
    print(f"Log: {directory}", flush=True)
    return 0
