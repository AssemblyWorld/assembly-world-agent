"""Explicit dataset selection for initial episode conversion."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .adapters import get_adapter
from .conversion import convert_dataset
from .models import PreparationConfig

DEFAULT_SAMPLES = ("Bench/applaro", "Chair/reidar", "Table/vittsjo_2")


def _positive_integer(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def _dataset(value):
    try:
        get_adapter(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _arguments(parser, *, generic):
    selection = parser.add_mutually_exclusive_group(required=generic)
    selection.add_argument("--sample-id", action="append")
    if generic:
        selection.add_argument(
            "--limit", type=_positive_integer, help="Convert the first N samples"
        )
    selection.add_argument("--all", action="store_true", help="Convert the entire pinned release")
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--logs", type=Path, default=Path("logs"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from .experiments.browser import DEFAULT_ENVIRONMENT

    run = commands.add_parser("run", help="Run isolated browser WebMCP experiments")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", type=Path)
    selection.add_argument("--dataset", type=_dataset)
    run.add_argument("--config-id")
    run.add_argument("--data", type=Path, default=Path("data"))
    samples = run.add_mutually_exclusive_group()
    samples.add_argument("--sample-id", action="append")
    samples.add_argument("--limit", type=_positive_integer)
    run.add_argument("--agent", choices=("codex", "claude"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument(
        "--effort", choices=("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    )
    run.add_argument("--concurrency", type=_positive_integer, default=1)
    run.add_argument("--timeout-seconds", type=_positive_integer)
    run.add_argument("--prompt-file", type=Path)
    run.add_argument("--manual", type=Path)
    run.add_argument("--logs", type=Path, default=Path("logs"))
    check = commands.add_parser("doctor", help="Check CLI, Chrome and live WebMCP discovery")
    check.add_argument("--agent", choices=("codex", "claude"), required=True)
    for command in (run, check):
        command.add_argument("--environment-url", default=DEFAULT_ENVIRONMENT)
        command.add_argument("--chrome-path")
        command.add_argument("--mcp-command", default="chrome-devtools-mcp")
    resume = commands.add_parser("resume", help="Resume unfinished samples in a new run")
    resume.add_argument("run_directory", type=Path)
    resume.add_argument("--concurrency", type=_positive_integer)
    resume.add_argument("--retry-failed", action="store_true")
    status = commands.add_parser("status", help="Read compact run progress")
    status.add_argument("run_directory", type=Path)
    convert = commands.add_parser("convert", help="Convert an explicit dataset selection")
    convert.add_argument("dataset", type=_dataset, help="Registered short name or full Hub ID")
    _arguments(convert, generic=True)
    convert.add_argument("--model-format", choices=("xml", "mjb"), default="xml")
    ikea = commands.add_parser(
        "convert-ikea", help="Convert the three IKEA pilot samples by default"
    )
    _arguments(ikea, generic=False)
    args = parser.parse_args(argv)
    if args.command in {"run", "doctor", "resume", "status"}:
        from .experiments import runner
        from .experiments.browser import doctor

        if args.command == "run":
            if args.dataset and not args.config_id:
                parser.error("--dataset requires --config-id")
            if args.episode and (args.config_id or args.sample_id or args.limit):
                parser.error("--config-id, --sample-id and --limit require --dataset")
            if args.agent == "claude" and args.effort in {"minimal", "ultra"}:
                parser.error("Claude does not support this effort")
        try:
            if args.command == "status":
                print(json.dumps(runner.status(args.run_directory), indent=2))
                return 0
            if args.command == "resume":
                return asyncio.run(
                    runner.launch(
                        {"concurrency": args.concurrency},
                        source=args.run_directory,
                        retry_failed=args.retry_failed,
                    )
                )
            options = {
                k: str(v.resolve()) if isinstance(v, Path) else v
                for k, v in vars(args).items()
                if k != "command"
            }
            if args.command == "doctor":
                print(json.dumps(asyncio.run(doctor(options)), indent=2))
                return 0
            return asyncio.run(runner.launch(options))
        except Exception as error:
            print(f"Experiment failed: {error}", file=sys.stderr, flush=True)
            return 1
    try:
        config = PreparationConfig(
            sampling_seed=args.sampling_seed, initialization_seed=args.initialization_seed
        )
    except ValueError as error:
        parser.error(str(error))
    selected = args.sample_id
    if args.command == "convert-ikea" and not args.all and selected is None:
        selected = DEFAULT_SAMPLES
    try:
        convert_dataset(
            args.dataset if args.command == "convert" else "ikea-manual",
            sample_ids=selected,
            limit=getattr(args, "limit", None),
            all_samples=args.all,
            preparation=config,
            revision=args.revision,
            cache_dir=args.cache_dir,
            output=args.output,
            logs=args.logs,
            model_format=getattr(args, "model_format", "xml"),
        )
    except Exception as error:
        print(f"Conversion failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0
