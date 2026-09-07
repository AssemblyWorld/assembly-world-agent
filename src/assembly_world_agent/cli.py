"""Explicit dataset selection for initial episode conversion."""

import argparse
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
    convert = commands.add_parser("convert", help="Convert an explicit dataset selection")
    convert.add_argument("dataset", type=_dataset, help="Registered short name or full Hub ID")
    _arguments(convert, generic=True)
    convert.add_argument("--model-format", choices=("xml", "mjb"), default="xml")
    ikea = commands.add_parser(
        "convert-ikea", help="Convert the three IKEA pilot samples by default"
    )
    _arguments(ikea, generic=False)
    args = parser.parse_args(argv)
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
