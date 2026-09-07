"""Serial dataset conversion into reusable initial episodes and isolated logs."""

import json
from contextlib import closing
from dataclasses import asdict
from numbers import Integral
from pathlib import Path

from .adapters import get_adapter
from .artifacts import experiment, read_config, write_json, write_task
from .loading import load_samples
from .models import PreparationConfig
from .preparation import prepare_sample


def convert_dataset(
    dataset: str,
    *,
    sample_ids=None,
    limit=None,
    all_samples: bool = False,
    revision=None,
    cache_dir=None,
    model_format="xml",
    preparation: PreparationConfig | None = None,
    output=Path("data"),
    logs=Path("logs"),
):
    """Convert one explicit selection, returning its configuration/count/log summary.

    Exactly one of sample_ids, limit or all_samples is required. Existing archives
    are regenerated and verified, never blindly skipped. Errors propagate after
    preserving progress and printing the failed run's log location.
    """
    if model_format not in ("xml", "mjb"):
        raise ValueError("Unsupported model format")
    adapter = get_adapter(dataset)
    if not isinstance(all_samples, bool):
        raise ValueError("all_samples must be a boolean")
    if sum((sample_ids is not None, limit is not None, all_samples)) != 1:
        raise ValueError("Select exactly one of sample_ids, limit or all_samples")
    if sample_ids is not None:
        if isinstance(sample_ids, (str, bytes)):
            raise ValueError("sample_ids must be a nonempty sequence of IDs")
        sample_ids = tuple(sample_ids)
        if not sample_ids or any(not isinstance(i, str) or not i for i in sample_ids):
            raise ValueError("sample_ids must contain nonempty strings")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")
    config = preparation or PreparationConfig()
    inputs = dict(
        dataset=adapter.REPO_ID,
        revision=revision if revision is not None else adapter.DEFAULT_REVISION,
        sample_ids=sample_ids,
        limit=limit,
        all_samples=all_samples,
        preparation=asdict(config),
        cache_dir=None if cache_dir is None else str(cache_dir),
        output=str(output),
        streaming=False,
        model_format=model_format,
    )
    summary = None
    try:
        with experiment("conversion", logs=logs, inputs=inputs) as (directory, meta, metrics):
            summary = dict(config_directories=[], converted_count=0, log_directory=str(directory))
            meta.update(configurations={}, config_directories=[], current_sample_id=None)
            metrics.update(samples=[], converted_count=0)
            try:
                with closing(
                    load_samples(
                        dataset,
                        revision=revision,
                        sample_ids=sample_ids,
                        limit=limit,
                        cache_dir=cache_dir,
                        streaming=False,
                    )
                ) as sources:
                    for source in sources:
                        meta["current_sample_id"] = source.sample_id
                        meta["resolved_revision"] = source.revision
                        write_json(directory / "meta.json", meta)
                        print(f"Converting: {source.sample_id}", flush=True)
                        task = write_task(
                            prepare_sample(source, config), output, model_format=model_format
                        )
                        metrics["samples"].append(task)
                        metrics["converted_count"] = len(metrics["samples"])
                        configuration = task["config_directory"]
                        meta["config_directories"] = sorted(
                            {row["config_directory"] for row in metrics["samples"]}
                        )
                        summary.update(
                            config_directories=meta["config_directories"],
                            converted_count=metrics["converted_count"],
                        )
                        if configuration not in meta["configurations"]:
                            meta["configurations"][configuration] = read_config(configuration)[
                                "identity"
                            ]
                        meta["current_sample_id"] = None
                        write_json(directory / "metrics.json", metrics)
                        write_json(directory / "meta.json", meta)
                        print(json.dumps(task), flush=True)
                        # Do not retain the previous high-resolution source while loading the next.
                        del source
                for configuration in meta["config_directories"]:
                    read_config(configuration)
            except BaseException as error:
                metrics["failure"] = dict(
                    sample_id=meta["current_sample_id"],
                    error=f"{type(error).__name__}: {error}",
                )
                raise
    finally:
        if summary is not None:
            print(f"Converted: {summary['converted_count']}", flush=True)
            for configuration in summary["config_directories"]:
                print(f"Configuration: {configuration}", flush=True)
            print(f"Log: {summary['log_directory']}", flush=True)
    return summary
