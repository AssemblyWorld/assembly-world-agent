"""Read-only experiment inputs with atomically replaced metric outputs."""

import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..artifacts import config_id, now, sample_name, write_json
from ..episode_io import read_episode
from ..episodes import sha256
from ..loading import load_samples
from ..models import PROTOCOL_VERSION, PreparationConfig
from ..preparation import prepare_sample
from ..similarity import SimilarityConfig, equivalence_metadata, resolve_equivalence
from ..utils import apply_pose, rotation_matrix
from .cache import (
    cache_entry,
    cache_key,
    cached_inputs,
    evaluation_cache_path,
    read_cache,
    write_cache,
)
from .geometry import (
    IMPROVEMENT_TOLERANCE,
    MAX_ITERATIONS,
    THRESHOLD,
    align,
    score_parts,
    transform,
)
from .inputs import evaluation_scale, final_poses, final_poses_against_initial

PROTOCOL = {
    "version": "assembly-evaluation-v2",
    "reference_commit": "df784cae8ee8ae512436f9e762e99aec47036b9a",
    "distance": "sum of bidirectional mean squared Euclidean nearest-neighbor distances",
    "scd_multiplier": 1000,
    "part_threshold": THRESHOLD,
    "threshold_comparison": "<=",
    "aggregation": "sample macro mean over successfully scored samples",
    "scale": "largest per-part vertex-PCA AABB diagonal = 1; shared fixed divisor",
    "sampling": "4096 area-weighted candidates, 1000 FPS points per part; task sampling seed",
    "alignment": "shared SE(3), minimum encountered whole-shape Chamfer; no scaling/reflection",
    "initializations": "24 proper whole-shape PCA axis mappings plus same-ID part pose transforms",
    "max_iterations": MAX_ITERATIONS,
    "improvement_tolerance": IMPROVEMENT_TOLERANCE,
    "matching": "configured atomic equivalence groups; unclipped Chamfer Hungarian cost",
    "limitations": "Approximate multistart registration. Sampling, alignment and equivalence "
    "groups differ from the original Manual-PA benchmark. No physical validation.",
}


class SampleRef(NamedTuple):
    """Identity of a sample whose evaluation inputs come from the cache, not from HF."""

    sample_id: str
    dataset: str
    revision: str


def prepare_evaluation_inputs(
    directory, expected, identity, source, *, similarity=None, initial=None
):
    """Validate and reconstruct the shared inputs used by scoring and inspection.

    With ``initial`` (the sample's initial episode inside a prepared configuration
    directory) the per-sample evaluation cache is consulted: a hit replaces source
    preparation and validates the final episode against the initial episode; a miss
    is computed from ``source`` as before and written to the cache.
    """
    similarity = similarity or SimilarityConfig()
    sample_id = source.sample_id
    inputs = json.loads((directory / "input.json").read_text())
    if inputs["sample_id"] != sample_id or inputs["revision"] != identity["revision"]:
        raise ValueError("Sample input identity/revision differs")
    for key in ("episode_id", "sha256", "parts"):
        if inputs[key] != expected[key]:
            raise ValueError(f"Sample input {key} differs from run configuration")
    if source.revision != identity["revision"]:
        raise ValueError("Loaded source revision differs")
    path = directory / "final.episode.zip"
    checksum = sha256(path.read_bytes())
    archive_record = directory / "archive.json"
    if archive_record.exists() and json.loads(archive_record.read_text())["sha256"] != checksum:
        raise ValueError("Final episode differs from archival checksum")
    result_path = directory / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text()).get("archive", {})
        if saved.get("sha256") and saved["sha256"] != checksum:
            raise ValueError("Final episode differs from result checksum")
    episode = read_episode(path)
    if episode["manifest"]["id"] != expected["episode_id"]:
        raise ValueError("Final episode identity differs")
    for key in ("producer", "contract", "engine"):
        if episode["manifest"][key] != identity[key]:
            raise ValueError(f"Episode {key} differs from run configuration")
    config = PreparationConfig(**identity["preparation"])
    if (config.surface_points, config.fps_points) != (4096, 1000):
        raise ValueError("Evaluation requires the 4096/1000 preparation sampling protocol")
    key = cache_key(
        sample_id,
        identity,
        expected,
        evaluation_protocol=PROTOCOL["version"],
        similarity=similarity,
    )
    cache_path = evaluation_cache_path(initial) if initial is not None else None
    cached = read_cache(cache_path, key) if cache_path is not None else None
    if cached is not None:
        values = cached_inputs(cached)
        if sha256(Path(initial).read_bytes()) != expected["sha256"]:
            raise ValueError("Initial episode differs from run configuration")
        ids, points, gt_poses, divisor, equivalence = (
            values[k] for k in ("part_ids", "points", "gt_poses", "divisor", "equivalence")
        )
        poses, state_index = final_poses_against_initial(episode, read_episode(initial), ids)
        sample = parts = None
    else:
        sample = prepare_sample(source, config)
        parts = sorted(sample.parts, key=lambda part: part.part_id)
        if len(parts) != expected["parts"]:
            raise ValueError("Source part count differs")
        poses, state_index = final_poses(episode, sample)
        divisor = evaluation_scale(sample)
        equivalence = resolve_equivalence(sample, config=similarity)
        ids = [p.part_id for p in parts]
        points = [p.points for p in parts]
        gt_poses = [p.gt_pose for p in parts]
        if cache_path is not None:
            write_cache(cache_path, cache_entry(sample, key, equivalence))
    prediction, target, seeds = [], [], []
    for cloud, gt_pose, (rotation, translation) in zip(points, gt_poses, poses):
        prediction.append(transform(cloud, rotation, translation) / divisor)
        target.append(apply_pose(cloud, gt_pose) / divisor)
        r = rotation_matrix(gt_pose.quaternion) @ rotation.T
        t = (gt_pose.position - r @ translation) / divisor
        seeds.append((r, t))
    groups = [[ids.index(pid) for pid in group] for group in equivalence["groups"]]
    ignored = equivalence["ignored_composite_self_groups"]
    return dict(
        equivalence=equivalence,
        sample=sample,
        parts=parts,
        part_ids=ids,
        points=points,
        gt_poses=gt_poses,
        cached=cached is not None,
        poses=poses,
        state_index=state_index,
        divisor=divisor,
        prediction=prediction,
        target=target,
        seeds=seeds,
        groups=groups,
        ignored=ignored,
        checksum=checksum,
        config=config,
    )


def evaluate_sample(directory, expected, identity, source, similarity=None, initial=None):
    started = time.monotonic()
    context = prepare_evaluation_inputs(
        directory, expected, identity, source, similarity=similarity, initial=initial
    )
    ids, prediction, target, seeds, groups = (
        context[key] for key in ("part_ids", "prediction", "target", "seeds", "groups")
    )
    checksum, state_index, divisor, ignored, config = (
        context[key] for key in ("checksum", "state_index", "divisor", "ignored", "config")
    )
    sample_id = source.sample_id
    path = directory / "final.episode.zip"
    rotation, translation, alignment = align(
        np.concatenate(prediction), np.concatenate(target), seeds
    )
    prediction = [transform(p, rotation, translation) for p in prediction]
    values, records = score_parts(prediction, target, groups, ids, alignment["chamfer"])
    if sha256(path.read_bytes()) != checksum:
        raise ValueError("Episode changed during evaluation")
    return dict(
        sample_id=sample_id,
        status="scored",
        **values,
        protocol_version=PROTOCOL["version"],
        dataset=source.dataset,
        revision=source.revision,
        episode_sha256=checksum,
        state_index=state_index,
        parts=records,
        equivalence_groups=[[ids[i] for i in group] for group in groups],
        ignored_composite_self_groups=ignored,
        scale_divisor=divisor,
        alignment=alignment,
        similarity=equivalence_metadata(context["equivalence"]),
        sampling_seed=config.sampling_seed,
        seconds=time.monotonic() - started,
        cached_inputs=context["cached"],
    )


def summarize(records, identity, similarity=None):
    scored = [row for row in records if row["status"] == "scored"]
    errors = [
        {"sample_id": row["sample_id"], "error": row["error"]}
        for row in records
        if row["status"] != "scored"
    ]
    return dict(
        status="complete" if not errors else "incomplete",
        expected_samples=len(records),
        scored_samples=len(scored),
        error_samples=len(errors),
        denominator=len(scored),
        errors=errors,
        **{
            key: float(np.mean([row[key] for row in scored])) if scored else None
            for key in ("SCD", "PA", "SR")
        },
        protocol={**PROTOCOL, "similarity": (similarity or SimilarityConfig()).protocol()},
        input_identity=identity,
    )


def error_row(sid, error, similarity=None):
    return dict(
        sample_id=sid,
        status="error",
        SCD=None,
        PA=None,
        SR=None,
        protocol_version=PROTOCOL["version"],
        error=f"{type(error).__name__}: {error}",
        similarity={"protocol": (similarity or SimilarityConfig()).protocol()},
    )


def score_source(directory, expected, identity, source, similarity=None, initial=None):
    try:
        return evaluate_sample(directory, expected, identity, source, similarity, initial)
    except Exception as exc:
        return error_row(source.sample_id, exc, similarity)


def read_run_config(run):
    """Read and validate the preparation provenance recorded with an experiment run."""
    run = Path(run).resolve()
    metadata_path = run / "run.json" if (run / "run.json").exists() else run / "meta.json"
    meta = json.loads(metadata_path.read_text())
    if not meta.get("config"):
        raise ValueError("This run has no dataset provenance for ground-truth evaluation")
    config = meta["config"]
    identity = config["identity"]
    if config["version"] != 1 or config["config_id"] != config_id(identity):
        raise ValueError("Invalid run preparation configuration")
    if identity["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported preparation protocol")
    if not re.fullmatch(r"[0-9a-f]{40}", identity["revision"]):
        raise ValueError("Evaluation requires a pinned HF commit revision")
    if not config["samples"]:
        raise ValueError("No configured samples")
    for sid in config["samples"]:
        sample_name(sid)
    return config, identity


def initial_locations(run, config):
    """Locate each configured sample's initial episode: the recorded path, else the data root.

    Samples whose initial episode cannot be found are simply absent; they are then
    evaluated from Hugging Face without a cache.
    """
    run = Path(run).resolve()
    metadata_path = run / "run.json" if (run / "run.json").exists() else run / "meta.json"
    options = json.loads(metadata_path.read_text()).get("options") or {}
    slug = config["identity"]["dataset"].split("/")[-1]
    found = {}
    for sid, expected in config["samples"].items():
        candidates = []
        input_path = run / "samples" / sample_name(sid) / "input.json"
        if input_path.is_file():
            recorded = json.loads(input_path.read_text()).get("initial_path")
            if recorded:
                candidates.append(Path(recorded))
        if options.get("data") and options.get("config_id") and expected.get("episode"):
            candidates.append(
                Path(options["data"]) / slug / options["config_id"] / expected["episode"]
            )
        for candidate in candidates:
            if candidate.is_file():
                found[sid] = candidate.resolve()
                break
    return found


def _validate_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")


def score_locations(
    locations,
    expected,
    identity,
    *,
    cache_dir=None,
    similarity=None,
    workers=4,
    initials=None,
):
    """Score every expected sample at its located directory; absent locations become errors.

    ``initials`` maps sample IDs to initial episodes inside prepared configuration
    directories. Samples with a matching evaluation cache entry are scored without
    loading source data; only the rest are read from Hugging Face.
    """
    similarity = similarity or SimilarityConfig()
    initials = initials or {}
    kwargs = dict(revision=identity["revision"], cache_dir=cache_dir, streaming=False)
    records, submitted = {}, set()
    workers = min(workers, os.cpu_count() or 1, len(expected))

    def record(row):
        sid = row["sample_id"]
        records[sid] = row
        print(
            f"[{len(records)}/{len(expected)}] {sid}: {row['status']} "
            f"{ {k: row[k] for k in ('SCD', 'PA', 'SR')} }",
            file=sys.stderr,
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def drain():
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                sid = pending.pop(future)
                try:
                    record(future.result())
                except Exception as exc:
                    record(error_row(sid, exc, similarity))

        def submit(source):
            sid = source.sample_id
            if sid in submitted:
                raise ValueError(f"Duplicate source sample: {sid}")
            if sid not in locations:
                submitted.add(sid)
                record(error_row(sid, FileNotFoundError("Missing final episode"), similarity))
                return
            future = pool.submit(
                score_source,
                locations[sid],
                expected[sid],
                identity,
                source,
                similarity,
                initials.get(sid),
            )
            pending[future] = sid
            submitted.add(sid)
            if len(pending) >= workers:
                drain()

        # Cached samples never touch source data; a cache entry whose key differs is an
        # error for that sample, not a silent recomputation.
        for sid in sorted(expected):
            if sid not in locations or sid not in initials:
                continue
            path = evaluation_cache_path(initials[sid])
            if path is None:
                continue
            key = cache_key(
                sid,
                identity,
                expected[sid],
                evaluation_protocol=PROTOCOL["version"],
                similarity=similarity,
            )
            try:
                hit = read_cache(path, key) is not None
            except ValueError as exc:
                submitted.add(sid)
                record(error_row(sid, exc, similarity))
                continue
            if hit:
                submit(SampleRef(sid, identity["dataset"], identity["revision"]))
        remaining = sorted(expected.keys() - submitted)
        # Bound in-flight source geometry to four samples. One HF scan normally;
        # isolate unresolved samples if source adaptation fails during that scan.
        try:
            if remaining:
                for source in load_samples(identity["dataset"], sample_ids=remaining, **kwargs):
                    submit(source)
        except Exception as exc:
            print(
                f"Source scan interrupted; checking remaining samples independently: {exc}",
                file=sys.stderr,
                flush=True,
            )
        for sid in sorted(expected.keys() - submitted):
            try:
                source = next(load_samples(identity["dataset"], sample_ids=[sid], **kwargs))
                submit(source)
            except Exception as exc:
                record(error_row(sid, exc, similarity))
        while pending:
            drain()
    return [records[sid] for sid in sorted(expected)]


def write_outputs(directory, rows, summary):
    """Atomically replace metrics.jsonl and metrics_summary.json in a directory."""
    directory = Path(directory)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, mode="w", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        os.replace(temporary, directory / "metrics.jsonl")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(directory / "metrics_summary.json", summary)


def evaluate_run(run, *, cache_dir=None, similarity=None, workers=4):
    """Score all configured samples, including partial outcomes, without changing inputs."""
    _validate_workers(workers)
    similarity = similarity or SimilarityConfig()
    run = Path(run).resolve()
    config, identity = read_run_config(run)
    expected = config["samples"]
    locations = {sid: run / "samples" / sample_name(sid) for sid in expected}
    rows = score_locations(
        locations,
        expected,
        identity,
        cache_dir=cache_dir,
        similarity=similarity,
        workers=workers,
        initials=initial_locations(run, config),
    )
    summary = summarize(rows, identity, similarity)
    write_outputs(run, rows, summary)
    return summary


def evaluate_runs(runs, output, *, cache_dir=None, similarity=None, workers=4, sample_ids=None):
    """Join resumed or split runs by sample identity into a new output directory.

    Archived run directories are never modified. Every expected sample is reported;
    a sample without a located final episode is recorded as an error row.
    """
    _validate_workers(workers)
    similarity = similarity or SimilarityConfig()
    runs = [Path(run).resolve() for run in runs]
    if not runs:
        raise ValueError("At least one run directory is required")
    locations, expected, identity, initials = {}, {}, None, {}
    for run in runs:
        config, run_identity = read_run_config(run)
        if identity is not None and run_identity != identity:
            raise ValueError("Runs have different preparation identities")
        identity = run_identity
        initials.update(initial_locations(run, config))
        for sid, item in config["samples"].items():
            if sid in expected and item != expected[sid]:
                raise ValueError(f"Conflicting sample identity: {sid}")
            expected[sid] = item
            directory = run / "samples" / sample_name(sid)
            if (directory / "final.episode.zip").is_file():
                if sid in locations:
                    raise ValueError(f"Multiple final episodes for {sid}")
                locations[sid] = directory
    if sample_ids is not None:
        selected = sorted(set(sample_ids))
        unknown = set(selected) - expected.keys()
        if unknown:
            raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")
        expected = {sid: expected[sid] for sid in selected}
        locations = {sid: locations[sid] for sid in selected if sid in locations}
        initials = {sid: initials[sid] for sid in selected if sid in initials}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "meta.json",
        dict(
            started_at=now(),
            runs=[str(run) for run in runs],
            identity=identity,
            protocol=PROTOCOL,
            similarity=similarity.protocol(),
            sample_ids=sorted(expected),
        ),
    )
    rows = score_locations(
        locations,
        expected,
        identity,
        cache_dir=cache_dir,
        similarity=similarity,
        workers=workers,
        initials=initials,
    )
    summary = summarize(rows, identity, similarity)
    write_outputs(output, rows, summary)
    return summary
