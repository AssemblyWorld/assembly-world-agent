"""Read-only experiment inputs with atomically replaced metric outputs."""

import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from ..artifacts import config_id, sample_name, write_json
from ..episode_io import read_episode
from ..episodes import sha256
from ..loading import load_samples
from ..models import PROTOCOL_VERSION, PreparationConfig
from ..preparation import prepare_sample
from ..similarity import SimilarityConfig, equivalence_metadata, resolve_equivalence
from ..utils import apply_pose, rotation_matrix
from .geometry import (
    IMPROVEMENT_TOLERANCE,
    MAX_ITERATIONS,
    THRESHOLD,
    align,
    score_parts,
    transform,
)
from .inputs import evaluation_scale, final_poses

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


def prepare_evaluation_inputs(directory, expected, identity, source, *, similarity=None):
    """Validate and reconstruct the shared inputs used by scoring and inspection."""
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
    sample = prepare_sample(source, config)
    parts = sorted(sample.parts, key=lambda part: part.part_id)
    if len(parts) != expected["parts"]:
        raise ValueError("Source part count differs")
    poses, state_index = final_poses(episode, sample)
    divisor = evaluation_scale(sample)
    prediction, target, seeds = [], [], []
    for part, (rotation, translation) in zip(parts, poses):
        prediction.append(transform(part.points, rotation, translation) / divisor)
        target.append(apply_pose(part.points, part.gt_pose) / divisor)
        r = rotation_matrix(part.gt_pose.quaternion) @ rotation.T
        t = (part.gt_pose.position - r @ translation) / divisor
        seeds.append((r, t))
    equivalence = resolve_equivalence(sample, config=similarity)
    ids = [p.part_id for p in parts]
    groups = [[ids.index(pid) for pid in group] for group in equivalence["groups"]]
    ignored = equivalence["ignored_composite_self_groups"]
    return dict(
        equivalence=equivalence,
        sample=sample,
        parts=parts,
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


def evaluate_sample(directory, expected, identity, source, similarity=None):
    started = time.monotonic()
    context = prepare_evaluation_inputs(
        directory, expected, identity, source, similarity=similarity
    )
    parts, prediction, target, seeds, groups = (
        context[key] for key in ("parts", "prediction", "target", "seeds", "groups")
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
    values, records = score_parts(
        prediction, target, groups, [p.part_id for p in parts], alignment["chamfer"]
    )
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
        equivalence_groups=[[parts[i].part_id for i in group] for group in groups],
        ignored_composite_self_groups=ignored,
        scale_divisor=divisor,
        alignment=alignment,
        similarity=equivalence_metadata(context["equivalence"]),
        sampling_seed=config.sampling_seed,
        seconds=time.monotonic() - started,
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


def score_source(directory, expected, identity, source, similarity=None):
    try:
        return evaluate_sample(directory, expected, identity, source, similarity)
    except Exception as exc:
        return error_row(source.sample_id, exc, similarity)


def evaluate_run(run, *, cache_dir=None, similarity=None):
    """Score all configured samples, including partial outcomes, without changing inputs."""
    similarity = similarity or SimilarityConfig()
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
    expected = config["samples"]
    if not expected:
        raise ValueError("No configured samples")
    for sid in expected:
        sample_name(sid)
    kwargs = dict(revision=identity["revision"], cache_dir=cache_dir, streaming=False)
    records, submitted = {}, set()
    workers = min(4, os.cpu_count() or 1, len(expected))

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
            future = pool.submit(
                score_source,
                run / "samples" / sample_name(sid),
                expected[sid],
                identity,
                source,
                similarity,
            )
            pending[future] = sid
            submitted.add(sid)
            if len(pending) >= workers:
                drain()

        # Bound in-flight source geometry to four samples. One HF scan normally;
        # isolate unresolved samples if source adaptation fails during that scan.
        try:
            for source in load_samples(identity["dataset"], sample_ids=list(expected), **kwargs):
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
    rows = [records[sid] for sid in sorted(expected)]
    summary = summarize(rows, identity, similarity)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=run, mode="w", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        os.replace(temporary, run / "metrics.jsonl")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(run / "metrics_summary.json", summary)
    return summary
