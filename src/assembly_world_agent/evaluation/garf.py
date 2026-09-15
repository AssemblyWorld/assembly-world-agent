"""GARF-style, anchor-aligned metrics; not an official benchmark reproduction."""

import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..artifacts import config_id, now, sample_name, write_json
from ..loading import load_samples
from ..similarity import SimilarityConfig
from ..utils import apply_pose, rotation_matrix, sample_surface, stable_rng
from ..utils.geometry import triangulate
from .geometry import chamfer, transform
from .runner import prepare_evaluation_inputs

PROTOCOL = {
    "version": "garf-style-anchor-v1",
    "reference_repository": "https://github.com/ai4ce/GARF",
    "reference_commit": "2d489c0b7b58de8af774aa1657c86c638e4744c2",
    "rotation": "intrinsic XYZ Euler degrees; wrapped component differences; per-part RMS",
    "translation": "RMS over three coordinates of sampled part centroids",
    "distance": "sum of bidirectional mean squared nearest-neighbor distances",
    "shape_cd": "global nearest neighbors, mean within each part, then equal part mean",
    "part_threshold": 0.01,
    "threshold_comparison": "<",
    "sampling": "5000 total area-weighted surface points; minimum 20 per part; no FPS",
    "scale": "max(1, largest source per-part axis-aligned extent), as public GARF loader",
    "alignment": "one shared SE(3) mapping largest sampled part pose to its GT pose; no ICP",
    "matching": "fixed source part IDs; anchor included in all part means",
    "aggregation": "sample macro mean; RMS is taken before averaging parts and samples",
    "units": {
        "RMSE_R": "degrees",
        "RMSE_T": "normalized units",
        "PA": "fraction",
        "CD": "squared normalized units",
    },
    "limitations": [
        "Paper describes 195 Fantastic Breaks objects; this HF revision contains 150.",
        "Official Fantastic Breaks HDF5 object manifest and upstream mesh scale are unverified.",
        "Deterministic local surface draws and input orientations differ from official random draws.",
        "Euler RMSE depends on input orientation; values are not directly interchangeable with Table 3.",
        "Free-moving agent outputs are anchor-aligned after inference; GARF normally fixes an anchor.",
        "Full colored meshes and iterative browser inference differ from GARF point-cloud inference.",
    ],
}
KEYS = ("RMSE_R", "RMSE_T", "PA", "CD")


def point_counts(areas, total=5000, minimum=20):
    areas = np.asarray(areas, dtype=float)
    if (
        not len(areas)
        or not np.isfinite(areas).all()
        or (areas <= 0).any()
        or total < minimum * len(areas)
    ):
        raise ValueError("Invalid surface areas or sampling budget")
    counts = minimum + ((total - minimum * len(areas)) * areas / areas.sum()).astype(int)
    counts[np.argmax(counts)] += total - counts.sum()
    return counts


def score_poses(points, predicted, target, anchor):
    """Score centered part frames after one shared reference-pose alignment."""
    ra, ta = predicted[anchor]
    rg, tg = target[anchor]
    rotation = rg @ ra.T
    translation = tg - rotation @ ta
    aligned = [(rotation @ r, rotation @ t + translation) for r, t in predicted]
    pred = [transform(p, r, t) for p, (r, t) in zip(points, aligned)]
    gt = [transform(p, r, t) for p, (r, t) in zip(points, target)]
    pred_tree, gt_tree = cKDTree(np.concatenate(pred)), cKDTree(np.concatenate(gt))
    rows = []
    for p, g, (rp, tp), (rt, tt) in zip(pred, gt, aligned, target):
        delta = np.abs(
            Rotation.from_matrix(rp).as_euler("XYZ", degrees=True)
            - Rotation.from_matrix(rt).as_euler("XYZ", degrees=True)
        )
        delta = np.minimum(delta, 360 - delta)
        cd = chamfer(p, g)
        rows.append(
            dict(
                RMSE_R=float(np.sqrt(np.mean(delta**2))),
                RMSE_T=float(np.sqrt(np.mean((tp - tt) ** 2))),
                PA=float(cd < 0.01),
                part_cd=cd,
                CD=float(np.mean(gt_tree.query(p)[0] ** 2) + np.mean(pred_tree.query(g)[0] ** 2)),
            )
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in KEYS}, rows


def evaluate_sample(directory, expected, identity, source, seed=42):
    started = time.monotonic()
    context = prepare_evaluation_inputs(
        directory, expected, identity, source, similarity=SimilarityConfig(policy="source")
    )
    sample, parts = context["sample"], context["parts"]
    areas = []
    for part in parts:
        xyz = part.mesh.vertices[triangulate(part.mesh)]
        areas.append(
            np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1).sum() / 2
        )
    counts = point_counts(areas)
    source_divisor = max(1.0, *(float(np.ptp(p.mesh.vertices, axis=0).max()) for p in source.parts))
    task_scale = np.linalg.norm(sample.source_to_world[:3, 0])
    divisor = source_divisor * task_scale
    points, prediction, target = [], [], []
    for part, count, (rp, tp) in zip(parts, counts, context["poses"]):
        cloud = sample_surface(
            part.mesh,
            int(count),
            stable_rng(seed, source.dataset, source.sample_id, part.part_id, "garf-evaluation"),
        )
        center = cloud.mean(0)
        points.append((cloud - center) / divisor)
        prediction.append((rp, (rp @ center + tp) / divisor))
        target.append(
            (rotation_matrix(part.gt_pose.quaternion), apply_pose(center, part.gt_pose) / divisor)
        )
    anchor = int(np.argmax(counts))
    values, rows = score_poses(points, prediction, target, anchor)
    return dict(
        sample_id=source.sample_id,
        status="scored",
        **values,
        parts=[
            dict(part_id=p.part_id, points=int(n), **row) for p, n, row in zip(parts, counts, rows)
        ],
        anchor_part_id=parts[anchor].part_id,
        source_scale_divisor=source_divisor,
        task_scale_divisor=divisor,
        sampling_seed=seed,
        episode_sha256=context["checksum"],
        state_index=context["state_index"],
        seconds=time.monotonic() - started,
    )


def score_location(directory, expected, identity, sid, seed, cache_dir):
    try:
        source = next(
            load_samples(
                identity["dataset"],
                revision=identity["revision"],
                sample_ids=[sid],
                streaming=False,
                cache_dir=cache_dir,
            )
        )
        return evaluate_sample(directory, expected, identity, source, seed)
    except Exception as exc:
        return dict(sample_id=sid, status="error", error=f"{type(exc).__name__}: {exc}")


def evaluate_runs(runs, output, *, cache_dir=None, seed=42, limit=None, workers=3):
    """Join resumed runs by sample identity, never silently omit unscored samples."""
    if workers < 1:
        raise ValueError("Workers must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    locations, expected, identity = {}, {}, None
    for run in map(Path, runs):
        meta = json.loads((run / "run.json").read_text())
        config = meta["config"]
        if config_id(config["identity"]) != config["config_id"]:
            raise ValueError("Invalid preparation configuration")
        if identity is not None and identity != config["identity"]:
            raise ValueError("Runs have different preparation identities")
        identity = config["identity"]
        for sid, item in config["samples"].items():
            if sid in expected and item != expected[sid]:
                raise ValueError(f"Conflicting sample identity: {sid}")
            expected[sid] = item
            directory = run / "samples" / sample_name(sid)
            if (directory / "final.episode.zip").is_file():
                if sid in locations:
                    raise ValueError(f"Multiple final episodes for {sid}")
                locations[sid] = directory
    if limit is not None:
        if limit <= 0:
            raise ValueError("Limit must be positive")
        expected = {sid: expected[sid] for sid in sorted(expected)[:limit]}
    write_json(
        output / "meta.json",
        dict(
            started_at=now(),
            runs=[str(Path(r).resolve()) for r in runs],
            identity=identity,
            protocol=PROTOCOL,
            seed=seed,
        ),
    )
    records = {}

    def record(row):
        records[row["sample_id"]] = row
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"[{len(records)}/{len(expected)}] {row['sample_id']}: {row['status']}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def drain():
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                sid = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(sample_id=sid, status="error", error=f"{type(exc).__name__}: {exc}")
                record(row)

        for sid in sorted(expected):
            if sid not in locations:
                record(dict(sample_id=sid, status="error", error="Missing final episode"))
                continue
            args = (locations[sid], expected[sid], identity, sid, seed, cache_dir)
            if workers == 1:
                record(score_location(*args))
            else:
                pending[pool.submit(score_location, *args)] = sid
                if len(pending) >= workers:
                    drain()
        while pending:
            drain()
    summary = summarize(list(records.values()), len(expected))
    write_json(output / "metrics_summary.json", summary)
    return summary


def summarize(records, expected_samples):
    """Aggregate verified sample records, keeping failures in the coverage denominator."""
    scored = [r for r in records if r["status"] == "scored"]
    summary = dict(
        status="complete" if len(scored) == expected_samples else "incomplete",
        expected_samples=expected_samples,
        scored_samples=len(scored),
        errors=[r for r in records if r["status"] != "scored"],
        protocol=PROTOCOL,
        finished_at=now(),
        **{key: float(np.mean([r[key] for r in scored])) if scored else None for key in KEYS},
    )
    return summary
