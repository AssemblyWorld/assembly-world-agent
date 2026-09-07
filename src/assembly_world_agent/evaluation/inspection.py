"""Read-only comparisons at the persisted, shared SCD-selected transform."""

import json
from pathlib import Path

import numpy as np

from ..artifacts import sample_name
from ..loading import load_samples
from ..similarity import SimilarityConfig
from .geometry import chamfer, score_parts, transform
from .runner import PROTOCOL, prepare_evaluation_inputs


def compare_run(run, *, cache_dir=None):
    """Compare source with saved geometry scoring, without writing any artifacts.

    Geometry groups are the recorded production result, not inferred from PA.
    Every episode is revalidated and scores are recomputed at its saved transform.
    """
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    identity, expected = meta["config"]["identity"], meta["config"]["samples"]
    saved = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if len(saved) != len(expected) or {r["sample_id"] for r in saved} != set(expected):
        raise ValueError("Saved metrics must contain every expected sample exactly once")
    saved = {r["sample_id"]: r for r in saved}
    results = []

    def inputs():
        resolved = set()
        try:
            for source in load_samples(
                identity["dataset"],
                revision=identity["revision"],
                sample_ids=sorted(expected),
                streaming=False,
                cache_dir=cache_dir,
            ):
                resolved.add(source.sample_id)
                yield source.sample_id, source
        except Exception:
            # Retry unresolved IDs separately so one malformed source cannot hide others.
            pass
        for sid in sorted(expected.keys() - resolved):
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
                yield sid, source
            except Exception as exc:
                yield sid, exc

    for sid, source in inputs():
        try:
            row = saved[sid]
            if row["status"] != "scored":
                raise ValueError(row.get("error", "Sample was not scored"))
            if (
                row["protocol_version"] != PROTOCOL["version"]
                or row["similarity"]["policy"] != "geometry"
            ):
                raise ValueError("Run comparison requires current geometry score files")
            if isinstance(source, Exception):
                raise source
            ctx = prepare_evaluation_inputs(
                run / "samples" / sample_name(sid),
                expected[sid],
                identity,
                source,
                similarity=SimilarityConfig("source"),
            )
            if row["episode_sha256"] != ctx["checksum"]:
                raise ValueError("Saved score episode hash differs")
            ids = [p.part_id for p in ctx["parts"]]
            alignment = row["alignment"]
            aligned = [
                transform(p, np.array(alignment["rotation"]), np.array(alignment["translation"]))
                for p in ctx["prediction"]
            ]
            shape_cd = chamfer(np.concatenate(aligned), np.concatenate(ctx["target"]))
            geometry_groups = [[ids.index(pid) for pid in g] for g in row["equivalence_groups"]]
            geometry, _ = score_parts(aligned, ctx["target"], geometry_groups, ids, shape_cd)
            if not np.allclose(
                [geometry[k] for k in ("SCD", "PA", "SR")],
                [row[k] for k in ("SCD", "PA", "SR")],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Saved geometry score differs from reconstructed score")
            source_values, _ = score_parts(aligned, ctx["target"], ctx["groups"], ids, shape_cd)
            diagnostics = row["similarity"]["group_diagnostics"]
            results.append(
                dict(
                    sample_id=sid,
                    status="scored",
                    SCD=geometry["SCD"],
                    geometry_PA=geometry["PA"],
                    geometry_SR=geometry["SR"],
                    source_PA=source_values["PA"],
                    source_SR=source_values["SR"],
                    delta_PA=geometry["PA"] - source_values["PA"],
                    delta_SR=geometry["SR"] - source_values["SR"],
                    source_groups=ctx["equivalence"]["groups"],
                    geometry_groups=row["equivalence_groups"],
                    above_threshold_pair_count=sum(
                        g["above_threshold_pair_count"] for g in diagnostics
                    ),
                    max_group_chamfer=max((g["max_chamfer"] for g in diagnostics), default=0),
                )
            )
        except Exception as exc:
            results.append(
                dict(sample_id=sid, status="error", error=f"{type(exc).__name__}: {exc}")
            )
    return sorted(results, key=lambda row: row["sample_id"])
