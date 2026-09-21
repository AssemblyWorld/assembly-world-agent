"""Recorded states, official scoring and geometry diagnostics for paper notebooks."""

import hashlib
import inspect
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import paper_analysis as pa

sys.path.insert(0, str(pa.ROOT / "src"))
from assembly_world_agent.evaluation.cache import (
    cache_key,
    cached_inputs,
    evaluation_cache_path,
    read_cache,
)
from assembly_world_agent.evaluation.geometry import align, chamfer, score_parts, transform
from assembly_world_agent.utils import apply_pose, rotation_matrix

EDIT_TOOLS = {"translate_objects", "rotate_objects", "set_object_pose"}


def state_cache_protocol():
    """Validate transitive scorer sources before reusing pose-level results."""
    source = pa.ROOT / "src/assembly_world_agent"
    expected = dict(
        protocol="assembly-evaluation-v2",
        implementation="official-multistart-recorded-state-v1",
        adapter_code_sha256=hashlib.sha256(
            (inspect.getsource(clouds_at) + inspect.getsource(score_state)).encode()
        ).hexdigest(),
        source_hashes={
            name: pa.digest(source / name)
            for name in [
                "models.py",
                "utils/transforms.py",
                "utils/registration.py",
                "evaluation/geometry.py",
            ]
        },
    )
    path = pa.CACHE / "state_score_protocol.json"
    if path.exists():
        if pa.read_json(path) != expected:
            raise ValueError("Scorer dependencies changed; use a fresh paper-analysis cache")
    elif any((pa.CACHE / "state_scores").glob("*.json")):
        raise ValueError("Unidentified state cache: protocol manifest is missing")
    else:
        pa.save_json(path, expected)
    return expected


def evaluator_inputs(row):
    """Resolve only the prepared-input cache named by the exported input record.

    These are immutable dataset-side evaluator inputs, not historical run outputs.
    Missing keys fail explicitly; there is no search of old experiment directories.
    """
    initial = pa.ROOT / row["input"]["initial_path"]
    path = evaluation_cache_path(initial)
    key = cache_key(
        row["sample_id"],
        row["identity"],
        row["expected"],
        evaluation_protocol="assembly-evaluation-v2",
    )
    value = read_cache(path, key)
    if value is None:
        raise FileNotFoundError(
            f"No matching evaluator inputs for {row['block']}/{row['sample_id']}"
        )
    return cached_inputs(value), pa.digest(path)


def recorded_episode(row):
    path = Path(row["sample_dir"]) / "final.episode.zip"
    checksum = pa.digest(path)
    expected = row["metric"].get("episode_sha256") or (row["result"].get("archive") or {}).get(
        "sha256"
    )
    if expected and checksum != expected:
        raise ValueError(f"Episode checksum mismatch: {path}")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        frames = [json.loads(s) for s in z.read("frames.jsonl").decode().splitlines() if s]
        raw = np.frombuffer(z.read("frames.bin"), dtype="<f8").reshape(
            len(frames), manifest["stateSize"]
        )
        n = len(manifest["objects"])
        states = {
            f["index"]: v[1 : 1 + 7 * n].reshape(n, 7).copy()
            for f, v in zip(frames, raw)
            if f["kind"] in ("initial", "state")
        }
        calls = [json.loads(s) for s in z.read("calls.jsonl").decode().splitlines() if s]
        events = [json.loads(s) for s in z.read("events.jsonl").decode().splitlines() if s]
        completions = {e["call_id"]: e["timestamp"] for e in events if e["kind"] == "complete"}
        for call in calls:
            if call["id"] not in completions:
                raise ValueError(f"Missing completion timestamp: {call['id']}")
            call["completed_at"] = completions[call["id"]]
    if manifest["id"] != row["input"]["episode_id"]:
        raise ValueError("Episode identity mismatch")
    return manifest, states, calls, checksum


def clouds_at(values, poses):
    pred, target, seeds = [], [], []
    for points, gt, q in zip(values["points"], values["gt_poses"], poses):
        r = rotation_matrix(q[3:])
        t = q[:3]
        pred.append(transform(points, r, t) / values["divisor"])
        target.append(apply_pose(points, gt) / values["divisor"])
        sr = rotation_matrix(gt.quaternion) @ r.T
        seeds.append((sr, (gt.position - sr @ t) / values["divisor"]))
    return pred, target, seeds


def score_state(values, poses):
    pred, target, seeds = clouds_at(values, poses)
    r, t, alignment = align(np.concatenate(pred), np.concatenate(target), seeds)
    pred = [transform(p, r, t) for p in pred]
    ids = values["part_ids"]
    groups = [[ids.index(i) for i in g] for g in values["equivalence"]["groups"]]
    scores, parts = score_parts(pred, target, groups, ids, alignment["chamfer"])
    return dict(
        **scores,
        parts=parts,
        alignment=dict(rotation=r.tolist(), translation=t.tolist(), seed=alignment["seed"]),
    )


def sample_checkpoints(row, states, calls):
    if not calls:
        return (
            [("fraction", k / 10, 0) for k in range(11)]
            + [("budget", float(b), 0) for b in pa.BUDGETS]
        ), np.array([])
    stamps = pd.to_datetime([c.get("completed_at", c["timestamp"]) for c in calls], utc=True)
    start = pd.Timestamp(row["result"]["started_at"])
    elapsed = (stamps - start).total_seconds().to_numpy()
    if np.any(np.diff(elapsed) < 0):
        raise ValueError("Non-monotone call timestamps")
    normalized = (elapsed - elapsed.min()) / max(elapsed.max() - elapsed.min(), 1e-9)

    def reached(mask):
        pos = np.flatnonzero(mask)
        return int(calls[pos[-1]]["state_index"]) if len(pos) else 0

    cps = [("fraction", 0.0, 0)]
    cps += [("fraction", k / 10, reached(normalized <= k / 10 + 1e-12)) for k in range(1, 11)]
    cps += [("budget", float(b), reached(elapsed <= b * 60)) for b in pa.BUDGETS]
    return cps, normalized


def audit_timing(frame):
    changed, delays, count = [], [], 0
    post_budget_changes = []
    for row in frame.to_dict("records"):
        _, states, calls, _ = recorded_episode(row)
        complete, _ = sample_checkpoints(row, states, calls)
        at60 = next(si for kind, t, si in complete if kind == "budget" and t == 60)
        terminal = row["metric"]["state_index"]
        if not np.array_equal(states[at60], states[terminal]):
            post_budget_changes.append(
                dict(
                    system=row["system"],
                    block=row["block"],
                    sample_id=row["sample_id"],
                    at60=at60,
                    final=terminal,
                )
            )
        requested, _ = sample_checkpoints(
            row, states, [{k: v for k, v in call.items() if k != "completed_at"} for call in calls]
        )
        count += len(calls)
        delays.extend(
            (pd.Timestamp(c["completed_at"]) - pd.Timestamp(c["timestamp"])).total_seconds()
            for c in calls
        )
        for old, new in zip(requested, complete):
            if old != new:
                changed.append(
                    dict(
                        system=row["system"],
                        block=row["block"],
                        sample_id=row["sample_id"],
                        before=old,
                        after=new,
                    )
                )
    report = dict(
        episodes=len(frame),
        calls_with_completion_event=count,
        max_request_completion_seconds=max(delays, default=0),
        calls_over_one_second=sum(d > 1 for d in delays),
        changed=changed,
        timestamp_basis="environment completion events",
        post_60_minute_pose_changes=post_budget_changes,
        maximum_recorded_minutes=float(frame.duration.max() / 60),
    )
    pa.save_json(pa.CACHE / "timestamp_audit.json", report)
    return report


def analysis_key(row, episode_sha, input_sha, checkpoints):
    key = dict(
        version=pa.VERSION,
        episode=episode_sha,
        evaluator_inputs=input_sha,
        scoring_code=pa.digest(pa.ROOT / "src/assembly_world_agent/utils/registration.py"),
        geometry_code=pa.digest(pa.ROOT / "src/assembly_world_agent/evaluation/geometry.py"),
        timestamp_basis="environment complete events",
        analysis_implementation=hashlib.sha256(
            (inspect.getsource(analyze_episode) + inspect.getsource(sample_checkpoints)).encode()
        ).hexdigest(),
        diagnostic_parameters=dict(
            centroid=0.1,
            centered_cd=0.01,
            far_displacement=3.0,
            translation_epsilon=1e-9,
            rotation_epsilon_degrees=1e-5,
        ),
        exported_records=hashlib.sha256(
            json.dumps(
                dict(input=row["input"], result=row["result"], metric=row["metric"]), sort_keys=True
            ).encode()
        ).hexdigest(),
        checkpoints=checkpoints,
    )
    return json.loads(json.dumps(key))


def analyze_episode(row, recompute=False, score=True):
    started = time.monotonic()
    manifest, states, calls, sha = recorded_episode(row)
    values, input_sha = evaluator_inputs(row)
    n = len(values["part_ids"])
    if len(manifest["objects"]) != n:
        raise ValueError("Part count mismatch")
    # Episode identities fix sorted source part order; verify names when available.
    for obj, pid in zip(manifest["objects"], values["part_ids"]):
        if obj.get("name", "").startswith("Part ") and obj["name"][5:] != pid:
            raise ValueError("Part ordering mismatch")
    cps, fractions = sample_checkpoints(row, states, calls)
    key = analysis_key(row, sha, input_sha, cps)
    path = (
        pa.CACHE
        / "trajectories"
        / row["system"]
        / row["block"]
        / f"{row['sample_id'].replace('/', '--')}.json"
    )
    key = json.loads(json.dumps(key))
    if path.exists() and not recompute:
        old = pa.read_json(path)
        if old["key"] == key and (old.get("scores") or not score):
            return old
    scored = {}
    if score:
        state_cache_protocol()
        # Deduplicate identical body poses (camera changes do not alter geometry).
        by_pose = {}
        # The stored final transform is an official result, revalidated here with
        # the same clouds, full shape CD and constrained Hungarian assignment.
        m = row["metric"]
        if m.get("alignment"):
            final_q = states[m["state_index"]]
            pred, targ, _ = clouds_at(values, final_q)
            r = np.asarray(m["alignment"]["rotation"])
            t = np.asarray(m["alignment"]["translation"])
            pred = [transform(p, r, t) for p in pred]
            ids = values["part_ids"]
            groups = [[ids.index(i) for i in g] for g in values["equivalence"]["groups"]]
            v, parts = score_parts(
                pred, targ, groups, ids, chamfer(np.concatenate(pred), np.concatenate(targ))
            )
            by_pose[final_q.tobytes()] = dict(
                **v,
                parts=parts,
                alignment=dict(rotation=r.tolist(), translation=t.tolist(), seed="recorded-final"),
            )
        geometry_hash = hashlib.sha256(
            b"".join(p.tobytes() for p in values["points"])
            + b"".join(np.r_[g.position, g.quaternion].tobytes() for g in values["gt_poses"])
            + json.dumps(values["equivalence"]["groups"]).encode()
            + str(values["divisor"]).encode()
        ).hexdigest()
        for si in sorted({c[2] for c in cps}):
            pose_key = states[si].tobytes()
            if pose_key not in by_pose:
                state_key = hashlib.sha256(
                    geometry_hash.encode()
                    + pose_key
                    + key["scoring_code"].encode()
                    + key["geometry_code"].encode()
                ).hexdigest()
                state_path = pa.CACHE / "state_scores" / f"{state_key}.json"
                if state_path.exists() and not recompute:
                    by_pose[pose_key] = pa.read_json(state_path)
                else:
                    by_pose[pose_key] = score_state(values, states[si])
                    tmp = state_path.with_suffix(f".{os.getpid()}.tmp")
                    pa.save_json(tmp, by_pose[pose_key])
                    tmp.replace(state_path)
            scored[str(si)] = by_pose[pose_key]
        final = scored[str(cps[10][2])]
        if row["scored"]:
            for metric in ("PA", "SR", "SCD"):
                if not np.isclose(final[metric], row[metric], atol=1e-7, rtol=1e-7):
                    raise ValueError(f"Final {metric} mismatch: {final[metric]} vs {row[metric]}")
    centers = [p.mean(axis=0) for p in values["points"]]
    action_rows, moved_parts = [], set()
    for c, f in zip(calls, fractions):
        a = dict(index=c["index"], name=c["name"], fraction=float(f), state=c["state_index"])
        if c["name"] in EDIT_TOOLS:
            q0, q1 = states[c["before_index"]], states[c["state_index"]]
            delta, angles = [], []
            for i, center in enumerate(centers):
                r0, r1 = rotation_matrix(q0[i, 3:]), rotation_matrix(q1[i, 3:])
                delta.append(
                    (r1 @ center + q1[i, :3] - r0 @ center - q0[i, :3]) / values["divisor"]
                )
                angles.append(np.degrees(np.arccos(np.clip((np.trace(r1 @ r0.T) - 1) / 2, -1, 1))))
            delta = np.asarray(delta)
            dist = np.linalg.norm(delta, axis=1)
            angles = np.asarray(angles)
            moved = (dist > 1e-9) | (angles > 1e-5)
            moved_parts.update(np.flatnonzero(moved).tolist())
            i = int(np.argmax(dist + angles * 0.001))
            a.update(
                displacement=float(dist.max()),
                angle=float(angles.max()),
                primary=i,
                vector=delta[i].tolist(),
                vectors={str(j): delta[j].tolist() for j in np.flatnonzero(moved)},
                moved=np.flatnonzero(moved).tolist(),
            )
        if c["name"] == "capture_scene":
            cam = (c.get("result") or {}).get("camera")
            if cam:
                a["camera"] = cam
        action_rows.append(a)
    # Use the authoritative final correspondence; never silently change matching.
    metric = row["metric"]
    diagnostics = []
    if metric.get("parts") and metric.get("alignment"):
        q = states[metric["state_index"]]
        pred, target, _ = clouds_at(values, q)
        r = np.asarray(metric["alignment"]["rotation"])
        t = np.asarray(metric["alignment"]["translation"])
        pred = [transform(p, r, t) for p in pred]
        ids = values["part_ids"]
        for p in metric["parts"]:
            i, j = ids.index(p["part_id"]), ids.index(p["target_part_id"])
            pc, tc = pred[i].mean(0), target[j].mean(0)
            distance = float(np.linalg.norm(pc - tc))
            centered = chamfer(pred[i] - pc, target[j] - tc)
            actual = chamfer(pred[i], target[j])
            if not np.isclose(actual, p["chamfer"], atol=1e-7, rtol=1e-7):
                raise ValueError("Saved correspondence does not reproduce part error")
            if p["correct"]:
                cat = "Correct"
            elif i not in moved_parts:
                cat = "Never moved"
            elif distance > 3:
                cat = "Far displaced"
            elif centered <= 0.01 and distance > 0.1:
                cat = "Translation dominated"
            elif distance <= 0.1 and centered <= 0.01:
                cat = "Near threshold"
            elif distance <= 0.1:
                cat = "Orientation dominated"
            else:
                cat = "Mixed"
            diagnostics.append(
                dict(
                    part_id=p["part_id"],
                    target_part_id=p["target_part_id"],
                    category=cat,
                    centroid_distance=distance,
                    centered_CD=float(centered),
                    chamfer=p["chamfer"],
                )
            )
    result = dict(
        key=key,
        system=row["system"],
        block=row["block"],
        sample_id=row["sample_id"],
        scored=row["scored"],
        scores=scored,
        actions=action_rows,
        diagnostics=diagnostics,
        runtime_seconds=time.monotonic() - started,
    )
    pa.save_json(path, result)
    return result


def audit_regressions(frame, analyses):
    """Rescore decreasing transitions with the preceding registration frozen.

    This is a sensitivity diagnostic, not a causal attribution of a tool call:
    multiple edits may fall between deciles. Persist each episode's result so
    rerendering does not repeat the nearest-neighbor and assignment calculations.
    """
    rows = {(r["system"], r["block"], r["sample_id"]): r for r in frame.to_dict("records")}
    output = []
    for d in analyses:
        if not d["scores"]:
            continue
        key = hashlib.sha256(
            (json.dumps(d["key"], sort_keys=True) + inspect.getsource(audit_regressions)).encode()
        ).hexdigest()
        path = pa.CACHE / "regression_audits" / f"{key}.json"
        if path.exists():
            output.extend(pa.read_json(path))
            continue
        cps = [(t, si) for kind, t, si in d["key"]["checkpoints"] if kind == "fraction"]
        decreases = []
        for (t0, i0), (t1, i1) in zip(cps, cps[1:]):
            before, after = d["scores"][str(i0)], d["scores"][str(i1)]
            if after["PA"] < before["PA"] - 1e-9:
                decreases.append((t0, t1, i0, i1, before, after))
        episode_rows = []
        if decreases:
            row = rows[d["system"], d["block"], d["sample_id"]]
            values, _ = evaluator_inputs(row)
            _, states, _, _ = recorded_episode(row)
            ids = values["part_ids"]
            groups = [[ids.index(i) for i in g] for g in values["equivalence"]["groups"]]
            for t0, t1, i0, i1, before, after in decreases:
                pred, target, _ = clouds_at(values, states[i1])
                r0 = np.asarray(before["alignment"]["rotation"])
                tprev = np.asarray(before["alignment"]["translation"])
                aligned = [transform(p, r0, tprev) for p in pred]
                frozen_scores, _ = score_parts(aligned, target, groups, ids, 0.0)
                previous_map = {p["part_id"]: p["target_part_id"] for p in before["parts"]}
                next_map = {p["part_id"]: p["target_part_id"] for p in after["parts"]}
                fixed_pa = float(
                    np.mean(
                        [
                            chamfer(aligned[i], target[ids.index(previous_map[pid])]) <= 0.01
                            for i, pid in enumerate(ids)
                        ]
                    )
                )
                r1 = np.asarray(after["alignment"]["rotation"])
                angle = float(np.degrees(np.arccos(np.clip((np.trace(r1 @ r0.T) - 1) / 2, -1, 1))))
                episode_rows.append(
                    dict(
                        system=d["system"],
                        block=d["block"],
                        sample_id=d["sample_id"],
                        fraction_before=t0,
                        fraction_after=t1,
                        state_before=i0,
                        state_after=i1,
                        PA_before=before["PA"],
                        PA_after=after["PA"],
                        PA_frozen_alignment=frozen_scores["PA"],
                        PA_frozen_alignment_and_matching=fixed_pa,
                        matching_changes=sum(previous_map[i] != next_map[i] for i in ids),
                        alignment_rotation_degrees=angle,
                        alignment_translation_change=float(
                            np.linalg.norm(np.asarray(after["alignment"]["translation"]) - tprev)
                        ),
                        drop_persists_frozen_alignment=bool(
                            frozen_scores["PA"] < before["PA"] - 1e-9
                        ),
                        drop_persists_frozen_alignment_and_matching=bool(
                            fixed_pa < before["PA"] - 1e-9
                        ),
                    )
                )
        pa.save_json(path, episode_rows)
        output.extend(episode_rows)
    pa.save_json(pa.CACHE / "regression_audit.json", output)
    pd.DataFrame(output).to_csv(pa.CACHE / "regression_audit.csv", index=False)
    return pd.DataFrame(output)


def worker(job):
    row, recompute, score = job
    try:
        d = analyze_episode(row, recompute, score)
        return dict(
            system=row["system"],
            block=row["block"],
            sample_id=row["sample_id"],
            ok=True,
            states=len(d["scores"]),
            seconds=d["runtime_seconds"],
        )
    except Exception as error:
        return dict(
            system=row["system"],
            block=row["block"],
            sample_id=row["sample_id"],
            ok=False,
            error=f"{type(error).__name__}: {error}",
        )


def initialize_worker(results, paper, cache):
    pa.configure(results, paper, cache)


def compute_trajectories(frame, workers=4, recompute=False, score=True):
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(row, recompute, score) for row in frame[frame.has_episode].to_dict("records")]
    reports = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(str(pa.RESULTS), str(pa.PAPER), str(pa.CACHE)),
    ) as pool:
        for i, result in enumerate(pool.map(worker, jobs), 1):
            reports.append(result)
            if i % 10 == 0 or not result["ok"]:
                print(f"{i}/{len(jobs)} episodes; last={result}", flush=True)
            pa.save_json(pa.CACHE / "trajectory_validation.json", reports)
    return pd.DataFrame(reports)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()
    frame = pa.load_benchmark()
    if args.limit:
        frame = frame.head(args.limit)
    report = compute_trajectories(frame, args.workers, score=not args.no_score)
    print(report.groupby(["system", "ok"]).size().to_string())
