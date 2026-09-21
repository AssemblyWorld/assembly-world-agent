"""Optional CPU-only scoring transport; all experiment inputs originate locally.

Only scoring source code is staged remotely. Private evaluator arrays travel in
memory over SSH, never to a remote data file. Results return to the notebook cache.
The default notebook workflow remains local; this is an optional accelerator.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import threading
import types
import zlib
from pathlib import Path


def remote_imports(source):
    for name in [
        "assembly_world_agent",
        "assembly_world_agent.utils",
        "assembly_world_agent.evaluation",
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(source / name.replace(".", "/"))]
        sys.modules[name] = package
    from assembly_world_agent.evaluation.geometry import align, chamfer, score_parts, transform
    from assembly_world_agent.utils.transforms import rotation_matrix

    return align, chamfer, score_parts, transform, rotation_matrix


def score_job(payload):
    import numpy as np
    import scipy

    align, _, score_parts, transform, rotation_matrix = remote_imports(
        Path(os.environ["PAPER_SCORER_SOURCE"])
    )
    job = pickle.loads(zlib.decompress(base64.b64decode(payload)))
    values = job["values"]
    ids, clouds, gt, divisor, groups = [
        values[k] for k in ["ids", "points", "gt", "divisor", "groups"]
    ]
    target = [transform(p, rotation_matrix(q[3:]), q[:3]) / divisor for p, q in zip(clouds, gt)]
    out = []
    for key, poses in job["states"]:
        pred, seeds = [], []
        for p, q, g in zip(clouds, poses, gt):
            r = rotation_matrix(q[3:])
            t = q[:3]
            pred.append(transform(p, r, t) / divisor)
            sr = rotation_matrix(g[3:]) @ r.T
            seeds.append((sr, (g[:3] - sr @ t) / divisor))
        r, t, a = align(np.concatenate(pred), np.concatenate(target), seeds)
        pred = [transform(p, r, t) for p in pred]
        scores, parts = score_parts(pred, target, groups, ids, a["chamfer"])
        out.append(
            (
                key,
                dict(
                    **scores,
                    parts=parts,
                    alignment=dict(rotation=r.tolist(), translation=t.tolist(), seed=a["seed"]),
                    backend=dict(
                        numpy=np.__version__, scipy=scipy.__version__, platform=sys.platform
                    ),
                ),
            )
        )
    return dict(id=job["id"], scores=out)


def serve(args):
    os.environ["PAPER_SCORER_SOURCE"] = args.source
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_job, line.strip()) for line in sys.stdin if line.strip()]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {"error": repr(error)}
            print(json.dumps(result), flush=True)


def dispatch(args):
    import numpy as np
    import paper_analysis as pa
    import paper_geometry as pg

    pg.state_cache_protocol()
    # Stage exact scoring source, without the package's unrelated eager imports.
    stage = f"/tmp/assembly-paper-scorer-{os.getpid()}"
    members = [
        "models.py",
        "utils/transforms.py",
        "utils/registration.py",
        "evaluation/geometry.py",
    ]
    source = pa.ROOT / "src/assembly_world_agent"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        for name in members:
            tf.add(source / name, arcname="src/assembly_world_agent/" + name)
        tf.add(Path(__file__), arcname="worker.py")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", args.host]
    subprocess.run(
        ssh + [f"mkdir -p {stage} && tar -xzf - -C {stage}"], input=archive.getvalue(), check=True
    )
    command = f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {args.python} {stage}/worker.py --serve --source {stage}/src --workers {args.workers}"
    proc = subprocess.Popen(
        ssh + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    frame = pa.load_benchmark()

    def jobs():
        seen = set()
        count = 0
        for row in frame.to_dict("records"):
            values, _ = pg.evaluator_inputs(row)
            _, states, calls, _ = pg.recorded_episode(row)
            cps, _ = pg.sample_checkpoints(row, states, calls)
            geometry_hash = hashlib.sha256(
                b"".join(p.tobytes() for p in values["points"])
                + b"".join(np.r_[g.position, g.quaternion].tobytes() for g in values["gt_poses"])
                + json.dumps(values["equivalence"]["groups"]).encode()
                + str(values["divisor"]).encode()
            ).hexdigest()
            a = pa.digest(source / "utils/registration.py")
            b = pa.digest(source / "evaluation/geometry.py")
            final = states[row["metric"]["state_index"]].tobytes()
            pending = []
            for si in sorted({c[2] for c in cps}):
                q = states[si]
                pose = q.tobytes()
                if pose == final:
                    continue
                key = hashlib.sha256(
                    geometry_hash.encode() + pose + a.encode() + b.encode()
                ).hexdigest()
                if key in seen or (pa.CACHE / "state_scores" / f"{key}.json").exists():
                    continue
                seen.add(key)
                pending.append((key, q))
            if pending:
                ids = values["part_ids"]
                value = dict(
                    ids=ids,
                    points=values["points"],
                    gt=[np.r_[g.position, g.quaternion] for g in values["gt_poses"]],
                    divisor=values["divisor"],
                    groups=[[ids.index(i) for i in g] for g in values["equivalence"]["groups"]],
                )
                chunk_size = args.states_per_job or len(pending)
                for offset in range(0, len(pending), chunk_size):
                    job = dict(
                        id=f"{row['system']}/{row['block']}/{row['sample_id']}:{offset}",
                        values=value,
                        states=pending[offset : offset + chunk_size],
                    )
                    yield base64.b64encode(zlib.compress(pickle.dumps(job, protocol=5))).decode()
                    count += 1
                    if args.limit and count >= args.limit:
                        return

    feed_errors = []

    def feed():
        try:
            for job in jobs():
                proc.stdin.write(job + "\n")
            proc.stdin.close()
        except Exception as error:
            feed_errors.append(repr(error))
            proc.stdin.close()

    thread = threading.Thread(target=feed)
    thread.start()
    reports = []
    for line in proc.stdout:
        result = json.loads(line)
        if "error" in result:
            print(result, flush=True)
            reports.append(result)
            continue
        for key, value in result["scores"]:
            path = pa.CACHE / "state_scores" / f"{key}.json"
            if path.exists():
                old = pa.read_json(path)
                if not all(
                    np.isclose(old[m], value[m], atol=1e-7, rtol=1e-7) for m in ["PA", "SR", "SCD"]
                ):
                    raise ValueError("Local/remote scoring parity mismatch")
            else:
                tmp = path.with_suffix(".remote.tmp")
                pa.save_json(tmp, value)
                tmp.replace(path)
        reports.append(dict(id=result["id"], states=len(result["scores"])))
        pa.save_json(
            pa.CACHE / Path(args.report).name,
            dict(
                host=args.host,
                source_hashes={m: pa.digest(source / m) for m in members},
                reports=reports,
            ),
        )
        print(
            f"Returned {len(reports)} episode groups; {result['id']}: {len(result['scores'])} states",
            flush=True,
        )
    thread.join()
    code = proc.wait()
    # The stage contains only our copied source; remove exactly that directory.
    subprocess.run(ssh + [f"rm -rf -- {stage}"], check=True)
    if code or feed_errors:
        raise RuntimeError((code, feed_errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--host", help="User-controlled SSH host (required for dispatch)")
    parser.add_argument(
        "--python",
        help="Remote Python interpreter (required for dispatch)",
    )
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--states-per-job",
        type=int,
        default=0,
        help="Zero batches a whole episode; one balances expensive tail states",
    )
    parser.add_argument("--report", default="remote_scoring.json")
    args = parser.parse_args()
    if not args.serve and (not args.host or not args.python):
        parser.error("Dispatch requires explicit --host and --python")
    serve(args) if args.serve else dispatch(args)
