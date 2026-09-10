"""Self-contained result snapshots from pinned tasks and saved runtime states."""

from __future__ import annotations

import base64
import gzip
import html
import io
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..adapters import get_adapter, ikea_manual
from ..episode_io import read_episode
from ..episodes import _render_mesh
from ..loading import load_samples
from ..models import Mesh, Pose, PreparationConfig, SourcePart, SourceSample
from ..preparation import prepare_sample
from ..utils.geometry import triangulate

WEB = Path(__file__).parent / "web"
ENCODING_WORKERS = min(16, os.cpu_count() or 1)


def load_result_samples(dataset, *, revision, sample_ids, cache_dir=None):
    """Read only IKEA geometry from cached Arrow, retaining other dataset adapters."""
    adapter = get_adapter(dataset)
    if adapter is not ikea_manual or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        yield from load_samples(
            dataset, revision=revision, sample_ids=sample_ids, cache_dir=cache_dir, streaming=False
        )
        return
    from datasets import load_dataset

    rows = (
        load_dataset(
            adapter.REPO_ID,
            split="full",
            revision=revision,
            streaming=False,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
        .select_columns(["object_id", "parts_ct", "parts"])
        .with_format("arrow")
    )
    wanted, found = set(sample_ids), set()
    for table in rows:
        sid = table["object_id"][0].as_py()
        if sid not in wanted:
            continue
        if sid in found:
            raise ValueError(f"Duplicate sample ID: {sid}")
        parts = []
        for record in table["parts"][0].values:
            vertices = np.asarray(record["vertices"].as_py(), dtype=np.float64)
            faces = tuple(tuple(f) for f in record["faces"].as_py())
            parts.append(
                SourcePart(
                    part_id=record["part_id"].as_py(),
                    mesh=Mesh(vertices, faces),
                    assembled_pose=Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
                    metadata={},
                )
            )
        if not parts or len(parts) != table["parts_ct"][0].as_py():
            raise ValueError(f"Source part count differs: {sid}")
        found.add(sid)
        yield SourceSample(
            dataset=adapter.REPO_ID,
            sample_id=sid,
            revision=revision,
            parts=tuple(parts),
            source_to_z_up=adapter.SOURCE_TO_Z_UP.copy(),
            metadata={},
            source_splits=[],
            manual_pages=(),
            manual=[],
            steps=(),
            annotations={},
        )
        if found == wanted:
            return
    if wanted - found:
        raise ValueError(f"Requested sample IDs not found: {sorted(wanted - found)}")


def render_geometry(part):
    """Build the episode-compatible arrays without OBJ text or repeated input validation."""
    vertices, triangles = _render_mesh(part, validate=False)
    return {"vertices": vertices.tolist(), "triangles": triangles.ravel().tolist()}


def parallel_map(function, values):
    """Keep ordered results and a bounded number of pending native encoding jobs."""
    values = iter(values)
    workers = ENCODING_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(pool.submit(function, next(values)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(function, next(values)))
            except StopIteration:
                pass


def packed(value):
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode()


def select_episode(directory, expected_id=None):
    """Prefer final; otherwise newest valid checkpoint. Report rejected candidates."""
    candidates = [directory / "final.episode.zip"]
    candidates += sorted(
        (directory / "checkpoints").glob("*.episode.zip"),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
        reverse=True,
    )
    errors = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            episode = read_episode(path, verify_hashes=False)
            if expected_id is not None and episode["manifest"]["id"] != expected_id:
                raise ValueError("Episode identity differs from configured sample")
            return episode, {"path": str(path.relative_to(directory))}, errors
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    return None, None, errors


def obj_mesh(payload):
    vertices, faces = [], []
    for line in payload.decode().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields and fields[0] == "f":
            indices = [int(v.split("/")[0]) for v in fields[1:]]
            faces.append(tuple(i - 1 if i > 0 else len(vertices) + i for i in indices))
    mesh = Mesh(np.asarray(vertices, dtype=float), tuple(faces))
    return {"vertices": mesh.vertices.tolist(), "triangles": triangulate(mesh).ravel().tolist()}


def episode_data(ep):
    """Use body poses, not compiled mesh offsets, on the original baked OBJ vertices."""
    import mujoco as mj

    files = ep["files"]
    from ..episode_model import compiled_parts, load_model

    model = load_model(ep)
    if ep["manifest"].get("model"):
        parts = compiled_parts(model, ep["manifest"]["objects"])
    else:
        root = ET.fromstring(files["world/model.xml"])
        assets = {e.attrib["name"]: e.attrib["file"] for e in root.findall("./asset/mesh")}
        parts = []
        for body in root.findall("./worldbody/body"):
            geoms = body.findall("geom")
            if len(geoms) != 1 or body.findall("body"):
                raise ValueError("Result viewer requires one mesh per independent assembly body")
            geom = geoms[0]
            if any(k in geom.attrib for k in ("pos", "quat", "euler", "axisangle")):
                raise ValueError("Unsupported local geom transform")
            mesh_name = geom.attrib["mesh"]
            resource = "world/" + assets[mesh_name]
            parts.append(
                {
                    "id": body.attrib["name"],
                    "resource": resource,
                    "geometry": obj_mesh(files[resource]),
                }
            )
    data = mj.MjData(model)
    body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, p["id"]) for p in parts]
    frames = {}
    for index, frame in ep["states"].items():
        mj.mj_setState(model, data, np.array(frame["integration"]), ep["manifest"]["stateSpec"])
        mj.mj_forward(model, data)
        frames[str(index)] = {
            "poses": [np.r_[data.xpos[b], data.xquat[b]].tolist() for b in body_ids],
            "camera": frame["camera"],
            "groups": frame["groups"],
        }
    calls = []
    for call in ep["calls"]:
        before, after = frames[str(call["before_index"])], frames[str(call["state_index"])]
        changed = (
            not np.allclose(before["poses"], after["poses"], atol=1e-10, rtol=0)
            or before["camera"] != after["camera"]
            or before["groups"] != after["groups"]
        )
        calls.append(
            {
                k: call[k]
                for k in (
                    "index",
                    "name",
                    "arguments",
                    "status",
                    "result",
                    "error",
                    "state_index",
                    "timestamp",
                )
                if k in call
            }
            | {"changed": changed}
        )
    return {"parts": parts, "frames": frames, "calls": calls}


def encode_manual_image(raw):
    """Encode lossless WebP pixels without resizing or a redundant decode pass."""
    with Image.open(io.BytesIO(raw)) as im:
        pixels = im.convert("RGBA")
        output = io.BytesIO()
        metadata = {key: im.info[key] for key in ("icc_profile", "exif") if key in im.info}
        pixels.save(output, format="WEBP", lossless=True, exact=True, method=6, **metadata)
        raw = output.getvalue()
    return {
        "data": "data:image/webp;base64," + base64.b64encode(raw).decode(),
        "image_format": "webp-lossless",
    }


def manual_pages(directory, errors):
    pages = []

    def encode_page(page):
        try:
            path = directory / "manualbook" / page["file"]
            if path.resolve().parent != (directory / "manualbook").resolve():
                raise ValueError("Unsafe manual path")
            raw = path.read_bytes()
            return page | encode_manual_image(raw)
        except Exception as error:
            return page | {"error": str(error)}

    try:
        manifest = json.loads((directory / "manualbook/pages.json").read_text())
        for page in parallel_map(encode_page, manifest["pages"]):
            pages.append(page)
            if "error" in page:
                errors.append(f"Manual {page['file']}: {page['error']}")
    except Exception as error:
        errors.append(f"Manual: {error}")
    return pages


def prompt_manual_pages(directory, errors):
    """Display only images actually supplied in the recorded initial user input."""
    try:
        inputs = json.loads((directory / "input.json").read_text())
        records = inputs["manual"]["pages"]
        with (directory / "conversation.jsonl").open() as stream:
            first = next(
                json.loads(line)
                for line in stream
                if line.strip() and json.loads(line).get("role") == "user"
            )
        content = first.get("content", [])
        images = (
            [b for b in content if b.get("type") == "image"] if isinstance(content, list) else []
        )
        if len(images) != len(records):
            raise ValueError("Initial image count differs from recorded input provenance")
        return [
            {
                **record,
                "file": record.get("source_file", f"page-{index + 1}.png"),
                **encode_manual_image(base64.b64decode(block["source"]["data"], validate=True)),
            }
            for index, (record, block) in enumerate(zip(records, images))
        ]
    except Exception as error:
        errors.append(f"Manual: {error}")
        return []


def export_results(run, output, *, cache_dir=None):
    """Build an atomic HTML snapshot; preserve the run and isolate sample failures."""
    run, output = Path(run).resolve(), Path(output).resolve()
    current = (run / "run.json").exists()
    meta = json.loads((run / ("run.json" if current else "meta.json")).read_text())
    config = meta["config"]
    identity = config["identity"]
    progress = (
        json.loads((run / "progress.json").read_text()) if (run / "progress.json").exists() else {}
    )
    rows = []
    # Load the pinned dataset once through the standard HF dataset cache.
    sources, source_error = {}, None
    print("Loading pinned GT source samples…", flush=True)
    try:
        for source in load_result_samples(
            identity["dataset"],
            revision=identity["revision"],
            sample_ids=list(config["samples"]),
            cache_dir=cache_dir,
        ):
            sources[source.sample_id] = source
            print(
                f"Loaded source {len(sources)}/{len(config['samples'])}: {source.sample_id}",
                flush=True,
            )
    except Exception as error:
        source_error = str(error)
    for sid in config["samples"]:
        print(f"Exporting {sid}", flush=True)
        directory = run / "samples" / sid.replace("/", "--")
        errors = []
        row = {
            "id": sid,
            "status": progress.get(sid, {}).get("status", "unknown"),
            "errors": errors,
            "manual": (
                prompt_manual_pages(directory, errors)
                if current and meta.get("options", {}).get("reference_mode") is not None
                else manual_pages(directory, errors)
            ),
        }
        if current and (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            row["status"] = f"execution: {result['status']}"
            if result.get("agent_outcome"):
                row["status"] += f" · agent: {result['agent_outcome']['status']}"
        ep, provenance, rejected = select_episode(
            directory, config["samples"][sid].get("episode_id")
        )
        errors.extend(rejected)
        row["episode_source"] = provenance
        if ep is not None:
            try:
                expected = config["samples"][sid]["episode_id"]
                if ep["manifest"]["id"] != expected:
                    raise ValueError("Episode identity differs from configured sample")
                row["replay"] = episode_data(ep)
            except Exception as error:
                errors.append(f"Replay: {error}")
        try:
            if sid not in sources:
                raise ValueError(source_error or "Pinned source sample unavailable")
            prepared = prepare_sample(
                sources.pop(sid), PreparationConfig(**identity["preparation"]), sample_points=False
            )
            parts = sorted(prepared.parts, key=lambda p: p.part_id)
            if "replay" in row and len(parts) != len(row["replay"]["parts"]):
                raise ValueError("GT part count differs")
            geometry = []
            for i, part in enumerate(parts):
                name = f"part-{i + 1:04d}"
                rebuilt = render_geometry(part)
                if "replay" in row:
                    actual = row["replay"]["parts"][i]
                    recorded = actual["geometry"]
                    if (
                        actual["id"] != name
                        or recorded["triangles"] != rebuilt["triangles"]
                        or np.shape(recorded["vertices"]) != np.shape(rebuilt["vertices"])
                        or not np.allclose(
                            recorded["vertices"], rebuilt["vertices"], atol=2e-7, rtol=0
                        )
                    ):
                        raise ValueError("Rebuilt GT geometry differs from recorded geometry")
                else:
                    geometry.append({"id": name, "geometry": rebuilt})
            row["gt"] = {
                "poses": [np.r_[p.gt_pose.position, p.gt_pose.quaternion].tolist() for p in parts]
            }
            if "replay" not in row:
                row["parts"] = geometry
        except Exception as error:
            errors.append(f"GT: {error}")
        rows.append(row)
    manifest = {
        "version": 1,
        "run": run.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "manual_format": "webp-lossless",
        "samples": [
            {k: r[k] for k in ("id", "status", "errors", "episode_source")}
            | {
                "call_count": len(r.get("replay", {}).get("calls", [])),
                "changed_calls": [
                    i for i, c in enumerate(r.get("replay", {}).get("calls", [])) if c["changed"]
                ],
            }
            for r in rows
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, suffix=".html", delete=False
    ) as stream:
        temp = Path(stream.name)
        try:
            stream.write(
                (WEB / "page.html").read_text().replace("__TITLE__", html.escape(run.name))
            )
            stream.write("<!--\n" + (WEB / "THREE-LICENSE.txt").read_text() + "\n-->\n")
            stream.write(
                '<script id="manifest" type="application/json">'
                + json.dumps(manifest).replace("<", "\\u003c")
                + "</script>\n"
            )
            print("Compressing and embedding result rows…", flush=True)
            for i, payload in enumerate(parallel_map(packed, rows)):
                stream.write(
                    f'<script id="sample-{i}" type="application/octet-stream">{payload}</script>\n'
                )
            for asset in ("three.bundle.js", "results.js"):
                stream.write(
                    "<script>"
                    + (WEB / asset).read_text().replace("</script", "<\\/script")
                    + "</script>\n"
                )
            stream.write("</body></html>")
            stream.close()
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)
    return {
        "output": str(output),
        "samples": len(rows),
        "replays": sum("replay" in r for r in rows),
        "ground_truths": sum("gt" in r for r in rows),
        "errors": {r["id"]: r["errors"] for r in rows if r["errors"]},
    }
