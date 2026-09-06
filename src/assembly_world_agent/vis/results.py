"""Self-contained result snapshots from pinned tasks and saved runtime states."""

from __future__ import annotations

import base64
import gzip
import html
import io
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..episodes import _obj, sha256
from ..loading import load_samples
from ..models import Mesh, PreparationConfig
from ..preparation import prepare_sample
from ..utils.geometry import triangulate
from .replay import read_episode

WEB = Path(__file__).parent / "web"


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
            before = sha256(path.read_bytes())
            episode = read_episode(path)
            if expected_id is not None and episode["manifest"]["id"] != expected_id:
                raise ValueError("Episode identity differs from configured sample")
            if sha256(path.read_bytes()) != before:
                raise ValueError("Episode changed during snapshot")
            return episode, {"path": str(path.relative_to(directory)), "sha256": before}, errors
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
            {"id": body.attrib["name"], "resource": resource, "geometry": obj_mesh(files[resource])}
        )
    model = mj.MjModel.from_xml_string(
        files["world/model.xml"].decode(),
        assets={
            k[6:]: v for k, v in files.items() if k.startswith("world/") and k != "world/model.xml"
        },
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
    """Encode verified lossless WebP pixels without resizing."""
    with Image.open(io.BytesIO(raw)) as im:
        pixels = im.convert("RGBA")
        output = io.BytesIO()
        metadata = {key: im.info[key] for key in ("icc_profile", "exif") if key in im.info}
        pixels.save(output, format="WEBP", lossless=True, exact=True, method=6, **metadata)
        raw = output.getvalue()
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.size != pixels.size or decoded.convert("RGBA").tobytes() != pixels.tobytes():
                raise ValueError("Lossless WebP pixel verification failed")
    return {
        "data": "data:image/webp;base64," + base64.b64encode(raw).decode(),
        "embedded_sha256": sha256(raw),
        "image_format": "webp-lossless",
    }


def manual_pages(directory, errors):
    pages = []
    try:
        manifest = json.loads((directory / "manualbook/pages.json").read_text())
        for page in manifest["pages"]:
            try:
                path = directory / "manualbook" / page["file"]
                if path.resolve().parent != (directory / "manualbook").resolve():
                    raise ValueError("Unsafe manual path")
                raw = path.read_bytes()
                if sha256(raw) != page["sha256"]:
                    raise ValueError("Manual checksum mismatch")
                pages.append(page | encode_manual_image(raw))
            except Exception as error:
                pages.append(page | {"error": str(error)})
                errors.append(f"Manual {page['file']}: {error}")
    except Exception as error:
        errors.append(f"Manual: {error}")
    return pages


def export_results(run, output, *, cache_dir=None):
    """Build an atomic HTML snapshot; preserve the run and isolate sample failures."""
    run, output = Path(run).resolve(), Path(output).resolve()
    meta = json.loads((run / "meta.json").read_text())
    config = meta["config"]
    identity = config["identity"]
    progress = (
        json.loads((run / "progress.json").read_text()) if (run / "progress.json").exists() else {}
    )
    rows = []
    # Read a single pinned dataset stream, avoiding one dataset scan per sample.
    sources, source_error = {}, None
    print("Loading pinned GT source samples…", flush=True)
    try:
        for source in load_samples(
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
            "manual": manual_pages(directory, errors),
        }
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
                sources.pop(sid), PreparationConfig(**identity["preparation"])
            )
            parts = sorted(prepared.parts, key=lambda p: p.part_id)
            geometry = []
            for i, part in enumerate(parts):
                name = f"part-{i + 1:04d}"
                raw = _obj(part)
                if "replay" in row:
                    actual = row["replay"]["parts"][i]
                    if actual["id"] != name or ep["files"][actual["resource"]] != raw:
                        raise ValueError("Rebuilt GT geometry differs from recorded geometry")
                geometry.append({"id": name, "geometry": obj_mesh(raw)})
            if "replay" in row and len(parts) != len(row["replay"]["parts"]):
                raise ValueError("GT part count differs")
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
            for i, row in enumerate(rows):
                stream.write(
                    f'<script id="sample-{i}" type="application/octet-stream">{packed(row)}</script>\n'
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
