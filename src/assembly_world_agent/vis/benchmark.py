"""Standalone Three.js preview of an AssemblyWorldBench data directory.

Every shape named by ``benchmark.json`` is embedded with its scattered initial
placement and its ground-truth assembly. Parts above a triangle budget are shown
through display-only vertex clustering; scoring never reads this page.
"""

from __future__ import annotations

import html
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..adapters import get_adapter
from ..artifacts import config_id
from ..benchmark import read_benchmark
from ..episodes import _render_mesh
from ..models import PreparationConfig
from ..preparation import prepare_sample
from .results import WEB, load_result_samples, packed, parallel_map

MESH_TRIANGLE_LIMIT = 60_000


def pose_list(pose):
    return [
        *np.asarray(pose.position, dtype=float).tolist(),
        *np.asarray(pose.quaternion, dtype=float).tolist(),
    ]


def compact_mesh(vertices, triangles, *, decimals=6):
    """Keep only the vertices this part's triangles reference and re-index them.

    Source meshes may share one vertex pool across every part of an object, so a
    part's triangle list can index a pool far larger than the part itself.
    """
    vertices = np.asarray(vertices, dtype=float)
    index = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    used, inverse = np.unique(index, return_inverse=True)
    return (
        np.round(vertices[used], decimals).tolist(),
        inverse.reshape(-1).astype(int).tolist(),
    )


def _canonical(triangles):
    """Rotate each triangle so its smallest index comes first, preserving winding."""
    shift = np.argmin(triangles, axis=1)
    rows = np.arange(len(triangles))[:, None]
    columns = (np.arange(3)[None, :] + shift[:, None]) % 3
    return triangles[rows, columns]


def decimate_mesh(vertices, triangles, *, limit=MESH_TRIANGLE_LIMIT):
    """Display-only vertex clustering on a uniform grid, refined until the budget fits.

    Returns ``(vertices, triangles, decimated)``. Meshes within the budget pass
    through unchanged. Clustered vertices move to their cell mean; degenerate and
    duplicate triangles are dropped. The result is for viewing, never for scoring.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(triangles) <= limit:
        return vertices, triangles, False
    low = vertices.min(axis=0)
    span = float(np.ptp(vertices, axis=0).max()) or 1.0
    resolution = 256
    while True:
        cells = np.minimum(
            np.floor((vertices - low) / span * resolution).astype(np.int64), resolution
        )
        keys = (cells[:, 0] * (resolution + 1) + cells[:, 1]) * (resolution + 1) + cells[:, 2]
        unique, inverse = np.unique(keys, return_inverse=True)
        inverse = inverse.reshape(-1)
        remapped = inverse[triangles]
        keep = (
            (remapped[:, 0] != remapped[:, 1])
            & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2])
        )
        remapped = np.unique(_canonical(remapped[keep]), axis=0)
        if len(remapped) <= limit or resolution <= 2:
            break
        resolution = max(2, int(resolution * 0.8))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    sums = np.zeros((len(unique), 3))
    np.add.at(sums, inverse, vertices)
    return sums / counts[:, None], remapped, True


def part_geometry(part, *, limit=MESH_TRIANGLE_LIMIT):
    """Compacted triangle mesh, decimated for display when above the triangle budget."""
    vertices, triangles = _render_mesh(part, validate=False)
    count = int(np.asarray(triangles).reshape(-1, 3).shape[0])
    vertices, triangles, decimated = decimate_mesh(vertices, triangles, limit=limit)
    compact_vertices, compact_triangles = compact_mesh(vertices, triangles)
    return {
        "kind": "mesh",
        "source_triangles": count,
        "decimated": decimated,
        "vertices": compact_vertices,
        "triangles": compact_triangles,
    }


def shape_metadata(benchmark):
    """One descriptor per distinct shape, with the blocks and reference modes it appears in."""
    shapes, index = [], {}
    for block in benchmark["blocks"]:
        for sid, row in block["samples"].items():
            key = (block["source"], sid)
            if key not in index:
                index[key] = len(shapes)
                shapes.append(
                    dict(
                        source=block["source"],
                        dataset=block["dataset"],
                        repo_id=block["repo_id"],
                        revision=block["revision"],
                        config_id=block["config_id"],
                        sample_id=sid,
                        category=row.get("category"),
                        parts=row["parts"],
                        band=row.get("band"),
                        sha256=row["sha256"],
                        data=block["data"],
                        blocks=[],
                        reference_modes=[],
                    )
                )
            shape = shapes[index[key]]
            shape["blocks"].append(block["name"])
            shape["reference_modes"].append(block["reference_mode"])
    return shapes


def shape_row(meta, sample, *, limit=MESH_TRIANGLE_LIMIT):
    """Geometry payload for one prepared shape, ordered by part id like the runtime."""
    parts = sorted(sample.parts, key=lambda p: p.part_id)
    if len(parts) != meta["parts"]:
        raise ValueError(f"{meta['sample_id']}: {len(parts)} parts, benchmark says {meta['parts']}")
    return dict(
        parts=[dict(id=p.part_id, **part_geometry(p, limit=limit)) for p in parts],
        initial=[pose_list(p.initial_pose) for p in parts],
        gt=[pose_list(p.gt_pose) for p in parts],
    )


def preparation_for(benchmark, shape):
    slug = get_adapter(shape["dataset"]).REPO_ID.split("/")[-1]
    path = Path(benchmark["_directory"]) / shape["data"] / slug / shape["config_id"] / "config.json"
    config = json.loads(path.read_text())
    if config.get("version") != 1 or config.get("config_id") != config_id(config["identity"]):
        raise ValueError(f"Invalid configuration identity: {path}")
    return PreparationConfig(**config["identity"]["preparation"])


def prepared_shapes(benchmark, shapes, *, cache_dir=None):
    """Yield (meta, prepared sample) in benchmark order, loading each source once."""
    by_source = {}
    for meta in shapes:
        by_source.setdefault(meta["source"], []).append(meta)
    for name, group in by_source.items():
        preparation = preparation_for(benchmark, group[0])
        wanted = {meta["sample_id"]: meta for meta in group}
        loaded = {}
        for source in load_result_samples(
            group[0]["dataset"],
            revision=group[0]["revision"],
            sample_ids=list(wanted),
            cache_dir=cache_dir,
        ):
            loaded[source.sample_id] = source
        missing = sorted(set(wanted) - set(loaded))
        if missing:
            raise ValueError(f"{name}: source samples not found: {missing}")
        for sid, meta in wanted.items():
            print(f"Preparing {name}/{sid}", file=sys.stderr, flush=True)
            yield meta, prepare_sample(loaded.pop(sid), preparation, sample_points=False)


def statistics(benchmark, shapes):
    """Descriptive counts derived from the benchmark definition alone."""
    sources = {}
    for source in benchmark["sources"]:
        rows = [s for s in shapes if s["source"] == source]
        sources[source] = dict(
            shapes=len(rows),
            band_counts=dict(sorted(Counter(str(r["band"]) for r in rows).items())),
            categories=dict(sorted(Counter(str(r["category"]) for r in rows).items())),
            parts_histogram=dict(
                sorted((str(k), v) for k, v in Counter(r["parts"] for r in rows).items())
            ),
        )
    return dict(
        evaluations=sum(len(b["samples"]) for b in benchmark["blocks"]),
        distinct_shapes=len(shapes),
        blocks=[
            dict(
                name=b["name"],
                source=b["source"],
                reference_mode=b["reference_mode"],
                samples=len(b["samples"]),
            )
            for b in benchmark["blocks"]
        ],
        sources=sources,
    )


def write_page(output, page, payloads):
    """Write the standalone page atomically; assets are inlined, nothing is fetched."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, suffix=".html", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(
                (WEB / "benchmark.html")
                .read_text()
                .replace("__TITLE__", html.escape(page["benchmark"]))
            )
            stream.write("<!--\n" + (WEB / "THREE-LICENSE.txt").read_text() + "\n-->\n")
            stream.write(
                '<script id="manifest" type="application/json">'
                + json.dumps(page).replace("<", "\\u003c")
                + "</script>\n"
            )
            for index, payload in enumerate(payloads):
                stream.write(
                    f'<script id="shape-{index}" type="application/octet-stream">{payload}</script>\n'
                )
            for asset in ("three.bundle.js", "benchmark.js"):
                stream.write(
                    "<script>"
                    + (WEB / asset).read_text().replace("</script", "<\\/script")
                    + "</script>\n"
                )
            stream.write("</body></html>")
            stream.close()
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def export_benchmark_preview(benchmark_path, output, *, cache_dir=None, limit=MESH_TRIANGLE_LIMIT):
    benchmark = read_benchmark(benchmark_path)
    shapes = shape_metadata(benchmark)
    rows = []
    decimated_parts = 0
    for meta, sample in prepared_shapes(benchmark, shapes, cache_dir=cache_dir):
        row = shape_row(meta, sample, limit=limit)
        decimated = [part["decimated"] for part in row["parts"]]
        decimated_parts += sum(decimated)
        meta["decimated"] = any(decimated)
        meta["source_triangles"] = sum(part["source_triangles"] for part in row["parts"])
        meta["shown_triangles"] = sum(len(part["triangles"]) // 3 for part in row["parts"])
        rows.append(row)
    for index, meta in enumerate(shapes):
        meta["script"] = f"shape-{index}"
    page = dict(
        version=2,
        benchmark=benchmark["benchmark"],
        generated=datetime.now(timezone.utc).isoformat(),
        mesh_triangle_limit=limit,
        statistics=statistics(benchmark, shapes),
        shapes=shapes,
    )
    print("Compressing and embedding shapes…", file=sys.stderr, flush=True)
    payloads = list(parallel_map(packed, rows))
    written = write_page(output, page, payloads)
    return dict(
        output=str(written),
        shapes=len(shapes),
        evaluations=page["statistics"]["evaluations"],
        decimated_parts=decimated_parts,
        bytes=written.stat().st_size,
    )
