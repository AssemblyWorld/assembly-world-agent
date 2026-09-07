"""Artifact bookkeeping for supervised browser-agent experiments."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from .artifacts import now, read_config, sample_name, write_json

PROMPT = """Assemble {sample_id} by following the supplied assembly manual.

Environment: {environment_url}
Initial episode: {episode_url}
Manualbook: {manual_url}
Manual page files (read only): {manual_directory}
Your report directory: {sample_directory}

Open a NEW dedicated Codex in-app browser tab for this sample. Discover and use
its native WebMCP tools. ALL scene observation, geometry queries, camera changes,
captures, translations, rotations, poses and groups must use those WebMCP tools.
Do not use UI manipulation, DOM/internal runtime inspection, local episode parsing,
local geometry analysis, GT, dataset annotations, other experiments, or external
assembly solutions to solve the task. Reading the supplied manual page images
with view_image or browser tools is allowed. Arithmetic on WebMCP results is allowed.
Do not read source code or any files outside this manualbook and your own reports.

Read the manual pages in order and assemble all supplied parts. Use the observations
and tool descriptions to infer the necessary transforms. Physics and collision
response are disabled. A group does not snap or weld parts. Body positions need
not equal geometric centers; use appropriate pivots when rotating.

After assembly, inspect the connections between parts shown by the manual using
multiple camera views and WebMCP captures. Check alignment, gaps, orientation and
obvious interpenetration. Correct problems and recheck. Report each checked pair,
the manual page/step, supporting capture call indices, and unresolved uncertainty.
Do not claim measured accuracy or physical stability. No fixed time/call budget.

At meaningful completed manual stages, notify the scheduler that a checkpoint is
available. At the end, write report.md and result.json in your report directory
(status: completed, partial, or unable; include checked_connections, uncertainties,
and browser_id/tab_id). The scheduler alone exports the full episode using the UI.
Before ending your turn, mark your tab for handoff so it remains available, then
report its browser and tab identifiers and ask the scheduler to export it. Never
close, reload or reset a tab containing unsaved work. Do not spawn other agents.
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(run, kind, **values):
    with (Path(run) / "scheduler.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(timestamp=now(), kind=kind, **values)) + "\n")


def create_run(configuration, logs, port=18765):
    configuration = Path(configuration).resolve()
    config = read_config(configuration)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = Path(logs).resolve() / f"{stamp}-ikea-manual-webmcp-102-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    write_json(
        run / "meta.json",
        dict(
            version=1,
            kind="ikea-manual-webmcp-102",
            status="preparing",
            started_at=now(),
            configuration=str(configuration),
            config=config,
            code_commit=commit,
            environment="https://3dwebagent.davidz.cn/",
            port=port,
            model_policy="inherit parent; record observed runtime configuration",
            concurrency=dict(pilot=1, initial=2, maximum=3),
            limits=dict(wall_time=None, calls=None, tokens=None),
            evaluation="agent final visual inspection; no GT scoring",
            manual_copy_authorization="Explicit user approval in experiment planning",
        ),
    )
    (run / "prompt-template.txt").write_text(PROMPT)
    write_json(run / "progress.json", {sid: dict(status="pending") for sid in config["samples"]})
    for sid, record in config["samples"].items():
        slug = sample_name(sid)
        directory = run / "samples" / slug
        directory.mkdir(parents=True)
        for name in ("manualbook", "trace", "checkpoints"):
            (directory / name).mkdir()
        episode_url = f"http://127.0.0.1:{port}/episodes/{quote(record['episode'])}"
        manual_url = f"http://127.0.0.1:{port}/manuals/{slug}/index.html"
        values = dict(
            sample_id=sid,
            episode_url=episode_url,
            manual_url=manual_url,
            environment_url="https://3dwebagent.davidz.cn/?episode=" + quote(episode_url, safe=""),
            manual_directory=str(directory / "manualbook"),
            sample_directory=str(directory),
        )
        write_json(
            directory / "input.json",
            dict(
                **values,
                **record,
                revision=config["identity"]["revision"],
                initial_path=str(configuration / record["episode"]),
            ),
        )
        (directory / "prompt.txt").write_text(PROMPT.format(**values))
    event(run, "created", samples=len(config["samples"]))
    return run


def prepare_manuals(run):
    from datasets import Image, load_dataset

    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    identity = meta["config"]["identity"]
    rows = load_dataset(
        identity["dataset"], revision=identity["revision"], split="full", streaming=True
    )
    features = rows.features.copy()
    features["manual_pages"].feature["image"] = Image(decode=False)
    rows = rows.cast(features).select_columns(["object_id", "manual_pages"])
    for row in rows:
        sid = row["object_id"]
        if sid not in meta["config"]["samples"]:
            continue
        directory = run / "samples" / sample_name(sid) / "manualbook"
        pages = []
        for number, page in enumerate(row["manual_pages"], 1):
            source = page["image"]
            payload = source["bytes"] or Path(source["path"]).read_bytes()
            name = f"page-{number:03d}.png"
            (directory / name).write_bytes(payload)
            pages.append(
                dict(
                    file=name,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    **{k: v for k, v in page.items() if k != "image"},
                )
            )
        write_json(
            directory / "pages.json",
            dict(sample_id=sid, revision=identity["revision"], pages=pages),
        )
        title = html.escape(sid)
        content = f'<!doctype html><meta charset="utf-8"><title>{title} manual</title>'
        content += "<style>body{max-width:1100px;margin:24px auto;font-family:sans-serif}img{width:100%;height:auto}figure{margin:16px 0}</style>"
        content += f"<h1>{title} assembly manual</h1>"
        for i, page in enumerate(pages, 1):
            content += f'<figure><figcaption>Page {i}</figcaption><img src="{page["file"]}" alt="Manual page {i}"></figure>'
        (directory / "index.html").write_text(content)
        update(run, sid, status="ready" if pages else "input_failed", manual_pages=len(pages))
        print(f"READY {sid}: {len(pages)} manual pages", flush=True)
    meta.update(status="running", inputs_ready_at=now())
    write_json(run / "meta.json", meta)


def update(run, sid, **values):
    run = Path(run)
    progress = json.loads((run / "progress.json").read_text())
    progress[sid].update(values, updated_at=now())
    write_json(run / "progress.json", progress)
    event(run, "sample_update", sample_id=sid, **values)
    summarize(run)


def summarize(run):
    run = Path(run)
    progress = json.loads((run / "progress.json").read_text())
    counts = {}
    outcomes = {}
    for record in progress.values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if outcome := record.get("agent_outcome"):
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    write_json(
        run / "metrics.json",
        dict(
            samples=len(progress),
            status_counts=counts,
            agent_outcome_counts=outcomes,
            evaluation="No computed assembly score; completion is agent-reported.",
        ),
    )
    with (run / "summary.csv").open("w") as stream:
        writer = csv.writer(stream)
        fields = ["status", "agent_outcome", "agent_id", "calls", "trace_records", "video", "error"]
        writer.writerow(["sample_id", *fields])
        for sid, row in progress.items():
            writer.writerow([sid] + [row.get(k, "") for k in fields])
    rows = []
    for sid, record in progress.items():
        directory = run / "samples" / sample_name(sid)
        links = []
        for name, label in (
            ("replay.mp4", "MP4 replay"),
            ("report.md", "Inspection report"),
            ("result.json", "Result"),
            ("evidence.json", "Evidence audit"),
            ("final.episode.zip", "Episode"),
            ("trace/records.jsonl", "Trace"),
        ):
            if (directory / name).is_file():
                path = quote(str((directory / name).relative_to(run)))
                links.append(f'<a href="{path}">{label}</a>')
        cells = [
            html.escape(sid),
            html.escape(record["status"]),
            html.escape(str(record.get("agent_outcome", ""))),
            str(record.get("calls", "")),
            " · ".join(links),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    (run / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>IKEA Manual experiment</title>"
        "<style>body{font:16px system-ui;margin:32px;color:#18212b;background:#fafbfc}"
        "table{border-collapse:collapse;width:100%;background:white}"
        "th,td{text-align:left;padding:10px;border-bottom:1px solid #dde2e7}"
        "th{position:sticky;top:0;background:#eef2f5}a{color:#075cb4}"
        "p{max-width:950px;line-height:1.5}</style>"
        "<h1>IKEA Manual experiment</h1>"
        f"<p>{len(progress)} samples. {html.escape(str(counts))}</p>"
        "<p>Completion is reported by the assembly agent after visual connection inspection. "
        "No ground-truth assembly score was computed. Replays show saved episode states and "
        "tool activity; they are not real-time screen recordings. "
        "Trace exports contain available messages and tool records, excluding private reasoning.</p>"
        '<p><a href="summary.csv">Summary CSV</a> · <a href="metrics.json">Metrics</a></p>'
        "<table><thead><tr><th>Sample</th><th>Archive status</th><th>Agent outcome</th>"
        "<th>WebMCP calls</th><th>Artifacts</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>"
    )


def archive_episode(run, sid, source, *, final=True):
    run = Path(run)
    source = Path(source)
    directory = run / "samples" / sample_name(sid)
    expected = json.loads((directory / "input.json").read_text())
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["id"] != expected["episode_id"]:
            raise ValueError("Downloaded episode belongs to a different sample")
        for name, checksum in manifest["hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"Invalid archive payload hash: {name}")
        calls = [json.loads(line) for line in archive.read("calls.jsonl").splitlines()]
    target = (
        directory / "final.episode.zip"
        if final
        else directory / "checkpoints" / f"calls-{len(calls):06d}.episode.zip"
    )
    if not final and target.exists() and digest(target) != digest(source):
        target = target.with_name(
            target.name.removesuffix(".episode.zip") + f"-{digest(source)[:12]}.episode.zip"
        )
    if target.exists() and digest(target) != digest(source):
        raise FileExistsError(target)
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(
        directory / "archive.json" if final else target.with_suffix(".json"),
        dict(
            source=str(source),
            path=str(target),
            sha256=digest(target),
            calls=len(calls),
            lifecycle=manifest["lifecycle"],
            verified_at=now(),
        ),
    )
    event(run, "episode_archived", sample_id=sid, calls=len(calls), path=str(target))
    return target


def move_verified_export(run, sid, source, checkpoint):
    """Move a verified browser download into its sample, preserving distinct bytes."""
    source, checkpoint = Path(source), Path(checkpoint)
    checksum = digest(source)
    if checksum != digest(checkpoint):
        raise ValueError("Download does not match its verified checkpoint")
    directory = Path(run) / "samples" / sample_name(sid) / "exports"
    directory.mkdir(exist_ok=True)
    target = directory / source.name
    if target.exists() and digest(target) != checksum:
        target = directory / f"{source.stem}-{checksum[:12]}{source.suffix}"
    if target.exists():
        if digest(target) != checksum:
            raise FileExistsError(target)
        source.unlink()
    else:
        shutil.move(str(source), target)
    if digest(target) != checksum:
        raise ValueError("Moved export checksum mismatch")
    event(
        run, "download_moved", sample_id=sid, source=str(source), path=str(target), sha256=checksum
    )
    return target


def collect_exports(run, downloads):
    """Verify exports and move browser downloads into their corresponding sample."""
    run, downloads = Path(run), Path(downloads)
    meta = json.loads((run / "meta.json").read_text())
    started = datetime.fromisoformat(meta["started_at"]).timestamp()
    progress = json.loads((run / "progress.json").read_text())
    index_path = run / "download-index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    patterns = {
        sid: re.compile(
            r"^ikea-manual___" + re.escape(sid.replace("/", "_")) + r"\.episode(?: \(\d+\))?\.zip$"
        )
        for sid, row in progress.items()
        if row["status"] in {"running", "archiving", "rendering", "archived"}
    }
    imported = []
    for source in downloads.glob("ikea-manual___*.zip"):
        stat = source.stat()
        if stat.st_mtime < started:
            continue
        key = f"{source.name}:{stat.st_size}:{stat.st_mtime_ns}"
        if key in index:
            record = index[key]
            if "checkpoint" in record:
                record["export"] = str(
                    move_verified_export(run, record["sample_id"], source, record["checkpoint"])
                )
                imported.append(record)
            continue
        sid = next(
            (sid for sid, pattern in patterns.items() if pattern.fullmatch(source.name)), None
        )
        if sid is None:
            continue
        try:
            if progress[sid]["status"] == "archived":
                directory = run / "samples" / sample_name(sid)
                known = list((directory / "checkpoints").glob("*.episode.zip"))
                known.append(directory / "final.episode.zip")
                checksum = digest(source)
                target = next((p for p in known if p.exists() and digest(p) == checksum), None)
                if target is None:
                    continue
            else:
                target = archive_episode(run, sid, source, final=False)
            moved = move_verified_export(run, sid, source, target)
        except (zipfile.BadZipFile, OSError):
            continue
        except ValueError as error:
            index[key] = dict(sample_id=sid, rejected=str(error))
            event(run, "download_rejected", sample_id=sid, source=str(source), error=str(error))
        else:
            index[key] = dict(sample_id=sid, checkpoint=str(target), export=str(moved))
            imported.append(index[key])
    write_json(index_path, index)
    return imported


def export_trace(run, sid, rollout):
    """Export task messages and tool records without private reasoning or instructions."""
    directory = Path(run) / "samples" / sample_name(sid) / "trace"
    source_stat = Path(rollout).stat()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous.get("source_size"), previous.get("source_mtime_ns")) == (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ):
            return previous["records"]
    records = []
    metadata = []
    for line in Path(rollout).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = row.get("payload", {})
        kind = row.get("type")
        if kind == "session_meta":
            metadata.append(
                {
                    k: payload[k]
                    for k in [
                        "id",
                        "timestamp",
                        "parent_thread_id",
                        "agent_path",
                        "model_provider",
                        "history_mode",
                    ]
                    if k in payload
                }
            )
        elif kind == "turn_context":
            metadata.append(
                {k: payload[k] for k in ["model", "effort", "reasoning_effort"] if k in payload}
            )
        elif kind == "token_usage_record":
            records.append(row)
        elif kind == "response_item":
            subtype = payload.get("type")
            if subtype in {
                "function_call",
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "agent_message",
            }:
                records.append(row)
            elif (
                subtype == "message"
                and payload.get("role") in {"assistant", "user"}
                and payload.get("channel") not in {"analysis", "justify", "confidence"}
            ):
                records.append(row)
    with tempfile.NamedTemporaryFile(dir=directory, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        for row in records:
            stream.write(json.dumps(row) + "\n")
    temporary.replace(directory / "records.jsonl")
    write_json(
        directory / "manifest.json",
        dict(
            source=str(rollout),
            metadata=metadata,
            records=len(records),
            exported_at=now(),
            source_sha256=digest(rollout),
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
            scope="Exportable task messages, tool calls/results, usage. Private reasoning and system/developer instructions excluded.",
        ),
    )
    return len(records)


def render_sample(run, sid):
    """Render and decode-check one saved sample inside its existing experiment."""
    if not json.loads((Path(run) / "meta.json").read_text()).get("generate_mp4", True):
        raise RuntimeError("MP4 generation is disabled for this experiment")
    from .vis import render_episode
    from .vis.replay import read_episode

    run = Path(run)
    directory = run / "samples" / sample_name(sid)
    source = directory / "final.episode.zip"
    read_episode(source)
    metadata = dict(status="rendering", started_at=now(), episode_sha256=digest(source))
    write_json(directory / "render.json", metadata)
    try:
        result = render_episode(source, directory / "replay", formats=("mp4",))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(directory / "replay.mp4"), "-f", "null", "-"],
            check=True,
            capture_output=True,
        )
        metadata.update(status="passed", result=result, decode_check=True)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["finished_at"] = now()
        write_json(directory / "render.json", metadata)
    event(run, "video_ready", sample_id=sid, path=str(directory / "replay.mp4"))
    return metadata


def audit_evidence(run, sid):
    """Resolve report capture references without changing the solver's report."""
    directory = Path(run) / "samples" / sample_name(sid)
    report = json.loads((directory / "result.json").read_text())
    references = set()
    for pair in report.get("checked_connections", []):
        if not isinstance(pair, dict):
            continue
        for key, values in pair.items():
            if "capture" in key and isinstance(values, list):
                references.update(value for value in values if type(value) is int)
    with zipfile.ZipFile(directory / "final.episode.zip") as archive:
        calls = [json.loads(line) for line in archive.read("calls.jsonl").splitlines()]
    captures = {call["index"]: call for call in calls if call["name"] == "capture_scene"}
    offsets = [
        offset
        for offset in (0, 1)
        if references and all(ref - offset in captures for ref in references)
    ]
    evidence = dict(
        status="passed" if len(offsets) == 1 else "unverified",
        report_sha256=digest(directory / "result.json"),
        episode_sha256=digest(directory / "final.episode.zip"),
        references=sorted(references),
        candidate_numbering_offsets=offsets,
        scope="Checks that numeric report references identify capture calls, not assembly accuracy.",
        verified_at=now(),
    )
    if len(offsets) == 1:
        offset = offsets[0]
        evidence["numbering"] = "one-based UI call count" if offset else "zero-based episode index"
        evidence["captures"] = [
            dict(
                reported_reference=ref,
                episode_call_index=ref - offset,
                state_index=captures[ref - offset]["state_index"],
                timestamp=captures[ref - offset]["timestamp"],
            )
            for ref in sorted(references)
        ]
    write_json(directory / "evidence.json", evidence)
    return evidence


def finalize_ready(run):
    """Require episode, report and trace, plus video when enabled by the run."""
    run = Path(run)
    require_video = json.loads((run / "meta.json").read_text()).get("generate_mp4", True)
    progress = json.loads((run / "progress.json").read_text())
    completed = []
    for sid, record in progress.items():
        if record["status"] not in {"archiving", "rendering"}:
            continue
        directory = run / "samples" / sample_name(sid)
        paths = {
            name: directory / name
            for name in (
                "result.json",
                "report.md",
                "final.episode.zip",
                "archive.json",
                "trace/manifest.json",
                "trace/records.jsonl",
            )
        }
        if not all(path.exists() for path in paths.values()):
            continue
        video_values = {}
        if require_video:
            render_path = directory / "render.json"
            video_path = directory / "replay.mp4"
            if not render_path.is_file() or not video_path.is_file():
                continue
            render = json.loads(render_path.read_text())
            if render["status"] != "passed" or not render.get("decode_check"):
                continue
            video_values["video"] = str(video_path)
        elif (directory / "replay.mp4").is_file():
            video_values["video"] = str(directory / "replay.mp4")
        result = json.loads(paths["result.json"].read_text())
        trace = json.loads(paths["trace/manifest.json"].read_text())
        evidence = audit_evidence(run, sid)
        update(
            run,
            sid,
            status="archived",
            agent_outcome=result["status"],
            trace_records=trace["records"],
            evidence_status=evidence["status"],
            **video_values,
        )
        completed.append(sid)
    return completed
