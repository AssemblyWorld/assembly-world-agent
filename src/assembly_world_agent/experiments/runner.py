"""Dataset selection, bounded workers, compact artifacts and fresh-run recovery."""

import asyncio
import io
import json
import signal
import tempfile
import time
import uuid
from pathlib import Path

from ..adapters import REFERENCE_MODES, get_adapter, reference_pages
from ..artifacts import now, read_config, sample_name, write_json
from ..episode_io import read_episode
from ..episodes import sha256
from .agents import prompt_content, run_agent
from .browser import DEFAULT_ENVIRONMENT, Browser, doctor
from .serving import EpisodeServer
from .transport import append

DEFAULT_TASK = """Assemble the supplied parts into a coherent object. If a manual is supplied,
read its pages in order and follow it. Otherwise infer the assembly from geometry.
Inspect connections from multiple camera views using capture_scene, correct gaps,
orientation and obvious interpenetration, and report uncertainties. Do not claim
computed accuracy or physical stability without evidence."""
PROTOCOL = """
The scheduler has already loaded your episode in the dedicated browser.
Use list_pages to find the environment page, then list_webmcp_tools and
execute_webmcp_tool. ALL scene observations, captures and transformations must use
these page WebMCP tools. Do not navigate, reload, close or reset the page.
Do not inspect source code, local geometry, ground truth or other experiments.
Do not spawn agents. Preserve the existing physics and tool settings.
Manual pages: {manual_count}. Read them with read_manual_page (one-based page).
The scheduler will export the full episode after your final answer; do not export
or write files yourself. End with a JSON object containing status (completed,
partial or unable), summary, checked_connections and uncertainties.
"""
REFERENCE_PROTOCOL = PROTOCOL.replace(
    "Manual pages: {manual_count}. Read them with read_manual_page (one-based page).",
    "Reference images, if supplied, are attached in page order. You may reread only\n"
    "these images with read_manual_page (one-based page).",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def select_inputs(options):
    config = None
    if options.get("episode"):
        path = Path(options["episode"]).resolve()
        paths = {path.name.removesuffix(".episode.zip"): path}
        if (path.parent / "config.json").exists():
            candidate = read_config(path.parent)
            matching = {
                sid: value
                for sid, value in candidate["samples"].items()
                if value["episode"] == path.name and value["sha256"] == sha256(path.read_bytes())
            }
            if matching:
                config = {**candidate, "samples": matching}
                paths = {sid: path for sid in matching}
    else:
        adapter = get_adapter(options["dataset"])
        directory = Path(options["data"]) / adapter.REPO_ID.split("/")[-1] / options["config_id"]
        config = read_config(directory)
        if config["identity"]["dataset"] != adapter.REPO_ID:
            raise ValueError("Dataset differs from preparation configuration")
        selected = options.get("sample_id") or list(config["samples"])
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate sample selection")
        unknown = set(selected) - config["samples"].keys()
        if unknown:
            raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")
        if options.get("limit"):
            selected = selected[: options["limit"]]
        config = {**config, "samples": {sid: config["samples"][sid] for sid in selected}}
        paths = {sid: (directory / config["samples"][sid]["episode"]).resolve() for sid in selected}
    inputs = {}
    slugs = set()
    for sid, path in paths.items():
        slug = sample_name(sid)
        if slug in slugs:
            raise ValueError("Sample directory collision")
        slugs.add(slug)
        episode = read_episode(path)
        record = config["samples"][sid] if config else {}
        inputs[sid] = {
            **record,
            "sample_id": sid,
            "initial_path": str(path),
            "sha256": sha256(path.read_bytes()),
            "episode_id": episode["manifest"]["id"],
            "initial_calls": len(episode["calls"]),
            "revision": config["identity"]["revision"] if config else None,
        }
    return config, inputs


def create_run(options, config, inputs, *, source_run=None, task=None, versions=None):
    directory = Path(options["logs"]).resolve() / (
        now().replace(":", "").replace("+0000", "Z") + "-browser-" + uuid.uuid4().hex[:8]
    )
    directory.mkdir(parents=True)
    task = (
        task
        if task is not None
        else (
            Path(options["prompt_file"]).read_text() if options.get("prompt_file") else DEFAULT_TASK
        )
    )
    meta = {
        "version": 1,
        "kind": "browser-webmcp",
        "status": "running",
        "started_at": now(),
        "options": options,
        "config": config,
        "samples": list(inputs),
        "task": task,
        "versions": versions or {},
        "source_run": source_run,
    }
    import subprocess

    try:
        meta["code_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        meta["code_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"]))
    except (OSError, subprocess.CalledProcessError):
        meta["code_commit"] = None
    write_json(directory / "run.json", meta)
    for sid, value in inputs.items():
        sample = directory / "samples" / sample_name(sid)
        write_json(sample / "input.json", value)
        write_json(sample / "result.json", {"status": "pending"})
        (sample / "prompt.txt").write_text(task)
        (sample / "conversation.jsonl").touch()
    return directory


def manual_pages(inputs, meta, root):
    """Read only manual columns from pinned data, never adapt GT geometry for the agent."""
    from PIL import Image

    pages = []
    mode = meta["options"].get("reference_mode")
    if mode is not None and mode not in REFERENCE_MODES:
        raise ValueError(f"Unsupported reference mode: {mode}")
    if mode == "none":
        inputs["manual"] = {"source": None, "reference_mode": mode, "pages": []}
        return []
    explicit = meta["options"].get("manual")
    if explicit:
        directory = Path(explicit)
        sources = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not sources:
            raise ValueError("Manual directory contains no supported images")
        provenance = {"source": str(directory.resolve())}
        if mode == "final-image":
            sources = sources[:1]
        selected = [{"source_file": p.name} for p in sources]
    elif meta.get("config"):
        from datasets import Image as HFImage
        from datasets import load_dataset

        identity = meta["config"]["identity"]
        adapter = get_adapter(identity["dataset"])
        rows = load_dataset(
            adapter.REPO_ID,
            split="full",
            revision=identity["revision"],
            **getattr(adapter, "LOAD_KWARGS", {}),
        )
        provenance = {"dataset": adapter.REPO_ID, "revision": identity["revision"]}
        if "manual_pages" not in rows.column_names:
            if mode is not None:
                raise ValueError("Pinned dataset has no reference images")
            inputs["manual"] = {**provenance, "pages": []}
            return []
        columns = [adapter.SAMPLE_ID_FIELD, "manual_pages"]
        if mode is not None:
            columns.extend(getattr(adapter, "REFERENCE_COLUMNS", ()))
        rows = rows.select_columns(columns)
        features = rows.features.copy()
        features["manual_pages"].feature["image"] = HFImage(decode=False)
        rows = rows.cast(features)
        found = next(
            (row for row in rows if row[adapter.SAMPLE_ID_FIELD] == inputs["sample_id"]), None
        )
        if found is None:
            raise ValueError("Sample missing from pinned manual source")
        selected = (
            found["manual_pages"] if mode is None else reference_pages(adapter.REPO_ID, found, mode)
        )
        sources = [p["image"] for p in selected]
    else:
        if mode is not None:
            raise ValueError("Reference images require pinned dataset provenance or --manual")
        inputs["manual"] = {"source": None, "pages": []}
        return []
    records = []
    for index, source in enumerate(sources, 1):
        if isinstance(source, dict):
            payload = source.get("bytes") or Path(source["path"]).read_bytes()
        else:
            payload = Path(source).read_bytes()
        path = Path(root) / f"manual-{index:04d}.png"
        with Image.open(io.BytesIO(payload)) as image:
            if mode is None:
                if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
                    image = image.convert("RGB")
                image.save(path)
            else:
                suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(image.format)
                if suffix is None:
                    raise ValueError(f"Unsupported reference image format: {image.format}")
                image.load()
                path = path.with_suffix(suffix)
                path.write_bytes(payload)
        pages.append(str(path))
        records.append(
            {
                "page": index,
                "source_sha256": sha256(payload),
                "image_sha256": sha256(path.read_bytes()),
            }
        )
        if mode is not None:
            records[-1].update(
                {
                    k: selected[index - 1][k]
                    for k in (
                        "source_file",
                        "manual_id",
                        "page_index",
                        "view_id",
                        "step_id",
                        "kind",
                    )
                    if k in selected[index - 1]
                }
            )
    if mode is not None:
        provenance["reference_mode"] = mode
    inputs["manual"] = {**provenance, "pages": records}
    return pages


def outcome(text):
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict) and value.get("status") in {
                "completed",
                "partial",
                "unable",
            }:
                return value
        except ValueError:
            continue
    return None


async def execute_sample(
    directory, sid, meta, *, episode_url, browser_factory=Browser, agent=run_agent
):
    sample = directory / "samples" / sample_name(sid)
    inputs = read_json(sample / "input.json")
    result = {
        "status": "running",
        "started_at": now(),
        "execution": {"status": "pending"},
        "archive": {"status": "not_saved"},
        "agent_outcome": None,
    }
    write_json(sample / "result.json", result)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="awa-sample-") as root:
        browser = browser_factory(root, meta["options"])
        try:
            if sha256(Path(inputs["initial_path"]).read_bytes()) != inputs["sha256"]:
                raise ValueError("Input episode changed since run creation")
            previous_manual = inputs.get("manual")
            preparation = asyncio.create_task(asyncio.to_thread(manual_pages, inputs, meta, root))
            try:
                pages = await asyncio.shield(preparation)
            except asyncio.CancelledError:
                # A running dataset thread must finish before its temporary directory is removed.
                await preparation
                raise
            if previous_manual is not None and inputs.get("manual") != previous_manual:
                raise ValueError("Manual changed since the source run")
            write_json(sample / "input.json", inputs)
            protocol = (
                REFERENCE_PROTOCOL
                if meta["options"].get("reference_mode") is not None
                else PROTOCOL.format(manual_count=len(pages))
            )
            prompt = meta["task"] + "\n" + protocol
            (sample / "prompt.txt").write_text(prompt)
            attachments = pages if meta["options"].get("reference_mode") is not None else []
            append(
                sample / "conversation.jsonl",
                "message",
                role="user",
                content=prompt_content(prompt, attachments) if attachments else prompt,
            )
            await browser.start(episode_url)
            bridge = Path(root) / "bridge.json"
            write_json(
                bridge,
                {
                    "command": browser.mcp_command(),
                    "manual": pages,
                    "conversation": str(sample / "conversation.jsonl"),
                },
            )
            result["execution"] = {"status": "running"}
            write_json(sample / "result.json", result)
            async with asyncio.timeout(meta["options"].get("timeout_seconds")):
                kwargs = {"images": attachments} if attachments else {}
                result["execution"] = await agent(
                    meta["options"], root, bridge, prompt, sample / "conversation.jsonl", **kwargs
                )
            result["agent_outcome"] = outcome(result["execution"].get("final_answer", ""))
        except TimeoutError:
            result["execution"] = {"status": "timeout"}
        except asyncio.CancelledError:
            result["execution"] = {"status": "interrupted"}
        except Exception as error:
            result["execution"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        finally:
            try:
                if browser.loaded:
                    result["archive"] = await browser.export(sample / "final.episode.zip", inputs)
            except Exception as error:
                result["archive"] = {"status": "failed", "error": str(error)}
            try:
                await browser.close()
            except Exception as error:
                result["cleanup_error"] = str(error)
            result["finished_at"] = now()
            result["duration_seconds"] = time.monotonic() - started
            result["status"] = (
                "completed"
                if result["execution"]["status"] == "completed"
                and result["archive"]["status"] == "saved"
                else "failed"
            )
            write_json(sample / "result.json", result)
    return result


def status(directory):
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    samples = {
        sid: read_json(directory / "samples" / sample_name(sid) / "result.json")["status"]
        for sid in meta["samples"]
    }
    counts = {state: list(samples.values()).count(state) for state in sorted(set(samples.values()))}
    return {"run": str(directory), "status": meta["status"], "counts": counts, "samples": samples}


async def schedule(directory, *, execute=execute_sample):
    directory = Path(directory)
    meta = read_json(directory / "run.json")
    with EpisodeServer(meta["options"].get("environment_url", DEFAULT_ENVIRONMENT)) as server:
        urls = {
            sid: server.add(
                read_json(directory / "samples" / sample_name(sid) / "input.json")["initial_path"]
            )
            for sid in meta["samples"]
        }
        return await _schedule(directory, meta, urls, execute)


async def _schedule(directory, meta, urls, execute):
    queue = asyncio.Queue()
    for sid in meta["samples"]:
        queue.put_nowait(sid)
    interrupted = False
    workers = []

    async def worker():
        while not interrupted:
            try:
                sid = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            print(f"{sid}: running", flush=True)
            result = await execute(directory, sid, meta, episode_url=urls[sid])
            print(f"{sid}: {result['status']} ({queue.qsize()} pending)", flush=True)
            queue.task_done()

    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            for task in workers:
                task.cancel()

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop)
    try:
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(meta["options"]["concurrency"], queue.qsize()))
        ]
        results = await asyncio.gather(*workers, return_exceptions=True)
        errors = [str(r) for r in results if isinstance(r, Exception)]
        if errors:
            meta["errors"] = errors
        counts = status(directory)["counts"]
        meta["status"] = (
            "interrupted"
            if interrupted
            else "completed"
            if counts.get("completed", 0) == len(meta["samples"])
            else "failed"
        )
        meta["finished_at"] = now()
        write_json(directory / "run.json", meta)
        return 0 if meta["status"] == "completed" else 130 if interrupted else 1
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def resume_inputs(source, retry_failed=False):
    source = Path(source).resolve()
    meta = read_json(source / "run.json")
    inputs = {}
    for sid in meta["samples"]:
        sample = source / "samples" / sample_name(sid)
        result = read_json(sample / "result.json")
        archive = result.get("archive", {})
        final = sample / "final.episode.zip"
        missing_archive = result["status"] == "completed" and (
            not final.is_file() or sha256(final.read_bytes()) != archive.get("sha256")
        )
        if (
            result["status"] in {"pending", "running"}
            or missing_archive
            or result.get("execution", {}).get("status") == "interrupted"
            or retry_failed
            and result["status"] == "failed"
        ):
            inputs[sid] = read_json(sample / "input.json")
    config = meta.get("config")
    if config:
        config = {**config, "samples": {sid: config["samples"][sid] for sid in inputs}}
    return meta, config, inputs


async def launch(options, *, source=None, retry_failed=False):
    if source:
        previous, config, inputs = resume_inputs(source, retry_failed)
        if not inputs:
            print("No samples need resuming.")
            return 0
        overrides = {k: v for k, v in options.items() if v is not None}
        options = {**previous["options"], **overrides}
        task = previous["task"]
    else:
        options = {"reference_mode": "manualbook", **options}
        config, inputs = select_inputs(options)
        task = None
    versions = await doctor(options)
    directory = create_run(
        options,
        config,
        inputs,
        source_run=str(Path(source).resolve()) if source else None,
        task=task,
        versions=versions,
    )
    print(f"Run: {directory}", flush=True)
    return await schedule(directory)
