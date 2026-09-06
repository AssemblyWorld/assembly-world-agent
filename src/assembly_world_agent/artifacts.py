"""Configuration-addressed initial archives and isolated experiment logs."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .episodes import export_episode, json_bytes, sha256
from .loading import load_samples
from .models import PreparationConfig
from .preparation import prepare_sample


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def sample_name(sample_id):
    name = sample_id.replace("/", "--")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Unsupported sample filename: {sample_id}")
    return name


def config_id(identity):
    config = identity["preparation"]
    return (
        f"prep-v1-s{config['sampling_seed']}-i{config['initialization_seed']}-"
        + sha256(json_bytes(identity))[:12]
    )


def read_config(directory):
    directory = Path(directory)
    value = json.loads((directory / "config.json").read_text())
    if value["version"] != 1 or value["config_id"] != config_id(value["identity"]):
        raise ValueError("Invalid configuration identity")
    names = set()
    for sample_id, entry in value["samples"].items():
        name = sample_name(sample_id) + ".episode.zip"
        if entry["episode"] != name or name in names:
            raise ValueError("Invalid or ambiguous episode filename")
        names.add(name)
        path = directory / name
        if sha256(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Episode checksum differs: {path}")
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest["id"] != entry["episode_id"]
            or manifest["producer"] != value["identity"]["producer"]
        ):
            raise ValueError("Episode identity differs from config")
    return value


def register_archive(archive, output, *, dataset, revision, protocol, preparation):
    """Register an existing initial ZIP without changing its bytes (also used for migration)."""
    payload = Path(archive).read_bytes()
    with zipfile.ZipFile(archive) as stream:
        manifest = json.loads(stream.read("manifest.json"))
        if manifest["lifecycle"] != "setup":
            raise ValueError("Only setup episodes belong in data")
        for name, digest in manifest["hashes"].items():
            if sha256(stream.read(name)) != digest:
                raise ValueError(f"Archive checksum differs: {name}")
    identity = dict(
        dataset=dataset,
        revision=revision,
        protocol=protocol,
        preparation=preparation,
        producer=manifest["producer"],
        contract=manifest["contract"],
        engine=manifest["engine"],
    )
    slug = dataset.split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug):
        raise ValueError("Invalid dataset directory")
    directory = Path(output) / slug / config_id(identity)
    return directory, identity, payload, manifest


def save_archive(archive, output, *, sample_id, dataset, revision, protocol, preparation):
    directory, identity, payload, manifest = register_archive(
        archive,
        output,
        dataset=dataset,
        revision=revision,
        protocol=protocol,
        preparation=preparation,
    )
    directory.mkdir(parents=True, exist_ok=True)
    config = (
        read_config(directory)
        if (directory / "config.json").exists()
        else dict(version=1, config_id=config_id(identity), identity=identity, samples={})
    )
    if config["identity"] != identity:
        raise ValueError("Configuration digest collision")
    name = sample_name(sample_id) + ".episode.zip"
    entry = dict(
        episode=name,
        episode_id=manifest["id"],
        sha256=sha256(payload),
        parts=len(manifest["objects"]),
    )
    for other, record in config["samples"].items():
        if other != sample_id and record["episode"] == name:
            raise ValueError("Sample filename collision")
    if sample_id in config["samples"] and config["samples"][sample_id] != entry:
        raise FileExistsError("Different sample already registered")
    path = directory / name
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Different archive exists: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    config["samples"][sample_id] = entry
    config["samples"] = dict(sorted(config["samples"].items()))
    write_json(directory / "config.json", config)
    return dict(config_directory=str(directory), sample_id=sample_id, **entry)


def write_task(sample, output=Path("data")):
    """Persist an initial episode and provenance only; source/GT stay in memory."""
    with tempfile.TemporaryDirectory(prefix="awa-export-") as temporary:
        result = export_episode(sample, Path(temporary) / "initial.zip")
        return save_archive(
            result.path,
            output,
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            revision=sample.revision,
            protocol=sample.protocol_version,
            preparation=asdict(sample.config),
        )


def load_prepared(directory, *, cache_dir=None, sample_ids=None):
    """Rebuild selected prepared samples from pinned HF data, never private sidecars."""
    config = read_config(directory)
    identity = config["identity"]
    selected = list(config["samples"]) if sample_ids is None else list(sample_ids)
    if not set(selected) <= config["samples"].keys():
        raise ValueError("Unknown configured sample")
    for source in load_samples(
        identity["dataset"], revision=identity["revision"], sample_ids=selected, cache_dir=cache_dir
    ):
        sample = prepare_sample(source, PreparationConfig(**identity["preparation"]))
        if sample.protocol_version != identity["protocol"]:
            raise ValueError("Preparation protocol differs from saved configuration")
        yield source, sample


def now():
    return datetime.now(UTC).isoformat()


@contextmanager
def experiment(kind, *, logs=Path("logs"), inputs=None):
    """Always retain status and partial metrics, including on a failed run."""
    name = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + sample_name(kind)
        + "-"
        + uuid.uuid4().hex[:8]
    )
    directory = Path(logs) / name
    directory.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).parent
    files = sorted(package.rglob("*.py"))
    build = sha256(
        b"".join(p.relative_to(package).as_posix().encode() + p.read_bytes() for p in files)
    )
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True)
    meta = dict(
        version=1,
        kind=kind,
        started_at=now(),
        status="running",
        command=sys.argv,
        python=platform.python_version(),
        code_commit=git.stdout.strip() if git.returncode == 0 else None,
        source_sha256=build,
        inputs=inputs,
    )
    packages = {}
    for name in ("numpy", "scipy", "datasets", "mujoco", "playwright"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    meta["packages"] = packages
    script = Path(sys.argv[0])
    if script.is_file():
        meta["entrypoint_sha256"] = sha256(script.read_bytes())
    lock = package.parents[1] / "uv.lock"
    if lock.is_file():
        meta["lock_sha256"] = sha256(lock.read_bytes())
    metrics = {}
    write_json(directory / "meta.json", meta)
    try:
        yield directory, meta, metrics
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        meta["status"] = "passed"
    finally:
        meta["finished_at"] = now()
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "meta.json", meta)
