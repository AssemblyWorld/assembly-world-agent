import json
from dataclasses import replace
from pathlib import Path

import pytest

from assembly_world_agent import PreparationConfig, adapt_sample, prepare_sample
from assembly_world_agent.artifacts import (
    config_id,
    experiment,
    load_prepared,
    read_config,
    write_task,
)

pytest.importorskip("mujoco")


def test_configuration_isolates_revision_seeds_and_producer(row, tmp_path):
    source = adapt_sample("ikea-manual", row, revision="one")
    sample = prepare_sample(source)
    a = write_task(sample, tmp_path)
    b = write_task(prepare_sample(source, PreparationConfig(initialization_seed=1)), tmp_path)
    c = write_task(replace(sample, revision="two"), tmp_path)
    assert len({r["config_directory"] for r in (a, b, c)}) == 3
    config = read_config(a["config_directory"])
    identity = config["identity"]
    altered = {**identity, "producer": {**identity["producer"], "build": "f" * 64}}
    assert config_id(identity) != config_id(altered)


def test_rebuild_pins_hf_and_does_not_write_resources(row, tmp_path, monkeypatch):
    source = adapt_sample("ikea-manual", row, revision="resolved-commit")
    sample = prepare_sample(source)
    record = write_task(sample, tmp_path)
    directory = Path(record["config_directory"])
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    requests = []

    def fake_load(dataset, **kwargs):
        requests.append((dataset, kwargs))
        yield source

    monkeypatch.setattr("assembly_world_agent.artifacts.load_samples", fake_load)
    loaded = list(load_prepared(directory, cache_dir=tmp_path / "hf-cache"))
    assert len(loaded) == 1 and loaded[0][1].sample_id == sample.sample_id
    assert requests == [
        (
            sample.dataset,
            dict(
                revision="resolved-commit",
                sample_ids=[sample.sample_id],
                cache_dir=tmp_path / "hf-cache",
            ),
        )
    ]
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    with pytest.raises(ValueError, match="Unknown"):
        list(load_prepared(directory, sample_ids=["missing"]))


def test_logs_unique_and_retain_failed_status(tmp_path):
    with experiment("test", logs=tmp_path, inputs={"revision": "pinned"}) as (first, _, metrics):
        metrics["count"] = 3
    with pytest.raises(RuntimeError):
        with experiment("test", logs=tmp_path) as (second, _, metrics):
            metrics["partial"] = True
            raise RuntimeError("fixture failure")
    assert first != second
    assert json.loads((first / "meta.json").read_text())["status"] == "passed"
    assert json.loads((second / "meta.json").read_text())["status"] == "failed"
    assert json.loads((second / "metrics.json").read_text()) == {"partial": True}


def test_config_rejects_corrupt_archive_and_wrong_identity(row, tmp_path):
    record = write_task(
        prepare_sample(adapt_sample("ikea-manual", row, revision="fixture")), tmp_path
    )
    directory = Path(record["config_directory"])
    path = directory / record["episode"]
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_config(directory)
    path.write_bytes(original)
    config_path = directory / "config.json"
    value = json.loads(config_path.read_text())
    value["identity"]["revision"] = "changed"
    config_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        read_config(directory)


def test_multiple_samples_share_configuration_without_overwriting(row, tmp_path):
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision="fixture"))
    first = write_task(sample, tmp_path)
    second = write_task(replace(sample, sample_id="Another/sample"), tmp_path)
    assert first["config_directory"] == second["config_directory"]
    value = read_config(first["config_directory"])
    assert set(value["samples"]) == {sample.sample_id, "Another/sample"}


def test_cli_routes_reports_to_logs(row, tmp_path, monkeypatch):
    from assembly_world_agent.cli import DEFAULT_SAMPLES, main

    source = adapt_sample("ikea-manual", row, revision="resolved-commit")

    def fake_load(dataset, **kwargs):
        assert dataset == "ikea-manual" and kwargs["sample_ids"] == DEFAULT_SAMPLES
        yield source

    monkeypatch.setattr("assembly_world_agent.conversion.load_samples", fake_load)
    data, logs = tmp_path / "data", tmp_path / "logs"
    assert main(["convert-ikea", "--output", str(data), "--logs", str(logs)]) == 0
    assert len(list(data.rglob("*.episode.zip"))) == 1
    assert len(list(data.rglob("config.json"))) == 1
    assert len(list(data.rglob("*.*"))) == 2
    meta = json.loads(next(logs.glob("*/meta.json")).read_text())
    assert next(iter(meta["configurations"].values()))["revision"] == "resolved-commit"
    assert meta["status"] == "passed"
