import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from assembly_world_agent import adapt_sample, cli, conversion
from assembly_world_agent.adapters import ADAPTERS
from assembly_world_agent.artifacts import read_config


@pytest.mark.parametrize("dataset", list(ADAPTERS))
@pytest.mark.parametrize("full_id", [False, True])
@pytest.mark.parametrize("model_format", [None, "mjb"])
def test_cli_routes_all_registered_datasets(monkeypatch, dataset, full_id, model_format):
    mock = Mock()
    monkeypatch.setattr(cli, "convert_dataset", mock)
    name = dataset if full_id else dataset.split("/")[-1]
    option = ["--model-format", model_format] if model_format else []
    assert cli.main(["convert", name, "--limit", "3", *option]) == 0
    assert mock.call_args.args == (name,)
    assert mock.call_args.kwargs["limit"] == 3
    assert mock.call_args.kwargs["sample_ids"] is None
    assert mock.call_args.kwargs["all_samples"] is False
    assert mock.call_args.kwargs["model_format"] == (model_format or "xml")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--limit", "0"],
        ["--limit", "-1"],
        ["--limit", "1.5"],
        ["--limit", "3", "--all"],
        ["--sample-id", "x", "--all"],
        ["--sample-id", "x", "--limit", "1"],
    ],
)
def test_cli_rejects_missing_or_invalid_selection_before_loading(monkeypatch, arguments):
    mock = Mock()
    monkeypatch.setattr(cli, "convert_dataset", mock)
    with pytest.raises(SystemExit) as error:
        cli.main(["convert", "fantastic-breaks", *arguments])
    assert error.value.code == 2
    mock.assert_not_called()


def test_unknown_dataset_rejected_before_workflow(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(cli, "convert_dataset", mock)
    with pytest.raises(SystemExit) as error:
        cli.main(["convert", "unknown", "--all"])
    assert error.value.code == 2
    mock.assert_not_called()


@pytest.mark.parametrize(
    "selection,expected",
    [
        (["--sample-id", "00/00002", "--sample-id", "00/00003"], ["00/00002", "00/00003"]),
        (["--all"], None),
    ],
)
def test_cli_forwards_selection_and_configuration(monkeypatch, selection, expected):
    mock = Mock()
    monkeypatch.setattr(cli, "convert_dataset", mock)
    assert (
        cli.main(
            [
                "convert",
                "fantastic-breaks",
                *selection,
                "--revision",
                "tag",
                "--cache-dir",
                "cache",
                "--output",
                "tasks",
                "--logs",
                "reports",
                "--sampling-seed",
                "7",
                "--initialization-seed",
                "8",
            ]
        )
        == 0
    )
    args = mock.call_args.kwargs
    assert args["sample_ids"] == expected
    assert args["all_samples"] == (expected is None)
    assert args["revision"] == "tag"
    assert args["cache_dir"] == Path("cache")
    assert args["output"] == Path("tasks") and args["logs"] == Path("reports")
    assert args["preparation"].sampling_seed == 7
    assert args["preparation"].initialization_seed == 8


@pytest.mark.parametrize(
    "arguments,selected,all_samples",
    [
        ([], cli.DEFAULT_SAMPLES, False),
        (["--sample-id", "Chair/reidar"], ["Chair/reidar"], False),
        (["--all"], None, True),
    ],
)
def test_legacy_ikea_selection(monkeypatch, arguments, selected, all_samples):
    mock = Mock()
    monkeypatch.setattr(cli, "convert_dataset", mock)
    assert cli.main(["convert-ikea", *arguments]) == 0
    assert mock.call_args.args == ("ikea-manual",)
    assert mock.call_args.kwargs["sample_ids"] == selected
    assert mock.call_args.kwargs["all_samples"] == all_samples


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"sample_ids": []},
        {"sample_ids": "x"},
        {"sample_ids": [""]},
        {"limit": 0},
        {"limit": True},
        {"limit": 1.5},
        {"all_samples": 1},
        {"limit": 1, "all_samples": True},
        {"sample_ids": ["x"], "limit": 1},
    ],
)
def test_api_requires_one_valid_selection_without_creating_logs(monkeypatch, tmp_path, selection):
    mock = Mock()
    monkeypatch.setattr(conversion, "load_samples", mock)
    with pytest.raises(ValueError):
        conversion.convert_dataset("fantastic-breaks", logs=tmp_path / "logs", **selection)
    mock.assert_not_called()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    "selection", [{"sample_ids": ["00/00002"]}, {"limit": 1}, {"all_samples": True}]
)
def test_workflow_progress_reuse_and_pinned_configuration(
    monkeypatch, fantastic_row, tmp_path, selection
):
    pytest.importorskip("mujoco")
    source = adapt_sample("fantastic-breaks", fantastic_row, revision="a" * 40)
    calls = []
    closed = []

    def load(dataset, **kwargs):
        calls.append((dataset, kwargs))
        try:
            yield source
            # Progress must already be durable before the iterator advances.
            log = sorted((tmp_path / "logs").iterdir())[-1]
            assert json.loads((log / "metrics.json").read_text())["converted_count"] == 1
        finally:
            closed.append(True)

    monkeypatch.setattr(conversion, "load_samples", load)
    first = conversion.convert_dataset(
        "fantastic-breaks", output=tmp_path / "data", logs=tmp_path / "logs", **selection
    )
    directory = Path(first["config_directories"][0])
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    second = conversion.convert_dataset(
        "fantastic-breaks", output=tmp_path / "data", logs=tmp_path / "logs", **selection
    )
    assert first["converted_count"] == second["converted_count"] == 1
    assert first["config_directories"] == second["config_directories"]
    assert first["log_directory"] != second["log_directory"]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert len(before) == 2 and len(closed) == 2
    assert all(kwargs["streaming"] is False for _, kwargs in calls)
    assert all(kwargs["limit"] == selection.get("limit") for _, kwargs in calls)
    assert set(read_config(directory)["samples"]) == {"00/00002"}
    meta = json.loads((Path(first["log_directory"]) / "meta.json").read_text())
    assert meta["status"] == "passed" and meta["resolved_revision"] == "a" * 40
    assert meta["configurations"][str(directory)]["revision"] == "a" * 40


@pytest.mark.parametrize("model_format", ["xml", "mjb"])
def test_partial_failure_preserves_success_and_closes_reader(
    monkeypatch, fantastic_row, tmp_path, capsys, model_format
):
    pytest.importorskip("mujoco")
    good = adapt_sample("fantastic-breaks", fantastic_row, revision="a" * 40)
    bad = replace(good, sample_id="00/bad", parts=())
    closed = []

    def load(*args, **kwargs):
        try:
            yield good
            yield bad
            raise AssertionError("Conversion must stop at the first failed sample")
        finally:
            closed.append(True)

    monkeypatch.setattr(conversion, "load_samples", load)
    assert (
        cli.main(
            [
                "convert",
                "fantastic-breaks",
                "--model-format",
                model_format,
                "--all",
                "--output",
                str(tmp_path / "data"),
                "--logs",
                str(tmp_path / "logs"),
            ]
        )
        == 1
    )
    assert closed == [True]
    log = next((tmp_path / "logs").iterdir())
    meta = json.loads((log / "meta.json").read_text())
    metrics = json.loads((log / "metrics.json").read_text())
    assert meta["status"] == "failed"
    assert metrics["converted_count"] == 1
    assert metrics["failure"]["sample_id"] == "00/bad"
    assert "ValueError" in metrics["failure"]["error"]
    assert len(list((tmp_path / "data").rglob("*.episode.zip"))) == 1
    assert "Log: " in capsys.readouterr().out


def test_loader_failure_does_not_blame_previous_sample(monkeypatch, fantastic_row, tmp_path):
    pytest.importorskip("mujoco")
    good = adapt_sample("fantastic-breaks", fantastic_row, revision="a" * 40)

    def load(*args, **kwargs):
        yield good
        raise ValueError("AssemblyWorld/fantastic-breaks/00/bad: Invalid polygon")

    monkeypatch.setattr(conversion, "load_samples", load)
    with pytest.raises(ValueError, match="00/bad"):
        conversion.convert_dataset(
            "fantastic-breaks", all_samples=True, output=tmp_path / "data", logs=tmp_path / "logs"
        )
    log = next((tmp_path / "logs").iterdir())
    failure = json.loads((log / "metrics.json").read_text())["failure"]
    assert failure["sample_id"] is None and "00/bad" in failure["error"]
