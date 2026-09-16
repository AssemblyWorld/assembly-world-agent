import json

import pytest

from assembly_world_agent import benchmark
from assembly_world_agent.artifacts import config_id, sample_name, write_json
from assembly_world_agent.episodes import sha256

REVISION = "c" * 40
TASKS = {"none": "Assemble without references.\n", "final-image": "Match the image.\n"}


def _identity(repo_id):
    return dict(
        dataset=repo_id,
        revision=REVISION,
        protocol="assembly-preparation-v1",
        preparation={"sampling_seed": 0, "initialization_seed": 0},
    )


def _block(name, source, mode, samples):
    identity = _identity(f"AssemblyWorld/{source}")
    return dict(
        name=name,
        source=source,
        dataset=source,
        repo_id=identity["dataset"],
        revision=REVISION,
        config_id=config_id(identity),
        data=name,
        reference_mode=mode,
        prompt_file=f"{name}/task.txt",
        prompt_sha256=sha256(TASKS[mode].encode()),
        samples=samples,
    )


@pytest.fixture
def benchmark_dir(tmp_path):
    pn = {
        sid: dict(category=c, parts=p, band=b, sha256=f"pn{sid}")
        for sid, c, p, b in [("1", "chair", 3, "low"), ("2", "table", 9, "high")]
    }
    ikea = {"Bench/a": dict(category="Bench", parts=4, band="low", sha256="ik")}
    ab = {"7": dict(category=None, parts=5, band="mid", sha256="ab")}
    fb = {"00/1": dict(category="00", parts=2, band=None, sha256="fb")}
    value = dict(
        version=1,
        benchmark="demo",
        sources=["partnet-manualpa", "ikea-manual", "assemblybench", "fantastic-breaks"],
        blocks=[
            _block("partnet-none", "partnet-manualpa", "none", pn),
            _block("partnet-final-image", "partnet-manualpa", "final-image", pn),
            _block("ikea-final-image", "ikea-manual", "final-image", ikea),
            _block("assemblybench-final-image", "assemblybench", "final-image", ab),
            _block("fantastic-breaks-none", "fantastic-breaks", "none", fb),
        ],
        evaluation=dict(
            protocol="assembly-evaluation-v2",
            similarity_policy="geometry",
            similarity_threshold=1e-4,
        ),
        aggregation=dict(rule="two-level", missing_sample_rule="SR=0, PA=0"),
    )
    for block in value["blocks"]:
        path = tmp_path / block["prompt_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TASKS[block["reference_mode"]])
    write_json(tmp_path / "benchmark.json", value)
    return tmp_path


def _write_run(directory, block, sample_ids, *, task=None, outcomes=None, sha=None):
    directory.mkdir(parents=True)
    identity = _identity(block["repo_id"])
    samples = {
        sid: dict(
            episode=sample_name(sid) + ".episode.zip",
            sha256=(sha or {}).get(sid, block["samples"][sid]["sha256"]),
            parts=block["samples"][sid]["parts"],
        )
        for sid in sample_ids
    }
    write_json(
        directory / "run.json",
        dict(
            options=dict(reference_mode=block["reference_mode"]),
            config=dict(
                version=1, config_id=config_id(identity), identity=identity, samples=samples
            ),
            samples=list(sample_ids),
            task=TASKS[block["reference_mode"]] if task is None else task,
        ),
    )
    for sid in sample_ids:
        write_json(
            directory / "samples" / sample_name(sid) / "result.json",
            dict(agent_outcome={"status": (outcomes or {}).get(sid, "completed")}),
        )
    return directory


def test_read_benchmark_validates_structure(benchmark_dir):
    value = benchmark.read_benchmark(benchmark_dir / "benchmark.json")
    assert [b["name"] for b in value["blocks"]][0] == "partnet-none"
    assert value["_directory"] == str(benchmark_dir)
    raw = json.loads((benchmark_dir / "benchmark.json").read_text())
    (benchmark_dir / "partnet-none" / "task.txt").write_text("edited")
    with pytest.raises(ValueError, match="Task text differs"):
        benchmark.read_benchmark(benchmark_dir / "benchmark.json")
    (benchmark_dir / "partnet-none" / "task.txt").write_text(TASKS["none"])
    broken = dict(raw, blocks=raw["blocks"] + [dict(raw["blocks"][0], name="copy")])
    write_json(benchmark_dir / "broken.json", broken)
    with pytest.raises(ValueError, match="share dataset"):
        benchmark.read_benchmark(benchmark_dir / "broken.json")
    write_json(benchmark_dir / "version.json", dict(raw, version=2))
    with pytest.raises(ValueError, match="version"):
        benchmark.read_benchmark(benchmark_dir / "version.json")


def test_block_for_run_matches_by_identity_and_task(benchmark_dir, tmp_path):
    value = benchmark.read_benchmark(benchmark_dir / "benchmark.json")
    blocks = {b["name"]: b for b in value["blocks"]}
    none = _write_run(tmp_path / "run-none", blocks["partnet-none"], ["1"])
    image = _write_run(tmp_path / "run-image", blocks["partnet-final-image"], ["1", "2"])
    assert benchmark.block_for_run(value, none)["name"] == "partnet-none"
    assert benchmark.block_for_run(value, image)["name"] == "partnet-final-image"
    grouped = benchmark.match_runs_to_blocks(value, [none, image])
    assert {k: [r.name for r in v] for k, v in grouped.items()} == {
        "partnet-none": ["run-none"],
        "partnet-final-image": ["run-image"],
    }
    other = _write_run(tmp_path / "run-task", blocks["partnet-none"], ["1"], task="different")
    with pytest.raises(ValueError, match="task text differs"):
        benchmark.block_for_run(value, other)
    foreign = _write_run(tmp_path / "run-sha", blocks["partnet-none"], ["1"], sha={"1": "zz"})
    with pytest.raises(ValueError, match="initial episode differs"):
        benchmark.block_for_run(value, foreign)
    stray = _write_run(
        tmp_path / "run-stray",
        dict(
            blocks["partnet-none"],
            samples={**blocks["partnet-none"]["samples"], "9": dict(sha256="q", parts=1)},
        ),
        ["9"],
    )
    with pytest.raises(ValueError, match="outside block"):
        benchmark.block_for_run(value, stray)
    assert benchmark.agent_outcomes([none, image], {"1", "2"}) == {"completed": 3}


def _rows(block, values):
    return {
        sid: dict(
            sample_id=sid,
            status="scored",
            protocol_version="assembly-evaluation-v2",
            dataset=block["repo_id"],
            revision=block["revision"],
            SR=sr,
            PA=float(sr),
            SCD=0.0,
            equivalence_groups=[[pid] for pid in range(block["samples"][sid]["parts"])],
        )
        for sid, sr in values.items()
    }


def test_summarize_two_level_overall_and_missing_rows(benchmark_dir):
    value = benchmark.read_benchmark(benchmark_dir / "benchmark.json")
    blocks = {b["name"]: b for b in value["blocks"]}
    rows = {
        "partnet-none": _rows(blocks["partnet-none"], {"1": 1, "2": 1}),
        "partnet-final-image": _rows(blocks["partnet-final-image"], {"1": 0, "2": 0}),
        "ikea-final-image": _rows(blocks["ikea-final-image"], {"Bench/a": 1}),
        "assemblybench-final-image": _rows(blocks["assemblybench-final-image"], {"7": 1}),
        "fantastic-breaks-none": _rows(blocks["fantastic-breaks-none"], {"00/1": 1}),
    }
    summary = benchmark.summarize_benchmark(value, rows)
    assert summary["status"] == "complete"
    assert summary["sources"] == {
        "partnet-manualpa": 0.5,
        "ikea-manual": 1.0,
        "assemblybench": 1.0,
        "fantastic-breaks": 1.0,
    }
    assert summary["overall_SR"] == pytest.approx(0.875)
    assert summary["blocks"]["partnet-none"]["by_category"] == {"chair": 1.0, "table": 1.0}
    assert summary["blocks"]["partnet-none"]["SR_wilson95"][0] > 0.3

    # A missing row is SR=0 and PA=0; SCD averages scored rows only.
    partial = dict(rows, **{"ikea-final-image": {}})
    summary = benchmark.summarize_benchmark(value, partial)
    assert summary["status"] == "incomplete"
    ikea = summary["blocks"]["ikea-final-image"]
    assert ikea["SR"] == 0.0 and ikea["PA"] == 0.0 and ikea["SCD"] is None
    assert ikea["errors"] == [{"sample_id": "Bench/a", "error": "missing row"}]
    assert summary["sources"]["ikea-manual"] == 0.0
    assert summary["overall_SR"] == pytest.approx(0.625)

    # Blocks without any run are reported but excluded from Overall.
    summary = benchmark.summarize_benchmark(
        value, {"fantastic-breaks-none": rows["fantastic-breaks-none"]}
    )
    assert summary["blocks"]["partnet-none"]["evaluated"] is False
    assert summary["overall_SR"] == 1.0 and summary["status"] == "incomplete"
    text = benchmark.format_summary(summary)
    assert "Overall" in text and "(no run)" in text and "100.0%" in text

    bad = dict(rows)
    bad["fantastic-breaks-none"]["00/1"]["protocol_version"] = "other"
    with pytest.raises(ValueError, match="protocol"):
        benchmark.summarize_benchmark(value, bad)


def test_evaluate_benchmark_scores_blocks_into_output(benchmark_dir, tmp_path, monkeypatch):
    from assembly_world_agent import evaluation

    value = benchmark.read_benchmark(benchmark_dir / "benchmark.json")
    blocks = {b["name"]: b for b in value["blocks"]}
    first = _write_run(tmp_path / "pn-1", blocks["partnet-none"], ["1"], outcomes={"1": "unable"})
    second = _write_run(tmp_path / "pn-2", blocks["partnet-none"], ["2"])
    fb = _write_run(tmp_path / "fb", blocks["fantastic-breaks-none"], ["00/1"])
    calls = []

    def fake_evaluate_runs(runs, output, **kwargs):
        calls.append(([r.name for r in runs], kwargs["similarity"].policy))
        block = next(b for b in value["blocks"] if b["data"] == output.name)
        ids = [sid for run in runs for sid in json.loads((run / "run.json").read_text())["samples"]]
        output.mkdir(parents=True)
        rows = _rows(block, {sid: 1 for sid in ids})
        (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        return {}

    monkeypatch.setattr(evaluation, "evaluate_runs", fake_evaluate_runs)
    output = tmp_path / "evaluation"
    summary = benchmark.evaluate_benchmark(
        benchmark_dir / "benchmark.json", [first, second, fb], output=output, workers=1
    )
    assert calls == [(["pn-1", "pn-2"], "geometry"), (["fb"], "geometry")]
    assert summary["blocks"]["partnet-none"]["SR"] == 1.0
    assert summary["blocks"]["partnet-none"]["agent_outcomes"] == {"completed": 1, "unable": 1}
    assert summary["blocks"]["ikea-final-image"]["evaluated"] is False
    assert summary["status"] == "incomplete"
    saved = json.loads((output / "benchmark_summary.json").read_text())
    assert saved["overall_SR"] == summary["overall_SR"] == 1.0
    meta = json.loads((output / "meta.json").read_text())
    assert set(meta["runs"]) == {"partnet-none", "fantastic-breaks-none"}
    from assembly_world_agent.similarity import SimilarityConfig

    with pytest.raises(ValueError, match="Similarity configuration differs"):
        benchmark.evaluate_benchmark(
            benchmark_dir / "benchmark.json",
            [fb],
            output=tmp_path / "other",
            similarity=SimilarityConfig("source"),
        )
