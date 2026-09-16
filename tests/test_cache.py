"""Per-sample caches next to prepared configurations: reference images and evaluation inputs."""

import io
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from assembly_world_agent import adapt_sample, export_episode, prepare_sample
from assembly_world_agent.artifacts import cache_directory, config_id, sample_name, write_json
from assembly_world_agent.episode_io import read_episode
from assembly_world_agent.episodes import sha256
from assembly_world_agent.evaluation import cache as evaluation_cache
from assembly_world_agent.evaluation import runner as evaluation_runner
from assembly_world_agent.evaluation.inputs import final_poses, final_poses_against_initial
from assembly_world_agent.experiments import runner
from assembly_world_agent.similarity import SimilarityConfig

REVISION = "a" * 40


def test_cache_directory_requires_configuration(tmp_path):
    episode = tmp_path / "Test--a.episode.zip"
    episode.write_bytes(b"zip")
    assert cache_directory(None) is None
    assert cache_directory(episode) is None
    (tmp_path / "config.json").write_text("{}")
    assert cache_directory(episode) == tmp_path / "cache" / "Test--a"
    assert cache_directory(tmp_path / "other.zip") is None


# --- reference images ---------------------------------------------------------------


def _png(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG")
    return stream.getvalue()


def _bench_dataset(pages):
    import datasets

    row = {
        "object_id": "test",
        "steps": [{"step_id": i} for i in range(len(pages))],
        "manual_pages": [
            {
                "step_id": i,
                "view_id": "000",
                "kind": "diagram",
                "source_file": f"diagram/000_{i}.png",
                "image": {"bytes": payload, "path": None},
            }
            for i, payload in enumerate(pages)
        ],
    }
    data = datasets.Dataset.from_list([row])
    features = data.features.copy()
    features["manual_pages"].feature["image"] = datasets.Image(decode=False)
    return data.cast(features)


@pytest.mark.parametrize("mode", ["final-image", "manualbook"])
def test_reference_cache_fills_then_replaces_hf(tmp_path, monkeypatch, mode):
    import datasets

    config_dir = tmp_path / "data" / "assemblybench" / "prep"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text("{}")
    initial = config_dir / "test.episode.zip"
    initial.write_bytes(b"zip")
    pages = [_png("red"), _png("blue")]
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs.get("revision"))
        return _bench_dataset(pages)

    monkeypatch.setattr(datasets, "load_dataset", load)
    meta = {
        "options": {"reference_mode": mode},
        "config": {"identity": {"dataset": "AssemblyWorld/assemblybench", "revision": "a" * 40}},
    }
    first_inputs = {"sample_id": "test", "initial_path": str(initial)}
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = runner.manual_pages(first_inputs, meta, first_root)
    assert calls == ["a" * 40]
    cache = config_dir / "cache" / "test" / "reference" / mode
    table = json.loads((cache / "pages.json").read_text())
    assert table["dataset"] == "AssemblyWorld/assemblybench" and table["reference_mode"] == mode
    assert [p["file"] for p in table["pages"]] == [
        f"page-{i:04d}.png" for i in range(1, len(first) + 1)
    ]
    for page, path in zip(table["pages"], first):
        assert (cache / page["file"]).read_bytes() == open(path, "rb").read()
    stripped = {
        **table,
        "pages": [{k: v for k, v in p.items() if k != "file"} for p in table["pages"]],
    }
    assert stripped == first_inputs["manual"]

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: pytest.fail("HF accessed"))
    second_inputs = {"sample_id": "test", "initial_path": str(initial)}
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = runner.manual_pages(second_inputs, meta, second_root)
    assert [open(p, "rb").read() for p in second] == [open(p, "rb").read() for p in first]
    assert second_inputs["manual"] == first_inputs["manual"]
    assert json.dumps(second_inputs["manual"]) == json.dumps(first_inputs["manual"])

    # An edited page is detected; a standalone episode never consults the cache.
    (cache / "page-0001.png").write_bytes(_png("green"))
    with pytest.raises(ValueError, match="differs from its checksum"):
        runner.manual_pages({"sample_id": "test", "initial_path": str(initial)}, meta, tmp_path)
    with pytest.raises(BaseException, match="HF accessed"):
        runner.manual_pages({"sample_id": "test"}, meta, tmp_path)


def test_reference_cache_rejects_other_provenance(tmp_path):
    (tmp_path / "pages.json").write_text(
        json.dumps({"dataset": "x", "revision": "y", "reference_mode": "manualbook", "pages": []})
    )
    with pytest.raises(ValueError, match="provenance differs"):
        runner.read_reference_cache(tmp_path, dataset="x", revision="z", mode="manualbook")
    with pytest.raises(ValueError, match="no pages"):
        runner.read_reference_cache(tmp_path, dataset="x", revision="y", mode="manualbook")
    assert (
        runner.read_reference_cache(tmp_path / "missing", dataset="x", revision="y", mode="m")
        is None
    )


# --- evaluation inputs ---------------------------------------------------------------


def _identity():
    return dict(
        dataset="AssemblyWorld/ikea-manual",
        revision=REVISION,
        protocol="assembly-preparation-v1",
        preparation=dict(
            surface_points=4096,
            fps_points=1000,
            sampling_seed=0,
            initialization_seed=0,
            min_gap=0.02,
        ),
        producer={"backend": "native"},
        contract="3dwebagent-runtime-1",
        engine="3.12.0",
    )


def _configured_sample(row, tmp_path):
    """A prepared configuration directory with one initial episode and a run directory."""
    pytest.importorskip("mujoco")
    sample = prepare_sample(adapt_sample("ikea-manual", row, revision=REVISION))
    config_dir = tmp_path / "data" / "ikea-manual" / "prep"
    config_dir.mkdir(parents=True)
    initial = config_dir / (sample_name(sample.sample_id) + ".episode.zip")
    export_episode(sample, initial)
    episode = read_episode(initial)
    identity = _identity()
    identity["producer"] = episode["manifest"]["producer"]
    identity["contract"] = episode["manifest"]["contract"]
    expected = dict(
        episode=initial.name,
        episode_id=episode["manifest"]["id"],
        sha256=sha256(initial.read_bytes()),
        parts=len(sample.parts),
    )
    write_json(
        config_dir / "config.json",
        dict(
            version=1,
            config_id=config_id(identity),
            identity=identity,
            samples={sample.sample_id: expected},
        ),
    )
    run = tmp_path / "run"
    directory = run / "samples" / sample_name(sample.sample_id)
    directory.mkdir(parents=True)
    write_json(
        directory / "input.json",
        dict(
            sample_id=sample.sample_id,
            revision=REVISION,
            initial_path=str(initial),
            **expected,
        ),
    )
    (directory / "final.episode.zip").write_bytes(initial.read_bytes())
    write_json(
        run / "run.json",
        dict(
            options={},
            config=dict(
                version=1,
                config_id=config_id(identity),
                identity=identity,
                samples={sample.sample_id: expected},
            ),
        ),
    )
    return sample, initial, directory, expected, identity


def test_final_poses_against_initial_matches_source_validation(row, tmp_path):
    sample, initial, directory, expected, identity = _configured_sample(row, tmp_path)
    episode = read_episode(directory / "final.episode.zip")
    reference = read_episode(initial)
    ids = sorted(p.part_id for p in sample.parts)
    poses, index = final_poses_against_initial(episode, reference, ids)
    source_poses, source_index = final_poses(episode, sample)
    assert index == source_index
    for (r, t), (sr, st) in zip(poses, source_poses):
        assert np.allclose(r, sr) and np.allclose(t, st)
    tampered = deepcopy(episode)
    name = next(k for k in tampered["files"] if k.endswith(".obj"))
    tampered["files"][name] = tampered["files"][name].replace(b"v ", b"v  ", 1)
    with pytest.raises(ValueError, match="world files differ"):
        final_poses_against_initial(tampered, reference, ids)
    with pytest.raises(ValueError, match="object catalog"):
        final_poses_against_initial(episode, reference, ids[:-1] + ["zz"])


def test_evaluation_cache_roundtrip_hit_and_key_mismatch(row, tmp_path, monkeypatch):
    sample, initial, directory, expected, identity = _configured_sample(row, tmp_path)
    similarity = SimilarityConfig("geometry", 1e-4)
    source = adapt_sample("ikea-manual", row, revision=REVISION)
    path = evaluation_cache.evaluation_cache_path(initial)
    assert path == initial.parent / "cache" / sample_name(sample.sample_id) / "evaluation.json"
    key = evaluation_cache.cache_key(
        sample.sample_id,
        identity,
        expected,
        evaluation_protocol=evaluation_runner.PROTOCOL["version"],
        similarity=similarity,
    )
    assert evaluation_cache.read_cache(path, key) is None

    fresh = evaluation_runner.evaluate_sample(
        directory, expected, identity, source, similarity, initial
    )
    assert fresh["status"] == "scored" and fresh["cached_inputs"] is False
    entry = evaluation_cache.read_cache(path, key)
    assert entry is not None and entry["key"] == key
    assert entry["part_ids"] == sorted(p.part_id for p in sample.parts)
    values = evaluation_cache.cached_inputs(entry)
    for pid, cloud in zip(values["part_ids"], values["points"]):
        part = next(p for p in sample.parts if p.part_id == pid)
        assert np.array_equal(cloud, part.points)

    monkeypatch.setattr(
        evaluation_runner, "prepare_sample", lambda *a, **k: pytest.fail("Source prepared")
    )
    ref = evaluation_runner.SampleRef(sample.sample_id, identity["dataset"], REVISION)
    cached = evaluation_runner.evaluate_sample(
        directory, expected, identity, ref, similarity, initial
    )
    assert cached["status"] == "scored" and cached["cached_inputs"] is True
    for field in ("SR", "PA", "SCD", "equivalence_groups", "scale_divisor", "state_index"):
        assert cached[field] == fresh[field]
    assert [p["correct"] for p in cached["parts"]] == [p["correct"] for p in fresh["parts"]]

    # Another similarity policy is a miss that leaves the stored entry untouched.
    other = evaluation_cache.cache_key(
        sample.sample_id,
        identity,
        expected,
        evaluation_protocol=evaluation_runner.PROTOCOL["version"],
        similarity=SimilarityConfig("source"),
    )
    assert evaluation_cache.read_cache(path, other) is None
    before = path.read_bytes()
    assert evaluation_cache.write_cache(path, {"version": 1}) is False
    assert path.read_bytes() == before
    # Any other difference is an error, never a recomputation.
    with pytest.raises(evaluation_cache.CacheKeyMismatch):
        evaluation_cache.read_cache(path, {**key, "initial_sha256": "0" * 64})
    with pytest.raises(evaluation_cache.CacheKeyMismatch):
        evaluation_cache.read_cache(path, {**key, "evaluation_protocol": "other"})


def test_score_locations_uses_cache_without_source_scan(row, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    sample, initial, directory, expected, identity = _configured_sample(row, tmp_path)
    monkeypatch.setattr(evaluation_runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    source = adapt_sample("ikea-manual", row, revision=REVISION)
    run = tmp_path / "run"
    config = json.loads((run / "run.json").read_text())["config"]
    initials = evaluation_runner.initial_locations(run, config)
    assert initials == {sample.sample_id: initial.resolve()}

    monkeypatch.setattr(evaluation_runner, "load_samples", lambda *a, **k: iter([source]))
    first = evaluation_runner.score_locations(
        {sample.sample_id: directory},
        {sample.sample_id: expected},
        identity,
        workers=1,
        initials=initials,
    )
    assert first[0]["status"] == "scored" and first[0]["cached_inputs"] is False

    monkeypatch.setattr(
        evaluation_runner, "load_samples", lambda *a, **k: pytest.fail("Source scan on cache hit")
    )
    second = evaluation_runner.score_locations(
        {sample.sample_id: directory},
        {sample.sample_id: expected},
        identity,
        workers=1,
        initials=initials,
    )
    assert second[0]["status"] == "scored" and second[0]["cached_inputs"] is True
    assert second[0]["SR"] == first[0]["SR"] and second[0]["PA"] == first[0]["PA"]
    assert [p["correct"] for p in second[0]["parts"]] == [p["correct"] for p in first[0]["parts"]]

    # evaluate_run wires the same lookup from the run's input.json.
    summary = evaluation_runner.evaluate_run(run, workers=1)
    assert summary["status"] == "complete" and summary["scored_samples"] == 1

    # A cache entry with a foreign key is a per-sample error, not a rewrite.
    path = evaluation_cache.evaluation_cache_path(initial)
    entry = json.loads(path.read_text())
    entry["key"]["initial_sha256"] = "0" * 64
    path.write_text(json.dumps(entry))
    rows = evaluation_runner.score_locations(
        {sample.sample_id: directory},
        {sample.sample_id: expected},
        identity,
        workers=1,
        initials=initials,
    )
    assert rows[0]["status"] == "error" and "cache key differs" in rows[0]["error"]
    assert json.loads(path.read_text())["key"]["initial_sha256"] == "0" * 64
