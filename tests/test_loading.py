from unittest.mock import Mock

import pytest

from assembly_world_agent import load_samples, loading
from assembly_world_agent.adapters import get_adapter


def test_pinned_loading_is_bounded_and_does_not_resolve_network(monkeypatch, row):
    def stream():
        yield row
        raise AssertionError("limit must stop before reading another row")

    mock = Mock(return_value=stream())
    monkeypatch.setattr(loading, "load_dataset", mock)
    monkeypatch.setattr(loading, "HfApi", Mock(side_effect=AssertionError("Unnecessary Hub query")))
    result = list(load_samples("ikea-manual", limit=1, cache_dir="cache"))
    expected = get_adapter("ikea-manual").DEFAULT_REVISION
    assert result[0].revision == expected
    assert mock.call_args.kwargs == dict(
        split="full", revision=expected, streaming=True, cache_dir="cache", token=None
    )


def test_branch_resolves_once_and_passes_sha(monkeypatch, row):
    api = Mock()
    api.dataset_info.return_value.sha = "a" * 40
    monkeypatch.setattr(loading, "HfApi", Mock(return_value=api))
    mock = Mock(return_value=[row])
    monkeypatch.setattr(loading, "load_dataset", mock)
    sample = next(load_samples("ikea-manual", revision="main", streaming=False))
    assert sample.revision == "a" * 40
    assert mock.call_args.kwargs["revision"] == sample.revision
    api.dataset_info.assert_called_once()


def test_ids_missing_empty_and_invalid_arguments(monkeypatch, row):
    # IKEA selects object_id, not an unrelated optional annotation.
    monkeypatch.setattr(loading, "load_dataset", Mock(return_value=[row]))
    assert (
        list(load_samples("ikea-manual", sample_ids=[row["object_id"]]))[0].sample_id
        == row["object_id"]
    )
    assert list(load_samples("ikea-manual", sample_ids=[])) == []
    with pytest.raises(ValueError, match="not found"):
        list(load_samples("ikea-manual", sample_ids=["absent"]))
    for kwargs in ({"limit": 0}, {"limit": 1.5}, {"sample_ids": "x"}, {"revision": ""}):
        with pytest.raises(ValueError):
            list(load_samples("ikea-manual", **kwargs))


@pytest.mark.parametrize("stop", ["limit", "close", "error"])
def test_partial_stream_is_closed(monkeypatch, row, stop):
    released = []

    def rows(*args, **kwargs):
        try:
            yield row
            raise AssertionError("Reader advanced beyond the bounded request")
        finally:
            released.append(True)

    if stop == "error":
        row["parts"] = []
    monkeypatch.setattr(loading, "load_dataset", Mock(side_effect=rows))
    samples = load_samples("ikea-manual", limit=1)
    if stop == "limit":
        assert len(list(samples)) == 1
    elif stop == "close":
        next(samples)
        assert released == [True]  # Release before the caller handles the last sample.
        samples.close()
    else:
        with pytest.raises(ValueError):
            next(samples)
    assert released == [True]
