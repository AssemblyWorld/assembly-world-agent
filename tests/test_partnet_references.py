"""PartNet reference selection follows published page order, not filenames."""

from copy import deepcopy

import pytest

from assembly_world_agent.adapters import reference_pages


def test_partnet_reference_modes_preserve_source():
    pages = [
        {"source_file": "step_2.png", "step_id": 2, "image": {"bytes": b"first"}},
        {"source_file": "step_10.png", "step_id": 10, "image": {"bytes": b"last"}},
    ]
    row = {"object_id": "example", "manual_pages": pages}
    original = deepcopy(row)
    assert reference_pages("partnet-manualpa", row, "final-image") == pages[-1:]
    assert reference_pages("partnet-manualpa", row, "manualbook") == pages
    assert reference_pages("partnet-manualpa", row, "none") == []
    assert row == original


def test_partnet_single_page_is_final_reference():
    pages = [{"image": {"bytes": b"only"}}]
    assert reference_pages("partnet-manualpa", {"manual_pages": pages}, "final-image") == pages


@pytest.mark.parametrize("mode", ["final-image", "manualbook"])
@pytest.mark.parametrize("row", [{}, {"manual_pages": []}])
def test_partnet_missing_reference_fails(row, mode):
    with pytest.raises(ValueError, match="missing reference images"):
        reference_pages("partnet-manualpa", row, mode)


def test_partnet_none_requires_no_manual():
    assert reference_pages("partnet-manualpa", {}, "none") == []
