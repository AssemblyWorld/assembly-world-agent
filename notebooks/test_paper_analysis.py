"""Checks for the analysis choices that can change scientific conclusions."""

import numpy as np
import pandas as pd
import paper_analysis as analysis
from paper_geometry import sample_checkpoints


def test_source_weighting_is_not_evaluation_pooling():
    rows = []
    for b in analysis.BLOCKS:
        for i in range(20):
            rows.append(dict(block=b, sample_id=str(i), SR=int(b == "fantastic-breaks-none")))
    frame = pd.DataFrame(rows)
    assert analysis.overall(frame, "SR") == 0.25
    assert frame.SR.mean() == 0.20


def test_partnet_condition_pairing_preserves_exact_cancellation():
    rows = []
    for b in analysis.BLOCKS:
        for i in range(20):
            value = (
                i % 2 if b == "partnet-none" else 1 - i % 2 if b == "partnet-final-image" else 0.5
            )
            rows.append(dict(block=b, sample_id=str(i), PA=value))
    values = analysis.shape_values(pd.DataFrame(rows), "PA")
    assert np.all(values == 0.5)
    assert analysis.bootstrap(values) == [0.5, 0.5]


def test_budget_selects_last_actual_state_and_persists_after_termination():
    row = {"result": {"started_at": "2026-09-21T00:00:00Z"}}
    calls = [
        {"timestamp": "2026-09-21T00:00:30Z", "state_index": 2},
        {"timestamp": "2026-09-21T00:01:40Z", "state_index": 5},
        {"timestamp": "2026-09-21T00:05:01Z", "state_index": 7},
    ]
    checkpoints, _ = sample_checkpoints(row, {0: None}, calls)
    budgets = {t: si for kind, t, si in checkpoints if kind == "budget"}
    assert budgets[1] == 2
    assert budgets[2] == budgets[5] == 5
    assert budgets[10] == budgets[60] == 7


def test_no_calls_retains_initial_state_at_every_budget():
    checkpoints, fractions = sample_checkpoints({}, {0: None}, [])
    assert not len(fractions)
    assert all(si == 0 for _, _, si in checkpoints)


def test_budget_uses_completion_not_request_time():
    calls = [
        {
            "timestamp": "2026-09-21T00:00:59Z",
            "completed_at": "2026-09-21T00:01:01Z",
            "state_index": 3,
        }
    ]
    checkpoints, _ = sample_checkpoints(
        {"result": {"started_at": "2026-09-21T00:00:00Z"}}, {0: None}, calls
    )
    budgets = {t: si for kind, t, si in checkpoints if kind == "budget"}
    assert budgets[1] == 0
    assert budgets[2] == 3


def test_holm_adjusts_the_whole_family_and_preserves_original_order():
    assert np.allclose(analysis.holm_adjust([0.04, 0.001, 0.01, 0.9]), [0.08, 0.004, 0.03, 0.9])
    assert analysis.paired_randomization(np.zeros((4, 20)), draws=1000) == 1


def test_exported_results_are_unchanged():
    before = analysis.read_json(analysis.CACHE / "results_file_snapshot.json")
    after = {
        str(p.relative_to(analysis.RESULTS)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in analysis.RESULTS.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_all_source_weighted_headlines_match_official_exports():
    frame = analysis.load_benchmark()
    for system in analysis.SYSTEMS:
        official = analysis.read_json(
            analysis.RESULTS / "assemblyworldbench" / system / "evaluation/benchmark_summary.json"
        )
        assert np.isclose(
            analysis.overall(frame[frame.system == system], "SR"),
            official["overall_SR"],
            atol=1e-12,
        )
