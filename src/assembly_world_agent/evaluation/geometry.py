"""Manual-PA-style scores and constrained Hungarian matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.registration import (  # Re-export for existing inspection clients.
    IMPROVEMENT_TOLERANCE,
    MAX_ITERATIONS,
    align,
    chamfer,
    transform,
)

__all__ = [
    "IMPROVEMENT_TOLERANCE",
    "MAX_ITERATIONS",
    "align",
    "chamfer",
    "transform",
    "THRESHOLD",
    "metrics_from_errors",
    "score_parts",
]

THRESHOLD = 0.01


def metrics_from_errors(errors, shape_chamfer):
    errors = np.asarray(errors, dtype=float)
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(shape_chamfer):
        raise ValueError("Metrics require finite, nonempty errors")
    correct = errors <= THRESHOLD
    return dict(SCD=float(shape_chamfer * 1000), PA=float(correct.mean()), SR=int(correct.all()))


def score_parts(prediction, target, groups, part_ids, shape_chamfer):
    """Match only within supplied equivalence groups, without clipping costs."""
    if sorted(i for group in groups for i in group) != list(range(len(part_ids))):
        raise ValueError("Equivalence groups must partition the parts")
    records = []
    for group in groups:
        costs = np.array([[chamfer(prediction[i], target[j]) for j in group] for i in group])
        rows, cols = linear_sum_assignment(costs)
        for a, b in zip(rows, cols):
            records.append(
                dict(
                    part_id=part_ids[group[a]],
                    target_part_id=part_ids[group[b]],
                    chamfer=float(costs[a, b]),
                    correct=bool(costs[a, b] <= THRESHOLD),
                )
            )
    records.sort(key=lambda record: record["part_id"])
    return metrics_from_errors([r["chamfer"] for r in records], shape_chamfer), records
