"""Bounded live-HF smoke test and visual inspection, not an artifact export API."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assembly_world_agent import load_samples, prepare_sample
from assembly_world_agent.adapters import ADAPTERS
from assembly_world_agent.artifacts import experiment
from assembly_world_agent.utils import apply_pose, separated, transform_points


def main():
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(ADAPTERS))
    parser.add_argument("--limit", type=int, default=1, help="Samples per dataset (default: 1)")
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    with experiment(
        "source-inspection", logs=args.logs, inputs=dict(datasets=args.dataset, limit=args.limit)
    ) as (directory, meta, metrics):
        metrics["samples"] = []
        inspect(args, directory, metrics["samples"])
    print(directory)


def inspect(args, directory, records):
    (directory / "screenshots").mkdir()
    datasets = args.dataset or list(ADAPTERS)
    figure = plt.figure(figsize=(12, 4.2 * len(datasets) * args.limit))
    for dataset in datasets:
        for source in load_samples(dataset, limit=args.limit, cache_dir=args.cache_dir):
            task = prepare_sample(source)
            assembled, initial, boxes, error = [], [], [], 0.0
            for raw, part in zip(source.parts, task.parts):
                gt = apply_pose(part.mesh.vertices, part.gt_pose)
                actual = apply_pose(part.mesh.vertices, part.initial_pose)
                restored = transform_points(gt, task.world_to_source)
                expected = apply_pose(raw.mesh.vertices, raw.assembled_pose)
                error = max(error, float(np.max(np.abs(restored - expected))))
                np.testing.assert_allclose(restored, expected, atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(actual[:, 2].min(), 0, atol=1e-12)
                assert part.points.shape == (task.config.fps_points, 3)
                assert np.isfinite(part.points).all()
                boxes.append(np.array([actual[:, :2].min(0), actual[:, :2].max(0)]))
                assembled.append(gt)
                initial.append(actual)
            for i, box in enumerate(boxes):
                assert all(
                    separated(box, other, task.config.min_gap - 1e-12) for other in boxes[:i]
                )
            gt_vertices = np.concatenate(assembled)
            diagonal = float(np.linalg.norm(np.ptp(gt_vertices, axis=0)))
            np.testing.assert_allclose(diagonal, 1, atol=1e-12)
            np.testing.assert_allclose(gt_vertices[:, 2].min(), 0, atol=1e-12)
            np.testing.assert_allclose((gt_vertices.max(0) + gt_vertices.min(0))[:2], 0, atol=1e-12)
            digest = hashlib.sha256(b"".join(p.points.tobytes() for p in task.parts)).hexdigest()
            record = dict(
                dataset=task.dataset,
                revision=task.revision,
                sample_id=task.sample_id,
                parts=len(task.parts),
                protocol=task.protocol_version,
                config=asdict(task.config),
                bbox_diagonal=diagonal,
                max_source_reconstruction_error=error,
                point_cloud_sha256=digest,
                grounded=True,
                separated=True,
            )
            for column, (kind, clouds) in enumerate(
                (("GT assembly", assembled), ("Initial layout", initial))
            ):
                ax = figure.add_subplot(
                    len(datasets) * args.limit, 2, 2 * len(records) + column + 1, projection="3d"
                )
                for index, part in enumerate(task.parts):
                    pose = part.gt_pose if column == 0 else part.initial_pose
                    points = apply_pose(part.points, pose)
                    ax.scatter(
                        *points.T, s=1.4, alpha=0.85, color=plt.get_cmap("tab20")(index % 20)
                    )
                vertices = np.concatenate(clouds)
                low, high = vertices.min(0), vertices.max(0)
                center = (low + high) / 2
                radius = max(high - low) * 0.55
                ax.set(
                    xlim=(center[0] - radius, center[0] + radius),
                    ylim=(center[1] - radius, center[1] + radius),
                    zlim=(center[2] - radius, center[2] + radius),
                    xlabel="X",
                    ylabel="Y",
                    zlabel="Z",
                )
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=24, azim=-55)
                ax.set_title(f"{task.dataset.split('/')[-1]}\n{task.sample_id}\n{kind}", fontsize=8)
            records.append(record)
            print(json.dumps(record), flush=True)
    figure.tight_layout()
    figure.savefig(directory / "screenshots/samples.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
