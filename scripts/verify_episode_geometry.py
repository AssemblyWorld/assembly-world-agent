"""Compare exported MuJoCo surfaces against freshly prepared, pinned HF samples."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from assembly_world_agent import PreparationConfig, export_episode, prepare_sample
from assembly_world_agent.artifacts import experiment, load_prepared, read_config
from assembly_world_agent.utils import apply_pose, transform_points

# PCA bases and matrix/quaternion round trips can accumulate order-1e-12
# differences in normalized coordinates (AssemblyBench 1613: about 6.2e-12).
# This bound remains 2000 times tighter than compiled float32 surface validation.
GT_TRANSFORM_ATOL = 1e-10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    configuration = read_config(args.config)
    with experiment("geometry-regression", logs=args.logs, inputs=configuration) as (
        directory,
        meta,
        metrics,
    ):
        metrics["samples"] = []
        verify(args, configuration, metrics["samples"])
    print(directory)


def verify(args, configuration, report, *, sample_ids=None):
    for source, sample in load_prepared(
        args.config, cache_dir=args.cache_dir, sample_ids=sample_ids
    ):
        original = [p.mesh.vertices.copy() for p in source.parts]
        task = dict(
            sample_id=sample.sample_id,
            config=configuration["identity"]["preparation"],
            part_ids={
                pid: f"part-{i + 1:04d}"
                for i, pid in enumerate(sorted(p.part_id for p in sample.parts))
            },
            **configuration["samples"][sample.sample_id],
        )
        alternative = prepare_sample(
            source,
            PreparationConfig(
                **{**task["config"], "initialization_seed": sample.config.initialization_seed + 1}
            ),
        )
        for a, b, raw, saved in zip(sample.parts, alternative.parts, source.parts, original):
            np.testing.assert_allclose(
                apply_pose(a.points, a.gt_pose), apply_pose(b.points, b.gt_pose), atol=1e-12
            )
            np.testing.assert_array_equal(raw.mesh.vertices, saved)
        maximum = 0.0
        max_gt_transform_error = 0.0
        with tempfile.TemporaryDirectory(prefix="awa-geometry-") as temporary:
            temp = Path(temporary)
            repeated = export_episode(sample, temp / "repeated.zip")
            assert repeated.path.read_bytes() == (args.config / task["episode"]).read_bytes()
            with zipfile.ZipFile(repeated.path) as archive:
                archive.extractall(temp / "archive")
            model = mujoco.MjModel.from_xml_path(str(temp / "archive/world/model.xml"))
            data = mujoco.MjData(model)
            for pose_name in ("initial_pose", "gt_pose"):
                expected_parts = []
                for part in sample.parts:
                    body = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, task["part_ids"][part.part_id]
                    )
                    address = model.jnt_qposadr[model.body_jntadr[body]]
                    pose = getattr(part, pose_name)
                    data.qpos[address : address + 7] = np.r_[pose.position, pose.quaternion]
                mujoco.mj_forward(model, data)
                for part in sample.parts:
                    body = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, task["part_ids"][part.part_id]
                    )
                    geom = model.body_geomadr[body]
                    mesh = model.geom_dataid[geom]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    actual = (
                        model.mesh_vert[start : start + count]
                        @ data.geom_xmat[geom].reshape(3, 3).T
                        + data.geom_xpos[geom]
                    )
                    expected = apply_pose(part.mesh.vertices, getattr(part, pose_name))
                    error = max(
                        cKDTree(actual).query(expected)[0].max(),
                        cKDTree(expected).query(actual)[0].max(),
                    )
                    # MuJoCo stores render mesh vertices as float32; state tolerances are separate.
                    assert error < 2e-7, (task["sample_id"], part.part_id, error)
                    maximum = max(maximum, float(error))
                    expected_parts.append(expected)
                if pose_name == "gt_pose":
                    vertices = np.concatenate(expected_parts)
                    gt_ground_error = abs(float(vertices[:, 2].min()))
                    np.testing.assert_allclose(
                        gt_ground_error,
                        0,
                        atol=GT_TRANSFORM_ATOL,
                        rtol=0,
                        err_msg=f"{sample.sample_id}: reconstructed GT grounding",
                    )
                    for part, raw in zip(sample.parts, source.parts):
                        assembled = apply_pose(raw.mesh.vertices, raw.assembled_pose)
                        actual = apply_pose(part.mesh.vertices, part.gt_pose)
                        expected = transform_points(assembled, sample.source_to_world)
                        max_gt_transform_error = max(
                            max_gt_transform_error, float(np.max(np.abs(actual - expected)))
                        )
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            atol=GT_TRANSFORM_ATOL,
                            rtol=0,
                            err_msg=f"{sample.sample_id}/{part.part_id}: GT coordinate transform",
                        )
                        np.testing.assert_allclose(
                            transform_points(actual, sample.world_to_source), assembled, atol=1e-9
                        )
        report.append(
            dict(
                sample_id=sample.sample_id,
                revision=sample.revision,
                status="passed",
                deterministic_archive=True,
                initialization_independent_points=True,
                input_unchanged=True,
                shape_only_scale=True,
                gt_grounded=True,
                inverse_transform=True,
                max_compiled_surface_error=maximum,
                compiled_surface_atol=2e-7,
                max_gt_transform_error=max_gt_transform_error,
                gt_transform_atol=GT_TRANSFORM_ATOL,
                gt_ground_error=gt_ground_error,
                gt_ground_atol=GT_TRANSFORM_ATOL,
            )
        )
        print(f"{sample.sample_id}: geometry passed", flush=True)


if __name__ == "__main__":
    main()
