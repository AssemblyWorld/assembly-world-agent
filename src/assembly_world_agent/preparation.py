"""One task protocol shared by every dataset adapter."""

from copy import deepcopy

import numpy as np

from .models import AssemblyPart, AssemblySample, Mesh, PreparationConfig, SourceSample
from .utils import (
    apply_pose,
    assembly_normalization,
    farthest_point_sample,
    make_pose,
    pca_frame,
    place_parts,
    rotation_matrix,
    sample_surface,
    stable_rng,
    transform_points,
)


def prepare_sample(sample: SourceSample, config: PreparationConfig | None = None) -> AssemblySample:
    """Return independent task geometry; never modify the HF/source sample."""
    config = config or PreparationConfig()
    if not sample.parts or len({p.part_id for p in sample.parts}) != len(sample.parts):
        raise ValueError("Expected nonempty, unique source parts")
    assembled = [apply_pose(p.mesh.vertices, p.assembled_pose) for p in sample.parts]
    scale = 2 * max(
        float(np.linalg.norm(p.mesh.vertices - p.mesh.vertices.mean(0), axis=1).max())
        for p in sample.parts
    )
    normalization = assembly_normalization(
        np.concatenate(assembled), sample.source_to_z_up, scale=scale
    )
    meshes, targets, clouds = {}, {}, {}
    for part in sample.parts:
        # Public geometry depends only on each input shape, never its target pose.
        center, basis = pca_frame(part.mesh.vertices)
        local = (part.mesh.vertices - center) @ basis / scale
        normals = part.mesh.normals @ basis
        target_center = transform_points(apply_pose(center, part.assembled_pose), normalization)
        target_basis = (
            sample.source_to_z_up @ rotation_matrix(part.assembled_pose.quaternion) @ basis
        )
        mesh = Mesh(local, part.mesh.faces, normals, part.mesh.face_normal_indices)
        try:
            candidates = sample_surface(
                mesh,
                config.surface_points,
                stable_rng(
                    config.sampling_seed, sample.dataset, sample.sample_id, part.part_id, "surface"
                ),
            )
            selected = farthest_point_sample(
                candidates,
                config.fps_points,
                stable_rng(
                    config.sampling_seed, sample.dataset, sample.sample_id, part.part_id, "fps"
                ),
            )
        except ValueError as error:
            raise ValueError(
                f"{sample.dataset}/{sample.sample_id}/{part.part_id}: {error}"
            ) from error
        meshes[part.part_id] = mesh
        targets[part.part_id] = make_pose(target_center, target_basis)
        clouds[part.part_id] = candidates[selected].copy()
    initial = place_parts(
        {pid: mesh.vertices for pid, mesh in meshes.items()},
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        seed=config.initialization_seed,
        gap=config.min_gap,
    )
    # Bake the scattered scene into geometry. Public body poses start at identity.
    # Targets now act on these baked vertices: T_target @ inverse(T_initial).
    for pid, pose in initial.items():
        mesh, target = meshes[pid], targets[pid]
        rotation = rotation_matrix(pose.quaternion)
        target_rotation = rotation_matrix(target.quaternion) @ rotation.T
        meshes[pid] = Mesh(
            apply_pose(mesh.vertices, pose),
            mesh.faces,
            mesh.normals @ rotation.T,
            mesh.face_normal_indices,
        )
        clouds[pid] = apply_pose(clouds[pid], pose)
        targets[pid] = make_pose(target.position - target_rotation @ pose.position, target_rotation)
    parts = tuple(
        AssemblyPart(
            part_id=p.part_id,
            mesh=meshes[p.part_id],
            points=clouds[p.part_id],
            initial_pose=make_pose(np.zeros(3), np.eye(3)),
            gt_pose=targets[p.part_id],
            metadata=deepcopy(p.metadata),
        )
        for p in sample.parts
    )
    return AssemblySample(
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        revision=sample.revision,
        parts=parts,
        source_to_world=normalization,
        world_to_source=np.linalg.inv(normalization),
        config=config,
        metadata=deepcopy(sample.metadata),
        source_splits=deepcopy(sample.source_splits),
        manual_pages=deepcopy(sample.manual_pages),
        manual=deepcopy(sample.manual),
        steps=deepcopy(sample.steps),
        annotations=deepcopy(sample.annotations),
    )
