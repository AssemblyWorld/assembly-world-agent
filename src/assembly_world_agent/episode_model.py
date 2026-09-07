"""Load public models and inspect compiled geometry without source sidecars."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .episodes import ENGINE
from .utils import rotation_matrix


def load_model(episode):
    import mujoco as mj

    if mj.__version__ != ENGINE:
        raise ValueError(f"Expected MuJoCo {ENGINE}")
    descriptor = episode["manifest"].get("model")
    files = episode["files"]
    if descriptor is not None:
        if descriptor != {"format": "mjb", "path": "model.mjb"}:
            raise ValueError("Unsupported episode model declaration")
        with TemporaryDirectory(prefix="awa-mjb-") as directory:
            path = Path(directory) / "model.mjb"
            path.write_bytes(files["world/model.mjb"])
            return mj.MjModel.from_binary_path(str(path))
    return mj.MjModel.from_xml_string(
        files["world/model.xml"].decode(),
        assets={k[6:]: v for k, v in files.items() if k.startswith("world/")},
    )


def compiled_parts(model, objects):
    """Return full triangle geometry in each body's frame (geom offsets applied once)."""
    import mujoco as mj

    parts = []
    for obj in objects:
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj["id"])
        if body < 1 or model.body_parentid[body] != 0 or model.body_geomnum[body] != 1:
            raise ValueError("Result viewer requires one mesh per independent assembly body")
        geom = model.body_geomadr[body]
        if model.geom_type[geom] != mj.mjtGeom.mjGEOM_MESH:
            raise ValueError("Expected mesh geometry")
        mesh = model.geom_dataid[geom]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = model.mesh_vert[va : va + vn].astype(float)
        vertices = vertices @ rotation_matrix(model.geom_quat[geom]).T + model.geom_pos[geom]
        parts.append(
            dict(
                id=obj["id"],
                resource=f"model.mjb#mesh-{mesh}",
                geometry=dict(
                    vertices=vertices.tolist(),
                    triangles=model.mesh_face[fa : fa + fn].ravel().tolist(),
                ),
            )
        )
    return parts


def validate_compiled_model(actual, expected):
    """Compare immutable task geometry and physical structure, excluding runtime options."""
    # Runtime physics settings can change options and contact masks; these are not geometry.
    fields = (
        "names",
        "name_bodyadr",
        "body_parentid",
        "body_jntadr",
        "body_jntnum",
        "body_geomadr",
        "body_geomnum",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_pos",
        "jnt_axis",
        "jnt_limited",
        "jnt_range",
        "qpos0",
        "dof_damping",
        "geom_type",
        "geom_bodyid",
        "geom_dataid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_rgba",
        "geom_friction",
        "geom_condim",
        "mesh_vertadr",
        "mesh_vertnum",
        "mesh_faceadr",
        "mesh_facenum",
        "mesh_vert",
        "mesh_face",
        "mesh_pos",
        "mesh_quat",
        "mesh_normal",
        "mesh_facenormal",
    )
    for name in (
        "nbody",
        "njnt",
        "ngeom",
        "nmesh",
        "nq",
        "nv",
        "nu",
        "na",
        "neq",
        "ntendon",
        "nplugin",
        "nflex",
        "nskin",
        "ntex",
    ):
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(f"Episode compiled model differs: {name}")
    for name in fields:
        a, b = getattr(actual, name), getattr(expected, name)
        if isinstance(a, bytes):
            equal = a == b
        else:
            a, b = np.asarray(a), np.asarray(b)
            equal = a.shape == b.shape and (
                np.allclose(a, b, atol=2e-7, rtol=0)
                if np.issubdtype(a.dtype, np.floating)
                else np.array_equal(a, b)
            )
        if not equal:
            raise ValueError(f"Episode compiled model differs: {name}")
