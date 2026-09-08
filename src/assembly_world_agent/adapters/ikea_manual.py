"""IKEA's assembled meshes are Y-up; see the coordinate inspection notes in README.md."""

import numpy as np

from .common import build_sample

REPO_ID = "AssemblyWorld/ikea-manual"
SAMPLE_ID_FIELD = "object_id"
DEFAULT_REVISION = "d2367e6f86610d38f2d2cc65278f680c9a444d83"
SOURCE_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
SOURCE_TO_Z_UP.flags.writeable = False


def reference_pages(row: dict, mode: str):
    """Preserve published manual order; its first page is the reference cover."""
    pages = row.get("manual_pages", [])
    if not pages:
        return []
    seen = set()
    last = {}
    for page in pages:
        identity = (page["manual_id"], page["page_index"])
        if identity in seen or page["page_index"] != last.get(page["manual_id"], -1) + 1:
            raise ValueError("IKEA manual pages must have unique consecutive original page indices")
        seen.add(identity)
        last[page["manual_id"]] = page["page_index"]
    return pages[:1] if mode == "final-image" else pages


def adapt(row: dict, revision: str):
    return build_sample(
        row,
        dataset=REPO_ID,
        revision=revision,
        source_to_z_up=SOURCE_TO_Z_UP,
        sample_id=row["object_id"],
    )
