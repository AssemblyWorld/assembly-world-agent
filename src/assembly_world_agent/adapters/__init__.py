"""Explicit source registry; no plugin discovery or cross-repository imports."""

from . import assemblybench, breaking_bad, fantastic_breaks, ikea_manual, partnet_manualpa

ADAPTERS = {
    adapter.REPO_ID: adapter
    for adapter in (ikea_manual, partnet_manualpa, breaking_bad, assemblybench, fantastic_breaks)
}

REFERENCE_MODES = ("none", "final-image", "manualbook")


def reference_pages(dataset: str, row: dict, mode: str):
    """Select ordered source pages without decoding geometry or exposing annotations."""
    if mode not in REFERENCE_MODES:
        raise ValueError(f"Unsupported reference mode: {mode}")
    if mode == "none":
        return []
    adapter = get_adapter(dataset)
    selector = getattr(adapter, "reference_pages", None)
    if selector is None:
        raise ValueError(f"{adapter.REPO_ID} does not define reference mode {mode}")
    pages = selector(row, mode)
    if not pages:
        raise ValueError(f"{adapter.REPO_ID}/{row.get('object_id')}: missing reference images")
    return pages


def get_adapter(dataset: str):
    repo_id = dataset if "/" in dataset else f"AssemblyWorld/{dataset}"
    if repo_id not in ADAPTERS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {list(ADAPTERS)}")
    return ADAPTERS[repo_id]


def adapt_sample(dataset: str, row: dict, *, revision: str):
    """Adapt a decoded HF row with explicitly supplied revision provenance."""
    adapter = get_adapter(dataset)
    try:
        return adapter.adapt(row, revision)
    except (ValueError, KeyError, TypeError) as error:
        identity = row.get("sample_id", row.get("object_id", "<unknown>"))
        raise ValueError(f"{adapter.REPO_ID}/{identity}: {error}") from error
