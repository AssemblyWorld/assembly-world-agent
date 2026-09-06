"""Explicit source registry; no plugin discovery or cross-repository imports."""

from . import assemblybench, breaking_bad, ikea_manual, partnet_manualpa

ADAPTERS = {
    adapter.REPO_ID: adapter
    for adapter in (ikea_manual, partnet_manualpa, breaking_bad, assemblybench)
}


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
