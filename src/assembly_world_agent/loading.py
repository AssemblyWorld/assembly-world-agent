"""Lazy HF loading with immutable revision provenance and bounded selection."""

import gc
import re
from collections.abc import Iterator, Sequence
from numbers import Integral
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from .adapters import adapt_sample, get_adapter
from .models import SourceSample


def load_samples(
    dataset: str,
    *,
    revision: str | None = None,
    streaming: bool = False,
    sample_ids: Sequence[str] | None = None,
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Iterator[SourceSample]:
    """Yield source samples from the single stored `full` split.

    Default revisions are pinned releases. Branch/tag overrides resolve to a SHA
    once before loading. The default prepares the full split in the HF cache;
    selection and limit only bound subsequent adaptation. Explicit streaming can
    avoid full-release preparation but may scan earlier rows to select late IDs.
    Source split memberships remain annotations and are never treated as mutually
    exclusive HF splits.
    """
    adapter = get_adapter(dataset)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError("sample_ids must be a sequence of IDs, not one string")
    wanted = None if sample_ids is None else set(sample_ids)
    if wanted is not None and any(not isinstance(i, str) or not i for i in wanted):
        raise ValueError("sample_ids must contain nonempty strings")
    if wanted == set():
        return
    pinned = adapter.DEFAULT_REVISION if revision is None else revision
    if not pinned:
        raise ValueError("revision must not be empty")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", pinned):
        pinned = HfApi(token=token).dataset_info(adapter.REPO_ID, revision=pinned).sha
    rows = load_dataset(
        adapter.REPO_ID,
        split="full",
        revision=pinned,
        streaming=streaming,
        cache_dir=None if cache_dir is None else str(cache_dir),
        token=token,
        **({} if streaming else getattr(adapter, "LOAD_KWARGS", {})),
    )
    found = set()
    emitted = 0
    iterator = iter(rows)

    def close_reader():
        nonlocal iterator, rows
        if iterator is None:
            return
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        iterator = rows = None
        if streaming:
            # Release partially consumed Arrow fragments before caller computation
            # or interpreter shutdown (apache/arrow#45214).
            gc.collect()

    try:
        while iterator is not None:
            try:
                row = next(iterator)
            except StopIteration:
                break
            identity = row[adapter.SAMPLE_ID_FIELD]
            if wanted is not None and identity not in wanted:
                continue
            if identity in found:
                raise ValueError(f"Duplicate sample ID: {identity}")
            found.add(identity)
            source = adapt_sample(adapter.REPO_ID, row, revision=pinned)
            del row
            emitted += 1
            last = (limit is not None and emitted >= limit) or (
                wanted is not None and found == wanted
            )
            if last:
                close_reader()
            yield source
            if last:
                return
    finally:
        close_reader()
    if wanted is not None and wanted - found:
        raise ValueError(f"Requested sample IDs not found: {sorted(wanted - found)}")
