# Assembly World Agent

Consume published Hugging Face datasets through adapters. Own task normalization,
sampling, initialization and ground-truth poses. Shared geometry belongs in
`src/assembly_world_agent/utils`; adapters only interpret source data. Preserve
source IDs, revisions, annotations and rights metadata. Do not modify input data.
Keep original polygons; sampling and episode rendering use derived triangulation.

Produce setup episodes using the pinned public 3DWebAgent contract. Do not invent
an episode format, copy runtime source, add HTTP serving or import sibling dataset
packages. Call the independent 3DWebAgent checkout through explicit paths for
native, WASM and browser conformance tests. Evaluation is future work; preparation
and deterministic GT fixtures are not an official benchmark or model success.

Keep reusable initial ZIPs directly in `data/<dataset>/<config-id>/` alongside
`config.json`. Never persist private GT, point clouds, manuals or source annotations.
Reload pinned HF data and reconstruct these resources in memory when needed. Use
standard HF caches or an explicit external cache path. Experiments only read data.

Every experiment writes a unique `logs/<run-id>/` with `meta.json`, `metrics.json`,
runtime episodes and optional screenshots. Record failures and partial checks.
Use separate experiment directories for GT-driven fixtures. Do not invent metrics
that were not computed. Centralize artifact and run organization in the package;
Python scripts and CLI entrypoints only orchestrate workflows.

Use uv in this independent project with its own lockfile. Keep `scripts/` Python-only,
retain offline tests, and put all maintained documentation in README.md. Do not add
`docs/`. Ignore data, logs, caches, environments and build outputs in Git. Complete
bounded real-data checks when geometry, adapters or artifact interfaces change.
Code, comments and documentation are English; conversation and plans are Chinese.

Offline MP4/GIF replay lives in `vis` and consumes episode records only. Restore
saved states without executing tools or loading datasets. Distinguish original
observations from native visualization, preserve complete paginated call text, and
record render metadata in logs. Explicit user output destinations are supported.
