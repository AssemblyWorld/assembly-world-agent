# Assembly World Agent

Consume published Hugging Face datasets through adapters. Own task normalization,
sampling, initialization and ground-truth poses. Shared geometry belongs in
`src/assembly_world_agent/utils`; adapters only interpret source data. Preserve
source IDs, revisions, annotations and rights metadata. Do not modify input data.
Keep original polygons; sampling and episode rendering use derived triangulation.

Produce setup episodes using the pinned public 3DWebAgent contract. Do not invent
an episode format, copy runtime source or import sibling dataset packages.
HTTP serving is limited to the experiment-owned loopback service for selected
episode archives; do not expose source datasets or task answers. Call the
independent 3DWebAgent checkout through explicit paths for
native, WASM and browser conformance tests. Offline evaluation computes versioned
free-space SCD/PA/SR from final episodes. Adapters normalize source equivalence
annotations; shared similarity processing supplies source or geometry groups on
demand, defaulting to geometry. Similarity configuration is separate from episode
preparation identity. Shape-pair registration must never modify assembly scoring
poses. Record grouping thresholds, provenance and transitive-closure diagnostics.
Preparation and deterministic GT fixtures are not evidence of model success.

Keep reusable initial ZIPs directly in `data/<dataset>/<config-id>/` alongside
`config.json`. Derived private data (reference images, GT poses, point clouds, equivalence
groups) may persist only in `data/**/cache/<sample>/`, keyed by preparation identity and
episode checksum, filled on first use and never overwritten on a key mismatch. Nothing from
the cache enters an episode or is served to an agent; the agent only receives the episode
ZIP. Review each source's license before publishing a data package that includes caches.
Everywhere else, reload pinned HF data and reconstruct in memory. Use standard HF caches
or an explicit external cache path. Experiments only read data.

Every experiment writes a unique `logs/<run-id>/` with `meta.json`, `metrics.json`,
runtime episodes and optional screenshots. Record failures and partial checks.
Use separate experiment directories for GT-driven fixtures. Do not invent metrics
that were not computed. Centralize artifact and run organization in the package;
Python scripts and CLI entrypoints only orchestrate workflows.

After smoke tests and development validation finish, remove their temporary log
directories, screenshots, exported runtime episodes, helper scripts and other
test artifacts, including failed attempts. Retain them only when deliberately
prepared for the user to inspect, and identify those retained artifacts in the
handoff. Preserve reusable prepared episodes, shared dataset caches and actual
experiment results; do not delete artifacts belonging to unrelated work.

Use uv in this independent project with its own lockfile. Keep `scripts/` Python-only,
retain offline tests, and put all maintained documentation in README.md. Do not add
`docs/`. Ignore data, logs, caches, environments and build outputs in Git. Complete
bounded real-data checks when geometry, adapters or artifact interfaces change.
Code, comments and documentation are English; conversation and plans are Chinese.

Offline MP4/GIF replay lives in `vis` and consumes episode records only. Restore
saved states without executing tools or loading datasets. Distinguish original
observations from native visualization, preserve complete paginated call text, and
record render metadata in logs. Explicit user output destinations are supported.

AssemblyWorldBench lives in `data/assemblyworldbench/` as five standard configuration
directories plus `benchmark.json`; its frozen selection (spec, exclusions, manifest,
overrides) is versioned under `benchmarks/assemblyworldbench/` and was chosen from task
properties only, never from model results. Changing a sample, quota, exclusion or task text
is a new benchmark version, not an edit. Run it with the ordinary `run` command and each
block's `task.txt`; score it with `scripts/evaluate_run.py --benchmark`, one evaluation
protocol for every block, aggregated by the rules recorded in `benchmark.json`.
