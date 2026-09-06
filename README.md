# Assembly World Agent

Prepare reproducible assembly tasks from published
[AssemblyWorld datasets](https://github.com/AssemblyWorld/assembly-world-datasets).

**Hugging Face → dataset adapter → shared geometry processing → AssemblySample → initial episode.**

This independent Python package owns task geometry, initialization and task data.
3DWebAgent owns the generic runtime and episode contract. No sibling Python source
is required for preparation or export. This project does not run a model assembly
benchmark, implement an evaluator, or serve HTTP.

## Repository layout

```text
data/                             # Ignored reusable initial tasks
  ikea-manual/<config-id>/
    config.json
    Bench--applaro.episode.zip
    Chair--reidar.episode.zip
    Table--vittsjo_2.episode.zip
logs/                             # Ignored individual experiments
  <UTC-time>-<experiment>-<unique-id>/
    meta.json
    metrics.json
    <sample>--runtime.episode.zip
    screenshots/
scripts/                          # Python command entrypoints
src/assembly_world_agent/         # APIs, adapters, contracts and shared utils
tests/                            # Offline regression tests
```

There is no separate documentation directory. Data, logs, HF caches, virtual
environments and build artifacts must not enter Git.

## Installation and preparation API

Python 3.12 or later; run uv from this project:

```sh
uv sync --locked
uv run python
```

```python
from assembly_world_agent import PreparationConfig, load_samples, prepare_sample
from assembly_world_agent.utils import apply_pose

source = next(load_samples("ikea-manual", sample_ids=["Bench/applaro"]))
sample = prepare_sample(source, PreparationConfig())
part = sample.parts[0]
initial_vertices = apply_pose(part.mesh.vertices, part.initial_pose)
gt_points = apply_pose(part.points, part.gt_pose)
```

`load_samples` accepts a short dataset name or full Hub ID, `revision`, `limit`,
`sample_ids`, `streaming` and `cache_dir`. Streaming defaults to true. ID selection
may scan earlier rows, but only selected samples are prepared. The actual resolved
HF commit is retained; defaults are pinned. Explicit branch/tag resolution requires
Hub access; pinned revisions can use a populated offline HF cache. Standard HF
cache locations and `HF_HOME` apply; no dataset-local cache is created by default.

Missing IDs raise at exhaustion; an explicit limit may stop selection earlier.
Partial iterators should be closed when no longer needed. For bounded selections,
the HF reader closes before yielding the final selected source sample. An empty
ID list loads nothing. `adapt_sample(dataset, row, revision=sha)` supports decoded
HF rows; the caller supplies the actual source commit.

## Dataset and geometry protocol

| Adapter | Assembly interpretation | Source up | Pinned revision |
| --- | --- | --- | --- |
| `ikea-manual` | Meshes already share assembled coordinates | Y | `d2367e6f86610d38f2d2cc65278f680c9a444d83` |
| `partnet-manualpa` | Original `objs` in a common shape frame | Y | `e79907b38a868589184063debd9727e59f480cf3` |
| `breaking-bad-volume-constrained` | Fragments share source coordinates | Z | `aa6c781cdba90b2131ba090457f502e144819953` |
| `assemblybench` | Apply final view-0 assembled poses to local meshes | Z | `266e883e1676af42c44c41484c77b662b987b966` |

Protocol `assembly-preparation-v1` is a task preprocessing convention, not an
official GARF, ManualPA, Breaking Bad or AssemblyBench evaluation reproduction.

1. Restore assembly and convert to Z-up. IKEA and PartNet rotate `(x,y,z)` to
   `(x,-z,y)`; others use identity. AssemblyBench requires every final view-0 pose
   and does not substitute identities or motion start frames.
2. Use one shape-only scale: twice the largest vertex radius about a part vertex
   centroid. The private assembly is centered in XY and grounded at `z=0`.
   The scale does not depend on assembled part poses; units are normalized.
3. Center each part at its vertex centroid and use right-handed vertex PCA, with
   the smallest principal axis as local Z. Axis signs and repeated eigenspaces
   use ordered centered vertices, never assembled world axes. Vertex order breaks
   exact symmetry ties.
4. Preserve original polygons and normal indices. Triangulate a derived mesh for
   surface sampling/rendering, including concave polygons. Sample 4096 area-weighted
   candidates then retain 1000 FPS points in local coordinates. Best-fit-plane
   polygon triangulation preserves source 3D vertices. Zero total surface area,
   invalid indices, nonfinite coordinates and zero overall scale raise errors.
5. Give every part an independent yaw, ground its lowest vertex and place it using
   full rotated-mesh XY AABBs with a 0.02 minimum gap. Bounded retries expand the
   region; shuffled grid slots provide a checked nonoverlapping fallback. All
   parts move; there is no fixed anchor or physical stability claim.
6. Bake placement into vertices, normals and sampled points. Initial body poses
   are identity (zero translation, wxyz quaternion `[1,0,0,0]`). Rebase private GT
   poses to act on this initial geometry; initialization transforms are not exported.
   Body positions are transform translations, not geometric centers. Use an explicit
   `pivot` for rotation about a part center; the runtime default uses body origins.

Defaults are `surface_points=4096`, `fps_points=1000`, `sampling_seed=0`,
`initialization_seed=0`, `min_gap=0.02`. SHA-256-derived PCG64 streams separate
sampling, FPS and initialization. Changing initialization preserves the sampled surface points after GT mapping. Bitwise reproducibility is checked with the locked runtime, not promised
across arbitrary numerical library/platform versions.

`initial_pose` and `gt_pose` map the same baked initial geometry to task world using
translation and **wxyz** quaternions. `source_to_world` and `world_to_source` are
inverse 4×4 similarity transforms. Source annotations remain in their original
coordinates and IDs. Stored HF split is `full`; original memberships are metadata,
not newly generated splits. Missing manuals stay empty. Preparation copies input;
frozen dataclasses still contain mutable NumPy buffers that callers must preserve.

Coordinate inspection on 2026-09-06 used IKEA `Bench/applaro`, `Chair/reidar`,
`Table/vittsjo_2`; PartNet `688`, `38391`, `45182`; Breaking Bad artifact `101902_sf`,
Cup `d8ea3aa39bcb162798910e50f05b8001`, WineBottle `5ad47181a9026fc728cc22dce7529b69`;
and AssemblyBench `3`, `3758`, `8005`. Explicit adapter constants reflect these
source-frame inspections, not per-object PCA guesses. The AssemblyBench
[pinned card](https://huggingface.co/datasets/AssemblyWorld/assemblybench/blob/266e883e1676af42c44c41484c77b662b987b966/README.md)
and upstream Blender world-pose export establish the WXYZ assembled-pose convention.
Source dataset licenses and redistribution rights remain separate from this code.

## Initial episodes and configuration directories

```sh
uv sync --locked --extra episodes
uv run --extra episodes python scripts/convert_ikea.py
uv run --extra episodes assembly-world-agent convert-ikea \
  --sample-id Bench/applaro --sample-id Chair/reidar \
  --sampling-seed 0 --initialization-seed 0 --output data --logs logs
```

Only the three pilot IDs in the layout example are selected by default. Full
conversion requires `--all`. `--revision` and `--cache-dir` remain available.
`--output` is the **data root**, not a single configuration directory. Configuration
IDs have a readable protocol/seed prefix and a digest of resolved revision,
preprocessing parameters and producer identity. A configuration contains initial
ZIPs directly plus `config.json`, which records provenance, sample-to-file mapping,
episode IDs and archive hashes. Identical writes are reusable; different existing
archives fail rather than being replaced. Use one converter per configuration at
a time. Conversion reports are separate timestamped experiments under `logs`.

The generic API remains `export_episode(sample, output)`. It requires the `episodes`
extra: MuJoCo **3.12.0** and JSON Schema validation. The packaged public schema and
checksum provenance target 3DWebAgent commit
`9ce88b6da3b0bcaab0c68432e41a4fc18c556753`, `3dwebagent-episode` v1 and
`3dwebagent-runtime-1`. No runtime implementation is copied.

Each part becomes a stable independent free body. Derived OBJ meshes use shell
inertia and explicit placeholder mass/inertia, not physical estimates. A lone
triangle is subdivided without thickness or surface change for MuJoCo compatibility.
No repair or extra collision model is constructed. Physics, detection and response
are disabled; force and simulation tools are disabled. Query, pose, grouping,
capture, camera and lifecycle tools remain enabled. The initial camera fits the
complete layout.

Each ZIP contains `manifest.json`, one initial row in `frames.jsonl`, native
`mjSTATE_INTEGRATION` as little-endian Float64 in `frames.bin`, empty `calls.jsonl`
and `events.jsonl`, plus `world/model.xml` and meshes. Lifecycle is `setup`, with
empty groups and no history. Stable ordering, fixed ZIP metadata and payload hashes
make identical exports byte reproducible. Import through 3DWebAgent's **Episode
file** control. Manuals, GT and evaluation resources never enter initial episodes.

## On-demand resources and experiments

There are no persisted private resource sidecars. Reload a configuration's pinned
HF data and reconstruct task geometry when an experiment needs GT, point clouds,
manual images or annotations:

```python
from assembly_world_agent.artifacts import load_prepared, read_config

configuration = read_config("data/ikea-manual/<config-id>")
for source, sample in load_prepared("data/ikea-manual/<config-id>"):
    gt_pose = sample.parts[0].gt_pose
    evaluation_points = sample.parts[0].points
    manual_pages = sample.manual_pages  # In memory only
```

Experiments read `data` and write unique `logs/<run-id>/` directories. `meta.json`
records inputs, command, revision, source fingerprint, package versions and status;
`metrics.json` stores measured checks and partial results even when a run fails.
No assembly score is fabricated. GT-driven fixtures have their own experiment
kind/directory and are not agent assembly successes.

## Verification

```sh
uv run --extra episodes pytest
uv run ruff check .
uv run ruff format --check .
uv run --group inspection python scripts/inspect_samples.py
uv run --extra episodes python scripts/verify_episode_geometry.py \
  --config data/ikea-manual/<config-id>
```

Inspection loads a bounded sample per adapter; plots and results go to `logs`.
The geometry check rebuilds pinned HF samples, verifies archive reproducibility,
GT reconstruction, inverse transforms, unchanged input, and seed-independent
point clouds. Compiled render meshes use float32 vertices, so surface comparisons
use a separate `2e-7` normalized tolerance.

Browser conformance needs an explicit independent checkout of the pinned
3DWebAgent commit with its own pnpm and Python dependencies installed. Build and
serve that app from its checkout:

```sh
pnpm exec vite build --outDir /tmp/awa-pinned-dist
pnpm exec vite preview --outDir /tmp/awa-pinned-dist \
  --host 127.0.0.1 --port 5184 --strictPort
```

From this project, use Python Playwright (Node is still required for the upstream
WASM validation command):

```sh
uv sync --locked --extra episodes --group browser
uv run --group browser playwright install chromium
uv run --extra episodes --group browser python scripts/verify_episodes.py \
  --config data/ikea-manual/<config-id> --webagent /absolute/path/to/pinned/3DWebAgent \
  --base-url http://127.0.0.1:5184/
```

Tests use isolated browser contexts and upstream's registration adapter to invoke
actual tool callbacks. They cover file import, setup-to-active query, transforms,
groups, camera/capture, free-view independence, active/ended reload, exact call and
state persistence, corrupted-hash rejection, and separate GT-driven fixtures.
This does not claim external MCP client connectivity. Native/WASM saved-state
restoration is exact; cross-backend action reexecution uses `atol=1e-9, rtol=1e-7`.
Native screenshot checks validate PNG integrity, not rendering or pixel parity.

## MP4 and GIF replay

Render an existing initial or recorded episode without loading HF data or executing
its tools. Install the `episodes` extra and a system FFmpeg with H.264/GIF support:

```sh
uv run --extra episodes python scripts/render_episode.py /path/to/run.episode.zip \
  --output /path/to/run.replay
```

The output argument is a filename **stem**: both `.mp4` and `.gif` are appended.
Omit it to save media in the automatically created replay experiment under `logs`.
Use `--format mp4` or `--format gif` to select just one. Existing outputs are not
overwritten. The original ZIP remains unchanged.

```python
from assembly_world_agent.vis import render_episode

result = render_episode("run.episode.zip", "run.replay", formats=("mp4", "gif"))
```

The video shows the saved agent-camera view, call progress, actor/status, timestamps,
and the actual MCP arguments and return values. Long JSON is paginated, not silently
truncated; inline image bytes are represented by a label. The main view consistently
uses native MuJoCo rendering from saved integration states, recorded camera
position/target and the platform's 38° vertical FOV, including screenshot calls.
This avoids switching to a different renderer mid-replay. Original PNG observations
remain in the archive and are referenced in call results; they do not replace the
main view. Native lighting can differ from those Three.js observations. Recorded
camera changes and discrete pose updates still take effect. No assembly score is inferred.

Defaults: MP4 1600×900 at 12 fps, GIF 1280×720 at up to 6 fps, one second per JSON
page. This is a readable step replay, not original wall-clock pacing. Recorded
physics traces are sampled to at most 24 frames per call; no new intermediate poses
are synthesized. Adjust `--seconds-per-page`, `--fps` and `--max-trace-frames` as
needed. A setup-only archive produces an initial-scene clip. Native rendering needs
an available OpenGL context (CoreGraphics on macOS, or a configured MuJoCo EGL/OSMesa
backend on headless Linux). The CLI records parameters and output details in `logs`.
