# Assembly World Agent

Prepare reproducible assembly tasks from published
[AssemblyWorld datasets](https://github.com/AssemblyWorld/assembly-world-datasets).

**Hugging Face → dataset adapter → shared geometry processing → AssemblySample → initial episode.**

This independent Python package owns task geometry, initialization and task data.
3DWebAgent owns the generic runtime and episode contract. No sibling Python source
is required for preparation or export. Browser experiments use isolated Chrome
instances through WebMCP; ground-truth scoring remains a separate offline workflow.
This project does not serve HTTP.

## Repository layout

```text
data/                             # Ignored reusable initial tasks
  ikea-manual/<config-id>/
    config.json
    Bench--applaro.episode.zip
    Chair--reidar.episode.zip
    Table--vittsjo_2.episode.zip
logs/                             # Ignored individual experiments
  <UTC-time>-browser-<unique-id>/
    run.json
    samples/<sample-id>/
      input.json
      prompt.txt
      conversation.jsonl
      result.json
      final.episode.zip
scripts/                          # Python command entrypoints
src/assembly_world_agent/         # APIs, adapters, contracts and shared utils
tests/                            # Offline regression tests
```

There is no separate documentation directory. Data, logs, HF caches, virtual
environments and build artifacts must not enter Git.

## Browser experiments: Codex and Claude Code

Install and authenticate the desired CLI separately. Install Chrome 150 or newer,
Node.js compatible with Chrome DevTools MCP, and the pinned MCP executable:

```sh
pnpm add --global chrome-devtools-mcp@1.8.0
uv sync --locked --extra episodes --group browser
uv run assembly-world-agent doctor --agent codex
uv run assembly-world-agent doctor --agent claude
```

`doctor` launches a temporary Chrome profile, checks the installed CLI,
MCP and Playwright versions, and verifies actual page WebMCP registration and MCP
connectivity. It does not invoke a model or establish account/model availability.
Chrome must be allowed to launch on the host. By default windows are visible and
Linux needs a graphical session. Add `--headless` to `doctor` or `run` to use
modern Chrome without visible windows. This mode has been tested on macOS;
WebGL rendering and required system libraries still need verification on a target
Linux server. Use regular Chrome, not the legacy `chrome-headless-shell`.
Use `--chrome-path /absolute/path/to/chrome` or
`--mcp-command /absolute/path/to/chrome-devtools-mcp` for nonstandard installations.
The runner never installs packages or changes global MCP configuration.

```sh
# One existing episode; use a model available to the authenticated account.
uv run assembly-world-agent run \
  --episode /absolute/path/sample.episode.zip \
  --agent codex --model MODEL_ID

# An explicitly selected preparation configuration, four concurrent samples.
uv run assembly-world-agent run \
  --dataset ikea-manual --config-id PREPARATION_CONFIG_ID \
  --agent claude --model MODEL_ID --concurrency 4

# Headless mode uses the same WebMCP tools, screenshots and full episode export.
uv run assembly-world-agent doctor --agent claude --headless
uv run assembly-world-agent run \
  --episode /absolute/path/sample.episode.zip \
  --agent claude --model MODEL_ID --headless

# Read progress; retry unfinished samples in a NEW run directory.
uv run assembly-world-agent status /absolute/path/logs/RUN_ID
uv run assembly-world-agent resume /absolute/path/logs/RUN_ID --concurrency 4
uv run assembly-world-agent resume /absolute/path/logs/RUN_ID --retry-failed
```

Dataset runs read `data/<dataset>/<config-id>/config.json` and existing episode
ZIPs without converting or changing them. Use `--data` for another data root and
`--logs` for another log root. Select a subset with repeated `--sample-id` or
`--limit`; otherwise all prepared samples run. The default concurrency is one.
`--timeout-seconds` limits the agent phase; by default it has no time limit.
Startup and final export have separate bounded timeouts. `--effort` selects a
provider-supported effort; unavailable models/settings fail in the CLI.

The default task assembles the object and checks connections through WebMCP.
`--prompt-file` replaces the task text while preserving transport instructions.
Pinned HF manuals are reconstructed on demand, or use `--manual` with a directory
of PNG/JPEG/WebP pages in filename order. An independent episode automatically
uses a neighboring preparation configuration only if its filename, identity and
checksum match. With no manual provenance, the default is geometry-based assembly.
Manual files live in temporary directories; only provenance and hashes are stored
in `input.json`. Images actually read by the agent remain in its conversation.

Each sample receives its own Chrome process/profile, MCP connection and CLI
process. The default environment is `https://3dwebagent.davidz.cn/`; supply
`--environment-url` to use an already-running pinned deployment. One static episode
service is shared by every worker in a run, including single-episode runs and
resumes. It binds to an ephemeral port on `127.0.0.1` and serves only explicitly
selected archives through unguessable URLs; there is no directory listing or
upload endpoint. Files are streamed from their existing locations, with no extra
dataset cache. Cross-origin reads are allowed only for the configured environment.
The service closes after the workers finish, including failed or interrupted runs.
The temporary Chrome profile grants local-network access only to the configured
environment origin so its HTTPS page can fetch the loopback episode URL.

The scheduler imports through the environment's public `?episode=` URL interface
and exports through the public UI. This avoids Playwright's 50 MiB limit for file
transfers over a non-local CDP connection. Chrome and the static service run on the
same host; external CDP browsers are not supported by this runner. URL loading
errors fail the sample instead of running the agent against an empty fallback
scene. Agents only receive page discovery,
WebMCP discovery/execution and supplied-manual reading tools. A stdio forwarding
adapter promotes serialized capture images to MCP image blocks and records full
tool results. It contains no scene implementation or native runtime backend.
Input physics settings and existing history are preserved. Local command execution
and additional agents are disabled; credentials are reused without being copied
into experiment artifacts. Host-managed client policies still apply.

The compact log has six file types:

| File | Purpose |
| --- | --- |
| `run.json` | Configuration, task text, versions, sample IDs, overall status and source run |
| `input.json` | Input path/hash, episode identity, dataset provenance and manual provenance |
| `prompt.txt` | Actual task prompt submitted to the CLI |
| `conversation.jsonl` | Incremental public messages, tool calls/results and extension events |
| `result.json` | Execution and export status, final answer, reported outcome, usage and errors |
| `final.episode.zip` | Verified full browser export, including existing and new history |

`result.json` distinguishes CLI completion, agent-reported assembly outcome, and
archive success. Unknown costs/usage stay null. Conversation records exclude private
reasoning and hidden instructions; unknown public events are retained. No separate
trace, stderr, progress, CSV, report, or attempt files are generated. A final ZIP
is absent if import/export failed; the input is never substituted as a final result.

On interruption, dispatch stops and active samples attempt to export before closing.
`resume` selects pending/running/interrupted samples; `--retry-failed` additionally
selects failed samples. Already archived successful executions, including an
agent-reported partial assembly, are skipped. Every retry starts from the original
input in a new run referencing its predecessor, leaving the old run untouched.
There is no automatic retry or browser-crash checkpoint recovery.
The browser mode is saved as `options.headless` in `run.json` and inherited by
`resume`. Older logs without this field retain visible-browser behavior.

Scoring and replay rendering remain separate commands. The evaluator accepts both
new `run.json` and legacy `meta.json` metadata; dataset provenance is required for
GT scoring. The browser run itself does not create metrics or videos.

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
`sample_ids`, `streaming` and `cache_dir`. Streaming defaults to **false** for every
dataset: the first load downloads and prepares the entire `full` split in the HF
cache; later loads reuse it. `sample_ids` and `limit` bound subsequent adaptation
and task preparation, not the initial download or Arrow cache preparation.
Explicit `streaming=True` avoids full-split preparation but may scan earlier rows
to select late IDs. Fantastic Breaks uses `writer_batch_size=1` for non-streaming
loads to bound Arrow writer batches for its high-resolution meshes. The actual resolved
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
| `fantastic-breaks` | Broken and synthetic repair meshes share supplied coordinates | Identity convention; physical up unverified | `654d94e30e3cb246a04f97aab1bc4ca9f0ad0af9` |

Protocol `assembly-preparation-v1` is a task preprocessing convention, not an
official GARF, ManualPA, Breaking Bad, AssemblyBench or Fantastic Breaks evaluation reproduction.

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
   `end_episode` is disabled in exported runtime capabilities; agents cannot end
   the episode through WebMCP.

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

Fantastic Breaks uses the original `object_id` strings, including leading zeros.
Its two assembly inputs are the scanned broken mesh and a **synthetic repair**
proxy, not two independently scanned fragments. The supplied coordinates define
GT; identity axis conversion is a task convention, not a verified physical up axis.
The source `annotation.transform` has unverified direction and units and is never
applied. Original polygons, normals, RGBA vertex colors, PLY headers, source paths,
hashes and annotations are preserved in memory. Colors remain source metadata;
they do not change the shared episode rendering contract.
`complete_reference` remains a separate in-memory annotation: it never contributes
to task geometry, normalization, point sampling, episode inputs or evaluation GT.
No source splits, class labels, physical units or dataset license are inferred.

```python
source = next(load_samples("fantastic-breaks", sample_ids=["00/00002"]))
sample = prepare_sample(source, PreparationConfig())
```

### Interactive sample inspection

Open `notebooks/inspect_samples.ipynb` after `uv sync --group inspection` and select
`.venv/bin/python` as the kernel. Set `DATASET`, `SAMPLE_ID`, `CACHE_DIR` and the
sampling/initialization seeds, then Run All. The default is Fantastic Breaks;
`SAMPLE_ID=None` selects the first row after preparing the full split in the cache.
Every registered adapter is supported, without experiment logs or episode export.

The notebook shows provenance and geometry counts, source assembly, normalized GT,
initial layout and production point clouds using consistent part colors and
interactive Plotly views. All mesh vertices are displayed with derived triangulation;
no input simplification is performed. High-resolution mesh views can take time to render.
Fantastic Breaks adds a separate reference-only complete mesh view and a literal
matrix/mask summary. Production functions reconstruct GT and verify the inverse
mapping back to source coordinates. These are preprocessing checks, not model scores.
Clear all outputs before saving/sharing; private GT and annotations stay in memory.

## Initial episodes and configuration directories

```sh
uv sync --locked --extra episodes
uv run --extra episodes assembly-world-agent convert fantastic-breaks \
  --sample-id 00/00002 --sample-id 00/00003 --sample-id 00/00005
uv run --extra episodes python scripts/convert.py assemblybench --limit 3
# Full conversion must be explicitly requested:
uv run --extra episodes assembly-world-agent convert fantastic-breaks --all
```

`convert <dataset>` accepts any registered short name or full Hub ID. Exactly one
selection is required: repeated `--sample-id`, a positive integer `--limit`, or
`--all`. Omitting selection or combining selection modes is a command-line error
before loading. IDs are preserved as strings. ID selections follow source row
order; `--limit` converts the first N rows. Missing requested IDs fail at exhaustion.
Loading is non-streaming and uses the standard HF cache; selection does not limit
first-load download/cache preparation. The default dataset revision is pinned;
`--revision` overrides it. `--cache-dir`, `--sampling-seed` and
`--initialization-seed` remain available; other preparation defaults are unchanged.

The compatibility command `convert-ikea` and `scripts/convert_ikea.py` retain their
existing arguments and the three default IKEA pilot IDs. They call the same
conversion workflow; their full conversion still requires `--all`.

```python
from assembly_world_agent.conversion import convert_dataset

summary = convert_dataset("fantastic-breaks", sample_ids=["00/00002", "00/00003", "00/00005"])
# summary: config_directories, converted_count, log_directory
```

The Python API also requires exactly one of `sample_ids`, `limit` or
`all_samples=True`. Pass a `PreparationConfig` through `preparation` when needed.
The workflow processes samples serially and prints each completed archive record.
Progress is saved after every sample. The final summary includes configuration
paths, the completed count and the log location. On failure, conversion stops,
keeps successful archives, records the error and the sample ID when available,
and the CLI returns a nonzero status. Loader errors retain their source context
in the error message rather than being attributed to the previous successful ID.
Exceptions propagate from the Python API. Re-running regenerates and verifies
archives before reuse; it is not a fast skip-existing resume operation.

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

By default, each ZIP contains `manifest.json`, one initial row in `frames.jsonl`, native
`mjSTATE_INTEGRATION` as little-endian Float64 in `frames.bin`, empty `calls.jsonl`
and `events.jsonl`, plus `world/model.xml` and meshes. Lifecycle is `setup`, with
empty groups and no history. Stable ordering, fixed ZIP metadata and payload hashes
make identical exports byte reproducible. Import through 3DWebAgent's **Episode
file** control. Manuals, GT and evaluation resources never enter initial episodes.

### Optional MJB conversion for high-resolution meshes

XML remains the default for every dataset and for `convert-ikea`. To move model
compilation to the native conversion stage, explicitly select MJB:

```bash
uv run --extra episodes assembly-world-agent convert fantastic-breaks \
  --model-format mjb \
  --sample-id 00/00002 --sample-id 00/00003 --sample-id 00/00005
```

MJB is an optional episode **v1** extension, declared as
`model: {"format":"mjb","path":"model.mjb"}` in the manifest. It preserves the
full compiled geometry and stores only `world/model.mjb` as the model asset.
Initial state comes from that same native MuJoCo 3.12.0 model. Source XML/OBJ,
GT, reference geometry and annotations are not added to the archive. Source data
remains reconstructible from the pinned HF revision. Standard non-streaming HF
cache behavior and selection/download semantics are unchanged.

The extended v1 schema is pinned separately under `contracts/mjb`, with its exact
hash and upstream commit provenance. The original XML schema and producer
remain unchanged. Format-specific configuration identity keeps MJB and XML
outputs in separate directories. Repeated identical conversions reuse files;
different bytes are never silently overwritten.

MJB requires the updated independent 3DWebAgent runtime and MuJoCo 3.12.0; older
XML-only runtimes cannot load it. Poses, grouping, queries, cameras, capture,
export and replay remain available. MJB has fixed parts, so adding/deleting
objects and editing model properties are unavailable. XML editing is unchanged.

The original Fantastic Breaks XML pilot `00/00002` exceeded the 2 GiB WASM heap
limit during mesh compilation in 3DWebAgent commit
`adfe1d9958711dd2b8fcac2df2e75e06aea0beba`. MJB avoids that compilation stage;
it does not reduce mesh resolution and may be larger than the source assets.
Use the MJB-capable local checkout for validation before browser experiments;
this change does not deploy the public site.

The three MJB pilots (`00/00002`, `00/00003`, `00/00005`) passed native geometry,
pinned-source GT reconstruction and exact initial-state restoration, plus WASM
loading of all three archives. Repeated conversion retained identical hashes.
The first pilot also passed local browser rendering, capture, pose/group changes,
export/reimport, history inspection and two repeated imports. Browser export
retained the original MJB bytes. These checks establish loading and replay
compatibility, not assembly success or physical stability.

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

## Browser-agent batch experiments

`assembly_world_agent.batch_experiment` organizes one parent experiment with a
separate `samples/<sample-id>/` directory per independent browser agent. It records
the exact prompt, pinned input identity, progress, full runtime archives, exportable
agent traces, final reports and MP4 render metadata. Archive ingestion checks the
sample identity and payload hashes and refuses conflicting final replacements.
Reported completion and connection inspection are agent claims, not GT scores.

Manual page copies are opt-in experiment inputs: use `prepare_manuals(run)` only
with explicit authorization to retain original manual pages in that experiment.
It preserves the pinned source page order and image bytes without retaining GT or
source assembly annotations. A user-authorized external loopback resource server
may supply initial archive and manual URLs to the online environment. The agent
runtime, browser tools and episode contract remain owned by 3DWebAgent.

`export_trace` retains available task messages, tool calls/results and usage from
a sample's own session log, excluding private reasoning and system instructions.
`render_sample` validates and renders the saved final archive in the same parent
experiment and checks that FFmpeg can decode the resulting MP4.

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

## Standalone HTML results

Export all configured samples from an experiment into one offline HTML snapshot:

```sh
uv run --extra episodes python scripts/export_results.py logs/<run-id> \
  --output logs/<run-id>/results.html
```

Open the HTML directly in a modern browser (`file://` works). Three.js, orbit
controls, original manual pages, meshes and recorded poses are embedded; no CDN,
server or neighboring files are required. Generation loads the experiment's pinned
HF revision with `streaming=False` to rebuild GT, using the standard HF cache
(`HF_HOME` or `--cache-dir`). The first run downloads and prepares the dataset;
later runs reuse that dataset cache. Each export regenerates the full HTML.
For pinned IKEA data, export reads geometry directly from Arrow without decoding
HF manual images, normals or annotations or running the full dataset adapter.
Other datasets retain their existing adapter paths.
Result export skips surface/FPS point-cloud sampling; geometry normalization,
initial placement and GT poses use the same preparation protocol.
It never executes recorded calls or physics. Re-running replaces only the requested
HTML atomically; the experiment inputs remain unchanged. Keep exports out of Git.

Manual pages are always embedded as lossless WebP, preserving RGBA pixels and
original dimensions without decoding them again for pixel verification. ICC profiles
and EXIF metadata are carried over when present. No separate compression command is needed.
Export skips manual and episode SHA-256 verification and embedded-image hashes;
archive parsing and structural validation still report unreadable recordings.
WebP conversion and row compression use up to 16 threads (capped by CPU count) with bounded pending
work, preserving page and sample order without adding another cache.
GT-only rows serialize mesh arrays directly; recorded rows reuse episode geometry
after comparing mesh arrays against the reconstructed task, without generating
temporary OBJ text or repeating full input validation.

Each row contains a paginated manual, GT viewer, recorded-state viewer and the
current MCP call with complete arguments and expandable return/error text. Original
observation references remain textual call metadata; the viewer renders geometry.
A valid final episode takes precedence over the newest valid checkpoint. Missing
recordings stay explicitly empty; unavailable resources and rejected archives are
reported per row. GT meshes must match the recorded baked initial geometry.

The page opens paused at each episode's last call, following its recorded camera.
Part colors are enabled by default, with matching colors in GT and replay views.
The global Part colors checkbox switches all parts between distinct colors and neutral gray.
Episodes without calls show their initial state. Global First/Last buttons pause playback.
Loop all/Pause all control every recorded
sample at one second per call. All calls includes queries and failed calls;
Changes only retains changes to body poses, groups or the recorded camera. Original
call numbers are preserved. Camera following defaults on; orbiting a replay view
turns it off for that row until re-enabled. Manual pagination is independent of
calls. Use the mouse to rotate, wheel to zoom, and right-drag to pan. Each viewer
has a reset control. This snapshot is a visual inspection aid, not a computed score.

The implementation lives in `vis/results.py` and `vis/web/`. Rows are gzip-packed
and decoded when approaching the viewport. Visible views share a single WebGL
renderer and copy its output into per-view 2D canvases, avoiding WebGL context
limits. Manual pages use lossless WebP to reduce the single-file size.
The embedded manifest records the snapshot time, pinned configuration and episode
source paths. Generation requires the `episodes` extra but no OpenGL context.

The checked-in `three.bundle.js` contains Three.js **0.180.0** and OrbitControls
under MIT (`vis/web/THREE-LICENSE.txt`). To rebuild it, use a temporary directory,
`pnpm add three@0.180.0 esbuild@0.25.10`, and bundle an entry containing:

```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
window.ResultThree = { THREE, OrbitControls };
```

Run esbuild with `--bundle --minify --format=iife` and write its output to
`vis/web/three.bundle.js`; no Node installation is needed to generate reports.

## Offline assembly evaluation

```bash
uv run --extra episodes scripts/evaluate_run.py logs/<run-id>
# Optional writable or existing Hugging Face dataset cache:
uv run --extra episodes scripts/evaluate_run.py logs/<run-id> --cache-dir /path/to/cache
```

The public Python entry point is `assembly_world_agent.evaluation.evaluate_run(run,
*, cache_dir=None, similarity=None)`. All configured samples are scored, including
agent-reported partial outcomes. At most four independent worker processes evaluate samples;
output rows always follow sorted sample IDs. Python scripts calling `evaluate_run`
should use an `if __name__ == "__main__"` guard for multiprocessing.
The evaluator reads `meta.json`, sample `input.json`, and
`final.episode.zip`; it does not require checkpoints, browser exports, a running
server, a display, or the original preparation directory. Pinned HF data rebuilds
GT and deterministic 4096-candidate/1000-FPS surface points in memory. No GT or
point-cloud artifacts are written to the experiment.

Protocol `assembly-evaluation-v2` follows Manual-PA's metric formulas at commit
`df784cae8ee8ae512436f9e762e99aec47036b9a`: bidirectional mean squared Chamfer
(summed directions), whole-shape SCD multiplied by 1000, PA as the fraction of
parts with Chamfer <= 0.01, and SR as whether every part passes. Aggregate all
three by sample macro mean. PA and SR use fractions, not percentages.

Evaluation divides both predicted and GT coordinates by the same largest
per-part vertex-PCA AABB diagonal, making that diagonal one. This converts the
task's radius-based scale without modifying recorded states or fitting scale to
predictions. One shared proper rigid transform minimizes whole-shape Chamfer
using 24 PCA axis starts and same-ID part-pose starts. Bidirectional nearest
neighbors and Kabsch refine each start for up to 100 iterations (improvement
threshold 1e-9); the best encountered objective wins, with stable first-candidate
ties. This is approximate multistart registration, not guaranteed global search.
During assembly scoring, no per-part alignment, reflection, scaling, tool execution
or simulation steps are allowed. MuJoCo restores the final logical state (initial
for a setup-only episode); groups are not applied again to the recorded body poses.

After alignment, Hungarian assignment uses unclipped part Chamfer costs within
configured equivalence groups. Adapters translate source atomic annotations into
actual part IDs; shared data processing resolves groups on demand. Metrics never
interpret dataset-specific equivalence syntax.

`SimilarityConfig(policy="geometry", threshold=1e-4)` is independent of task
preparation and episode identity. `resolve_equivalence(prepared_sample, config=...)`
returns stable ID groups and diagnostics. Two policies are supported:

- `geometry` (default): compare the prepared 1000-point input clouds using the same
  shared furniture scale. Each pair uses 24 proper PCA starts and symmetric rigid
  ICP (100 iterations, improvement tolerance 1e-9), without GT pose starts, scale
  fitting or reflections. Use raw bidirectional squared CD <= 0.0001 (SCD <= 0.1)
  as an edge, then take connected components. Groups may contain pairs above the
  threshold through transitive closure; per-group maxima and such pairs are recorded.
- `source`: use only source `geometric_equivalence_relation` annotations mapped
  through annotation IDs. Missing annotations yield singleton groups and an explicit
  missing flag. Composite self-relations such as `"0,2,3": ["0,2,3"]` do not make
  their constituent parts interchangeable. Unknown references, malformed relations
  and nontrivial composite equivalences are errors. Original annotations are retained
  in memory. No geometric fallback or union is applied.

Pair registration discovers shape equivalence only: its transforms never alter the
predicted assembly. The assembly PA threshold remains 0.01. Smaller parts contribute
smaller absolute errors under the common furniture scale. The geometry threshold is
configurable, not selected to maximize predicted scores. Sampling, free-space alignment
and group construction differ from Manual-PA; this is not exact paper reproduction or
physical stability validation.

```bash
uv run --extra episodes scripts/evaluate_run.py <run> --similarity-policy geometry --similarity-threshold 0.0001
uv run --extra episodes scripts/evaluate_run.py <run> --similarity-policy source
```

Python callers pass `similarity=SimilarityConfig(...)` to `evaluate_run`. The threshold
CLI option is geometry-only. Only the current policy's two output files are retained;
switching policies replaces them, rather than creating parallel score files.

Outputs are atomically replaced on every invocation:

- `metrics.jsonl`: one row per expected sample, SCD/PA/SR, part errors and matching,
  fixed scale divisor, shared alignment and candidate diagnostics, final episode
  SHA-256, source revision, protocol version, similarity configuration and group diagnostics.
- `metrics_summary.json`: sample macro averages, expected/scored/error counts,
  explicit scored-sample denominator, errors and full protocol/input identity.

Existing scheduler `metrics.json`, episodes, reports and traces remain unchanged.
A missing/corrupt episode or mismatched identity/geometry produces a row with
`status="error"`, null scores and an error message. Other samples continue. The
summary is `incomplete` and the CLI exits with status 1 if any sample fails; no
failed sample is silently omitted from the expected count. With zero scored
samples the averages are null. A complete run exits with status 0.

### Interactive metric inspection

`notebooks/inspect_metrics.ipynb` is a read-only, step-by-step inspection notebook.
Install its optional environment with `uv sync --extra episodes --group inspection`
and select `.venv/bin/python` as the notebook kernel. Set `LOG` and `SAMPLE_ID` in
the first cell, then Run All. Optional globals select an HF cache, an individual
part, a diagnostic alignment candidate, `SIMILARITY_POLICY`, `SIMILARITY_THRESHOLD`,
and `SIMILARITY_PAIR`. `COMPARE_RUN` enables the full-run read-only comparison.

The notebook imports the evaluator's shared input reconstruction, alignment and
metric functions. It displays actual meshes before/after alignment, directional
SCD distances, source equivalence annotations, the complete part cost matrix and
allowed Hungarian assignments, per-part threshold checks, and a selected pair's
nearest neighbors. Candidate transforms are optionally returned by `align` for
inspection; the default evaluator and its selection objective remain unchanged.

The notebook also displays source/geometry groups, aligned input-pair CD matrices,
threshold edges, transitive-closure diagnostics, and pair clouds before/after rigid
registration with shared axes and relative size preserved. Both assignment policies
are compared at the same global transform. Unrestricted matching and alternative
alignment candidates remain explicitly labeled counterfactual diagnostics.

`compare_run` reconstructs all saved samples at the persisted geometry alignments,
validates hashes and recalculates both policies without writing files. It displays
sample means, changes, group diagnostics and errors in the notebook. Run the current
geometry evaluator first; this comparison requires its persisted groups and transforms.
Clear notebook outputs before saving/sharing: reconstructed GT stays in memory,
not in the committed notebook. Existing metric and episode hashes are checked after
execution.
