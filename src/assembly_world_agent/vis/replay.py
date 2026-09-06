"""Render recorded states, original observations and paginated MCP records."""

from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..episodes import CONTRACT_DIRECTORY, ENGINE, sha256

WIDTH, HEIGHT = 1600, 900


def read_episode(path):
    """Validate the public archive without extracting resources to the filesystem."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("Unsafe archive member")
        files = {name: archive.read(name) for name in names}
    manifest = json.loads(files["manifest.json"])
    if manifest["format"] != "3dwebagent-episode" or manifest["version"] != 1:
        raise ValueError("Unsupported episode format")
    if manifest["engine"] != ENGINE:
        raise ValueError(f"Expected MuJoCo {ENGINE}")
    if set(files) != {"manifest.json", *manifest["hashes"]}:
        raise ValueError("Archive hash inventory differs")
    for name, digest in manifest["hashes"].items():
        if sha256(files[name]) != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    rows = [json.loads(line) for line in files["frames.jsonl"].splitlines()]
    size = manifest["stateSize"]
    if size <= 0 or len(files["frames.bin"]) != len(rows) * size * 8:
        raise ValueError("Invalid binary state size")
    states = np.frombuffer(files["frames.bin"], dtype="<f8").reshape(len(rows), size)
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite recorded state")
    initial, snapshots, traces = [], {}, []
    for row, state in zip(rows, states):
        frame = {k: v for k, v in row.items() if k != "kind"}
        frame["integration"] = state.tolist()
        if row["kind"] == "initial":
            initial.append(frame)
        elif row["kind"] == "state":
            if frame["index"] in snapshots:
                raise ValueError("Duplicate state index")
            snapshots[frame["index"]] = frame
        elif row["kind"] == "trace":
            traces.append(frame)
        else:
            raise ValueError("Unknown frame kind")
    if len(initial) != 1:
        raise ValueError("Expected exactly one initial frame")
    calls = [json.loads(line) for line in files["calls.jsonl"].splitlines()]
    events = [json.loads(line) for line in files["events.jsonl"].splitlines()]
    logical = dict(
        manifest=manifest,
        initial=initial[0],
        states=list(snapshots.values()),
        calls=calls,
        events=events,
        trajectory=traces,
        revision=0,
    )
    import jsonschema

    jsonschema.validate(
        logical, json.loads((CONTRACT_DIRECTORY / "episode.schema.json").read_text())
    )
    snapshots[0] = initial[0]
    for call in calls:
        if call["state_index"] not in snapshots or call["before_index"] not in snapshots:
            raise ValueError("Call references a missing state")
        if not 0 <= call["trace_start"] <= call["trace_end"] <= len(traces):
            raise ValueError("Call references an invalid trace range")
    return dict(manifest=manifest, files=files, states=snapshots, traces=traces, calls=calls)


def observation(episode, call):
    result = call.get("result")
    if not isinstance(result, dict):
        return None
    for item in result.get("content", []):
        if item.get("type") == "image" and "observation" in item:
            path = "observations/" + item["observation"]
            if path not in episode["files"]:
                raise ValueError(f"Missing observation: {path}")
            with Image.open(io.BytesIO(episode["files"][path])) as source:
                source.load()
                return source.convert("RGB")
    return None


def display_value(value):
    """Replace inline binary payloads only; preserve textual argument/result values."""
    if isinstance(value, dict):
        return {
            k: (
                "<inline image bytes; shown in scene panel>"
                if k == "data" and value.get("type") == "image"
                else display_value(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [display_value(v) for v in value]
    return value


def text_pages(call, width=45, lines=29):
    fields = [
        ("ARGUMENTS", call.get("arguments", {})),
        ("RETURN", call.get("result", {"error": call.get("error", "No result recorded")})),
    ]
    if call.get("error"):
        fields.append(("ERROR", call["error"]))
    output = []
    for title, value in fields:
        output.extend([title, ""])
        for line in json.dumps(display_value(value), indent=2, ensure_ascii=False).splitlines():
            output.extend(
                textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False)
                or [""]
            )
        output.append("")
    return [output[i : i + lines] for i in range(0, len(output), lines)] or [[]]


@lru_cache(maxsize=16)
def _font(size, mono=False):
    candidates = (
        [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ]
        if mono
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def compose(scene, episode, frame, call, page, page_index, page_count, index, total, label):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#101727")
    draw = ImageDraw.Draw(canvas)
    draw.text((26, 18), episode["manifest"]["name"][:78], font=_font(29), fill="#edf2fc")
    draw.text(
        (26, 57),
        "EPISODE REPLAY  /  Recorded states; no tool reexecution",
        font=_font(17),
        fill="#9cabca",
    )
    scene = ImageOps.contain(scene, (1024, 736), Image.Resampling.LANCZOS)
    canvas.paste(scene, (24 + (1024 - scene.width) // 2, 130 + (736 - scene.height) // 2))
    draw.rectangle((24, 96, 1048, 122), fill="#182236")
    draw.text((36, 99), label, font=_font(15), fill="#a8ddf0")
    draw.rounded_rectangle((1072, 96, 1576, 866), radius=12, fill="#1a2438")
    name = call["name"] if call else "Initial scene"
    color = "#8cdfac" if not call or call.get("status") == "completed" else "#ffac99"
    draw.text((1092, 114), name, font=_font(24), fill=color)
    details = f"{call.get('actor', 'agent')}  /  {call.get('status', '')}" if call else "setup"
    draw.text((1092, 151), details, font=_font(16), fill="#9cabca")
    if call:
        draw.text((1092, 177), str(call.get("timestamp", "")), font=_font(14), fill="#9cabca")
    for y, line in enumerate(page):
        draw.text((1092, 214 + y * 21), line, font=_font(17, True), fill="#dfe7f7")
    draw.text(
        (1092, 835),
        f"Record page {page_index + 1}/{page_count}  |  state {frame['index']}",
        font=_font(15),
        fill="#9cabca",
    )
    draw.rectangle((24, 879, 1576, 885), fill="#344059")
    draw.rectangle((24, 879, 24 + 1552 * index / max(total, 1), 885), fill="#e9b765")
    draw.text(
        (1130, 52),
        f"CALL {index:02d} / {total:02d}   |   {len(frame['groups'])} groups",
        font=_font(20),
        fill="#e9b765",
    )
    return canvas


class StateRenderer:
    """Native visualization of saved states with the recorded 38-degree agent camera."""

    def __init__(self, episode):
        import mujoco

        if mujoco.__version__ != ENGINE:
            raise ValueError(f"Expected MuJoCo {ENGINE}")
        self.mj = mujoco
        files = episode["files"]
        self.model = mujoco.MjModel.from_xml_string(
            files["world/model.xml"].decode(),
            assets={
                k[6:]: v
                for k, v in files.items()
                if k.startswith("world/") and k != "world/model.xml"
            },
        )
        self.model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
        self.model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
        self.model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
        self.data = mujoco.MjData(self.model)
        self.spec = episode["manifest"]["stateSpec"]
        if mujoco.mj_stateSize(self.model, self.spec) != episode["manifest"]["stateSize"]:
            raise ValueError("Compiled state size differs")
        self.model.vis.global_.offwidth = 1024
        self.model.vis.global_.offheight = 768
        self.renderer = mujoco.Renderer(self.model, height=768, width=1024)

    def render(self, frame):
        mj = self.mj
        mj.mj_setState(self.model, self.data, np.array(frame["integration"]), self.spec)
        mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data)
        position = np.array(frame["camera"]["position"])
        target = np.array(frame["camera"]["target"])
        forward = target - position
        length = np.linalg.norm(forward)
        if length < 1e-12:
            raise ValueError("Degenerate recorded camera")
        forward /= length
        right = np.cross(forward, [0, 0, 1])
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        for camera in self.renderer.scene.camera:
            camera.pos[:] = position
            camera.forward[:] = forward
            camera.up[:] = up
            camera.orthographic = 0
            camera.frustum_near = 0.01
            camera.frustum_far = 1000
            camera.frustum_top = 0.01 * math.tan(math.radians(19))
            camera.frustum_bottom = -camera.frustum_top
            camera.frustum_center = 0
            # Zero lets MuJoCo derive horizontal extent from the viewport aspect.
            camera.frustum_width = 0
        for light in self.renderer.scene.lights[: self.renderer.scene.nlight]:
            if light.headlight:
                light.pos[:] = position
                light.dir[:] = forward
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_CULL_FACE] = False
        pixels = self.renderer.render()
        pixels[np.all(pixels == 0, axis=2)] = [24, 24, 37]
        return Image.fromarray(pixels)

    def close(self):
        self.renderer.close()


def render_episode(
    episode,
    output,
    *,
    formats=("mp4", "gif"),
    fps=12,
    seconds_per_page=1.0,
    max_trace_frames=24,
    ffmpeg="ffmpeg",
):
    """Export a narrated-by-records visual replay. Output is a filename stem.

    Calls are paced uniformly; long JSON is paginated without truncation. Physics
    traces are sampled to max_trace_frames per call and played at fps. No states
    are interpolated. The main view always uses native rendering, including capture
    calls, to avoid switching projection and lighting between renderers. The input archive is never modified. Requires [episodes] and FFmpeg.
    """
    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats) - {"mp4", "gif"}:
        raise ValueError("Formats must be mp4 and/or gif")
    if (
        not isinstance(fps, int)
        or fps < 1
        or not math.isfinite(seconds_per_page)
        or seconds_per_page <= 0
    ):
        raise ValueError("Positive fps and seconds_per_page required")
    if not isinstance(max_trace_frames, int) or max_trace_frames < 0:
        raise ValueError("max_trace_frames must be nonnegative")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise FileNotFoundError("FFmpeg is required on PATH")
    output = Path(output)
    outputs = {f: Path(str(output) + "." + f) for f in formats}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("Output already exists")
    source = Path(episode)
    digest = sha256(source.read_bytes())
    ep = read_episode(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    pages = 0
    sampled = 0
    original_images = 0
    with tempfile.TemporaryDirectory(prefix="awa-replay-", dir=output.parent) as temporary:
        temp = Path(temporary)
        movie = temp / "replay.mp4"
        with (temp / "ffmpeg.log").open("w+b") as log:
            encoder = subprocess.Popen(
                [
                    executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{WIDTH}x{HEIGHT}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(movie),
                ],
                stdin=subprocess.PIPE,
                stderr=log,
            )
            renderer = None
            try:
                renderer = StateRenderer(ep)

                def emit(image, duration):
                    nonlocal count
                    data = image.tobytes()
                    repeats = max(1, round(duration * fps))
                    for _ in range(repeats):
                        encoder.stdin.write(data)
                    count += repeats

                initial = ep["states"][0]
                emit(
                    compose(
                        renderer.render(initial),
                        ep,
                        initial,
                        None,
                        [
                            "Initial recorded state.",
                            "",
                            "Playback progress is not an",
                            "assembly success score.",
                        ],
                        0,
                        1,
                        0,
                        len(ep["calls"]),
                        "Restored scene / recorded agent camera",
                    ),
                    seconds_per_page,
                )
                for index, call in enumerate(ep["calls"], 1):
                    text = text_pages(call)
                    pages += len(text)
                    trace = [
                        f
                        for f in ep["traces"][call["trace_start"] : call["trace_end"]]
                        if f.get("phase") == "physics"
                    ]
                    if trace and max_trace_frames:
                        selection = np.unique(
                            np.linspace(
                                0, len(trace) - 1, min(len(trace), max_trace_frames), dtype=int
                            )
                        )
                        for i in selection:
                            frame = trace[i]
                            sampled += 1
                            emit(
                                compose(
                                    renderer.render(frame),
                                    ep,
                                    frame,
                                    call,
                                    text[0],
                                    0,
                                    len(text),
                                    index,
                                    len(ep["calls"]),
                                    "Saved physics trace / sampled; time compressed",
                                ),
                                1 / fps,
                            )
                    frame = ep["states"][call["state_index"]]
                    if observation(ep, call) is not None:
                        original_images += 1
                    view = renderer.render(frame)
                    label = "Restored scene / recorded agent camera (MuJoCo rendering)"
                    for i, page in enumerate(text):
                        emit(
                            compose(
                                view,
                                ep,
                                frame,
                                call,
                                page,
                                i,
                                len(text),
                                index,
                                len(ep["calls"]),
                                label,
                            ),
                            seconds_per_page,
                        )
                encoder.stdin.close()
                if encoder.wait() != 0:
                    log.seek(0)
                    raise RuntimeError(log.read().decode())
            except BaseException:
                encoder.kill()
                encoder.wait()
                raise
            finally:
                if renderer is not None:
                    renderer.close()
        if "gif" in formats:
            filters = f"fps={min(fps, 6)},scale=1280:-1:flags=lanczos"
            for command in (
                [
                    executable,
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(movie),
                    "-vf",
                    filters + ",palettegen=stats_mode=diff",
                    str(temp / "palette.png"),
                ],
                [
                    executable,
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(movie),
                    "-i",
                    str(temp / "palette.png"),
                    "-lavfi",
                    f"[0:v]{filters}[v];[v][1:v]paletteuse=dither=sierra2_4a",
                    "-loop",
                    "0",
                    str(temp / "replay.gif"),
                ],
            ):
                subprocess.run(command, check=True, capture_output=True)
        if sha256(source.read_bytes()) != digest:
            raise ValueError("Input changed during rendering")
        for kind, path in outputs.items():
            with path.open("xb") as stream, (temp / f"replay.{kind}").open("rb") as rendered:
                shutil.copyfileobj(rendered, stream)
    return dict(
        input=str(source),
        input_sha256=digest,
        outputs={k: str(v) for k, v in outputs.items()},
        calls=len(ep["calls"]),
        states=len(ep["states"]),
        original_observations_available=original_images,
        original_observations_displayed=0,
        view_mode="consistent native rendering with recorded agent camera",
        seconds_per_page=seconds_per_page,
        record_pages=pages,
        sampled_trace_frames=sampled,
        video_frames=count,
        fps=fps,
        duration_seconds=count / fps,
        resolution=[WIDTH, HEIGHT],
        input_unchanged=True,
        timing="uniform JSON-page holds; sampled recorded physics traces; no interpolation",
    )
