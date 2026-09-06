"""Export a recorded episode as MP4 and/or GIF without executing its calls."""

import argparse
from pathlib import Path

from assembly_world_agent.artifacts import experiment
from assembly_world_agent.vis import render_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Output filename stem; defaults to the run directory"
    )
    parser.add_argument("--format", action="append", choices=["mp4", "gif"])
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-page", type=float, default=1.0)
    parser.add_argument("--max-trace-frames", type=int, default=24)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    args = parser.parse_args()
    with experiment("episode-replay", logs=args.logs, inputs={"episode": str(args.episode)}) as (
        run,
        meta,
        metrics,
    ):
        output = args.output or run / args.episode.name.removesuffix(".episode.zip")
        metrics.update(
            render_episode(
                args.episode,
                output,
                formats=args.format or ("mp4", "gif"),
                fps=args.fps,
                seconds_per_page=args.seconds_per_page,
                max_trace_frames=args.max_trace_frames,
                ffmpeg=args.ffmpeg,
            )
        )
        print(metrics["outputs"], flush=True)
    print(run)


if __name__ == "__main__":
    main()
