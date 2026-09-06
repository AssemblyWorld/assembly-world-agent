"""Export an experiment as a single offline interactive HTML page."""

import argparse
import json
from pathlib import Path

from assembly_world_agent.vis import export_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_results(args.run, args.output, cache_dir=args.cache_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
