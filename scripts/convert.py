"""Convert an explicit selection from any registered dataset into initial episodes."""

import sys

from assembly_world_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["convert", *sys.argv[1:]]))
