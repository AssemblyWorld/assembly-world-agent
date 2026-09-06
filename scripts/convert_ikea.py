"""Convert selected IKEA samples into configuration-addressed initial episodes."""

import sys

from assembly_world_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["convert-ikea", *sys.argv[1:]]))
