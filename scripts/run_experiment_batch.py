"""Launch an explicit JSON experiment batch plan, with no automatic retries."""

import argparse
import asyncio
import json

from assembly_world_agent.experiments.batch import launch_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(launch_batch(args.plan)), indent=2))
