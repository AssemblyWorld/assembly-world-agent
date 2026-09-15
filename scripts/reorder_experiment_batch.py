"""Continue an operator-drained batch in a new explicit group order."""

import argparse
import asyncio
import json

from assembly_world_agent.experiments.batch import continue_reordered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_plan")
    parser.add_argument("destination")
    parser.add_argument("--order", nargs="+", required=True)
    parser.add_argument("--stop-after")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                continue_reordered(
                    args.source_plan, args.destination, args.order, stop_after=args.stop_after
                )
            ),
            indent=2,
        )
    )
