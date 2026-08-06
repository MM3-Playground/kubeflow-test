from __future__ import annotations

import argparse
import os
from getpass import getpass

from prefect.blocks.system import Secret


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update a Prefect Secret block for an HTTPS Git token.")
    parser.add_argument("block_name", help="Prefect Secret block name, e.g. github-read-token")
    parser.add_argument(
        "--env",
        dest="env_name",
        help="Read the token from this environment variable instead of prompting.",
    )
    args = parser.parse_args()

    token = os.environ.get(args.env_name, "") if args.env_name else getpass("HTTPS Git token: ")
    if not token:
        raise SystemExit("Token is empty")

    Secret(value=token).save(args.block_name, overwrite=True)
    print(f"Saved Prefect Secret block: {args.block_name}")


if __name__ == "__main__":
    main()
