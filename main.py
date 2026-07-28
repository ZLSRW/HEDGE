#!/usr/bin/env python3
"""Command-line entry point for HEDGE."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def dispatch(module_name, arguments):
    module = __import__(module_name, fromlist=["main"])
    original_arguments = sys.argv[:]
    try:
        sys.argv = [f"{Path(__file__).name} {module_name}", *arguments]
        module.main()
    finally:
        sys.argv = original_arguments
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="HEDGE: heterogeneous epigenetic diffusion-and-gating encoder."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-data", add_help=False, help="Validate all packaged data tables.")
    commands.add_parser("train", add_help=False, help="Run five-fold cross-validation.")
    commands.add_parser("ratio", add_help=False, help="Run reduced-supervision experiments.")
    commands.add_parser("search", add_help=False, help="Run the HEDGE hyperparameter search.")
    arguments, remainder = parser.parse_known_args()
    modules = {
        "validate-data": "validate_data",
        "train": "cv_5fold",
        "ratio": "cv_5fold_ratio",
        "search": "grid_search",
    }
    return dispatch(modules[arguments.command], remainder)


if __name__ == "__main__":
    raise SystemExit(main())
