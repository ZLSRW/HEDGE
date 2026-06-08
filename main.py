#!/usr/bin/env python3
"""Unified entrypoint for the ASD Crosstalk manuscript package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _dispatch(module_name: str, argv: list[str]) -> int:
    module = __import__(module_name, fromlist=["main"])
    old_argv = sys.argv[:]
    try:
        sys.argv = [f"{Path(__file__).name} {module_name}"] + argv
        module.main()
    finally:
        sys.argv = old_argv
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASD Crosstalk manuscript code entrypoint."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run 5-fold cross-validation.")
    subparsers.add_parser("ratio", help="Run train:val ratio robustness experiments.")
    subparsers.add_parser("search", help="Run hyperparameter search.")

    return parser


def main() -> int:
    parser = build_parser()
    args, remainder = parser.parse_known_args()

    if args.command == "train":
        return _dispatch("cv_5fold", remainder)
    if args.command == "ratio":
        return _dispatch("cv_5fold_ratio", remainder)
    if args.command == "search":
        return _dispatch("grid_search", remainder)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
