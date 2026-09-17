#!/usr/bin/env python3
"""Recover interrupted human-teacher staging folders as rejected attempts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soarm_nbv.hazard_episode_collector import recover_interrupted_attempts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    recovered = recover_interrupted_attempts(args.root.expanduser().resolve())
    for path in recovered:
        print(f">>> [hazard_collect] recovered interrupted attempt -> {path}")


if __name__ == "__main__":
    main()
