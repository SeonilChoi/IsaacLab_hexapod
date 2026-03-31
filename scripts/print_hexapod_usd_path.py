#!/usr/bin/env python3
"""Print expected hexapod.usd path in this repo (no Isaac Sim / isaaclab required)."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEV_USD = (
    _REPO_ROOT / "source" / "hexapod" / "hexapod" / "assets" / "data" / "robots" / "hexapod.usd"
).resolve()

print(f"Dev-tree USD: {_DEV_USD}")
print(f"exists: {_DEV_USD.is_file()}")
print()
print(
    "At runtime, hexapod uses ISAACLAB_ASSETS_DATA_DIR from the installed package "
    "(see hexapod/assets/__init__.py). To print that path inside Isaac, run a task that "
    "imports HEXAPOD_CFG (e.g. Template-HexapodWalk-Direct-v0) with:\n"
    "  HEXAPOD_DEBUG_USD_PATH=1 python scripts/skrl/train.py ...\n"
    "Output goes to stderr with flush so it is visible in Kit consoles."
)
