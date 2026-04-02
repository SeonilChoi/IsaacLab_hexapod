#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Scan ``raw_data/<id>/`` for joint angles and save per-DOF min/max for imitation policy mapping.

Reads, per subfolder:

- ``joint_position.npy`` (T, 18) if present
- ``target_joint_position.npy`` (T, 18) if present (same sim joint order as the robot)

Both may exist; all rows are concatenated. Data should already be in sim frame (offsets applied at export).

Writes ``imitation_joint_limits.npz`` with arrays ``q_min``, ``q_max`` shape (18,) float32.

Usage:
  python compute_imitation_joint_limits.py
  python compute_imitation_joint_limits.py --root ./raw_data -o ./imitation_joint_limits.npz --margin 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = _PACKAGE_DIR / "raw_data"
_DEFAULT_OUT = _PACKAGE_DIR / "raw_data" / "imitation_joint_limits.npz"


def collect_joint_rows(raw_root: Path) -> np.ndarray:
    raw_root = raw_root.resolve()
    chunks: list[np.ndarray] = []
    for sub in sorted(raw_root.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        jp = sub / "joint_position.npy"
        tjp = sub / "target_joint_position.npy"
        if jp.is_file():
            chunks.append(np.load(jp).astype(np.float32))
        if tjp.is_file():
            chunks.append(np.load(tjp).astype(np.float32))
    if not chunks:
        raise FileNotFoundError(
            f"No joint_position.npy or target_joint_position.npy under subfolders of {raw_root}"
        )
    return np.concatenate(chunks, axis=0)


def compute_limits(rows: np.ndarray, margin_rad: float) -> tuple[np.ndarray, np.ndarray]:
    if rows.ndim != 2 or rows.shape[1] != 18:
        raise ValueError(f"expected (N, 18), got {rows.shape}")
    q_min = rows.min(axis=0).astype(np.float32)
    q_max = rows.max(axis=0).astype(np.float32)
    span = np.maximum(q_max - q_min, 1e-4)
    m = np.float32(margin_rad)
    q_min = q_min - m * span
    q_max = q_max + m * span
    return q_min, q_max


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute imitation joint q_min/q_max from raw_data demos")
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT, help="Folder containing trajectory subdirs")
    ap.add_argument("-o", "--output", type=Path, default=_DEFAULT_OUT, help="Output .npz path")
    ap.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Fraction of per-joint span to expand min/max (avoids degenerate range)",
    )
    args = ap.parse_args()

    rows = collect_joint_rows(args.root)
    q_min, q_max = compute_limits(rows, args.margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, q_min=q_min, q_max=q_max)
    print(
        f"Wrote {args.output} from {rows.shape[0]} rows (q_min..q_max per joint rad, margin={args.margin})\n"
        f"  q_min: {q_min}\n  q_max: {q_max}"
    )


if __name__ == "__main__":
    main()
