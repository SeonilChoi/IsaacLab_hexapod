#!/usr/bin/env python3
"""Inspect the structure of an HDF5 dataset file.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --file dataset.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def print_tree(h5_file: h5py.File) -> None:
    """Print group/dataset tree with shapes and dtypes."""

    def _visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            print(f"[G] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[D] {name} shape={obj.shape} dtype={obj.dtype}")

    h5_file.visititems(_visitor)


def print_summary(h5_file: h5py.File) -> None:
    """Print a small summary of demo keys and lengths."""
    if "data" not in h5_file or not isinstance(h5_file["data"], h5py.Group):
        return

    demo_keys = sorted(
        [k for k in h5_file["data"].keys() if k.startswith("demo_")],
        key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else s,
    )
    print(f"\n[Summary] demos={len(demo_keys)}")
    for key in demo_keys[:10]:
        actions = h5_file["data"][key].get("actions")
        if isinstance(actions, h5py.Dataset):
            print(f"- {key}: T={actions.shape[0]} action_dim={actions.shape[1]}")
    if len(demo_keys) > 10:
        print(f"- ... ({len(demo_keys) - 10} more demos)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset layout")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset.hdf5",
        help="Path to HDF5 file (default: ./dataset.hdf5)",
    )
    args = parser.parse_args()

    h5_path = args.file.resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {h5_path}")

    print(f"path: {h5_path}")
    print(f"size_bytes: {h5_path.stat().st_size}")

    with h5py.File(h5_path, "r") as f:
        print_tree(f)
        print_summary(f)

        if f.attrs:
            print("\n[File attrs]")
            for k, v in f.attrs.items():
                print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

