#!/usr/bin/env python3
"""Build an HDF5 mimic dataset from paired *.npy trajectories (matches prior dataset layout where possible).

Numpy inputs live under ``raw_data/`` by default (``--dir``). Output HDF5 defaults to
``dataset.hdf5`` next to ``raw_data/`` (the ``hexapod`` dataset folder).

For each index from ``--indices`` (default ``auto``: all ``<id>_action.npy`` in ``--dir``):
  - ``{i}_action.npy``   shape (T, 18) joint targets/radians (npy order; stored after
        ``+ JOINT_POS_OFFSET`` then ``JOINT_REORDER`` / ``JOINT_DIRECTION``; see script)
  - ``{i}_obs.npy``      shape (T, 48) with layout:
        [0:6]   pose (stored pose z += ``POSE_Z_OFFSET``; see script)
        [6:12]  twist
        [12:30] joint_position (npy order; same transform as actions)
        [30:48] joint_velocity (npy order; reorder + direction only, no offset)
  - ``{i}_condition.npy`` shape (T, 3) with:
        [0:2] command
        [2]   progress (stored as shape (T, 1))

Typical demos: **T=1000 rows = 10 s** at **100 Hz** — set ``HexapodAmpMimicEnvCfg.motion_fps`` to that rate (default 100).

Writes:
  data/demo_{k}/actions
  data/demo_{k}/obs/{pose,twist,joint_pos,joint_vel,command,progress,actions}
  data/demo_{k}/initial_state/articulation/robot/{joint_position,joint_velocity,root_pose,root_velocity}

HDF5 attrs: ``source``, ``indices``, ``num_demos`` (for loaders / debugging).

Usage:
  python build_dataset_from_npy.py
  python build_dataset_from_npy.py -o ./dataset.hdf5 --indices auto
  python build_dataset_from_npy.py --dir ./raw_data --indices 1-50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# .../hexapod/mimic/dataset/hexapod/
_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT_DIR = _PACKAGE_DIR / "raw_data"
_DEFAULT_OUTPUT_HDF5 = _PACKAGE_DIR / "dataset.hdf5"

# Applied to obs[:, 12:30] joint_position and to actions (rad, 18 DOF, same order as npy).
JOINT_POS_OFFSET = np.array(
    [
        0.0,
        0.6981317007977318,
        -2.443460952792061,
        0.0,
        0.6981317007977318,
        -2.443460952792061,
        0.0,
        0.6981317007977318,
        -2.443460952792061,
        0.0,
        -0.6981317007977318,
        2.443460952792061,
        0.0,
        -0.6981317007977318,
        2.443460952792061,
        0.0,
        -0.6981317007977318,
        2.443460952792061,
    ],
    dtype=np.float32,
)

JOINT_REORDER = np.array([15, 12, 9, 6, 3, 0, 16, 13, 10, 7, 4, 1, 17, 14, 11, 8, 5, 2], dtype=np.intp)

JOINT_DIRECTION = np.array(
    [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1],
    dtype=np.float32,
)

POSE_Z_OFFSET = np.float32(-0.7788)


def _apply_joint_reorder_and_direction(block: np.ndarray) -> np.ndarray:
    """(T, 18) npy-order joints -> robot/sim order, element-wise ``* JOINT_DIRECTION``."""
    if block.ndim != 2 or block.shape[1] != JOINT_REORDER.shape[0]:
        raise ValueError(f"expected (T, {JOINT_REORDER.shape[0]}), got {block.shape}")
    out = block[:, JOINT_REORDER] * JOINT_DIRECTION
    return out.astype(np.float32)


def parse_indices(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def discover_trajectory_indices(directory: Path) -> list[int]:
    """Sorted integer ids from ``{id}_action.npy`` in ``directory``."""
    suf = "_action.npy"
    found: set[int] = set()
    if not directory.is_dir():
        return []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.endswith(suf):
            continue
        stem = name[: -len(suf)]
        if stem.isdigit():
            found.add(int(stem))
    return sorted(found)


def resolve_indices(spec: str, directory: Path) -> list[int]:
    if spec.strip().lower() == "auto":
        ids = discover_trajectory_indices(directory)
        if not ids:
            sample = sorted(p.name for p in directory.iterdir() if p.is_file())[:25] if directory.is_dir() else []
            extra = f" Files there: {sample}" if sample else " (directory missing or empty)"
            raise ValueError(
                f"No '<id>_action.npy' under {directory}.{extra} "
                f"Use --dir path/to/raw_data or --indices 1-10."
            )
        return ids
    return parse_indices(spec)


def write_demo(
    h5: h5py.File,
    demo_name: str,
    actions: np.ndarray,
    obs: np.ndarray,
    condition: np.ndarray,
) -> None:
    if obs.ndim != 2 or obs.shape[1] < 48:
        raise ValueError(f"{demo_name}: obs must be (T, >=48), got {obs.shape}")
    if condition.ndim != 2 or condition.shape[1] < 3:
        raise ValueError(f"{demo_name}: condition must be (T, >=3), got {condition.shape}")
    t = actions.shape[0]
    if obs.shape[0] != t or condition.shape[0] != t:
        raise ValueError(
            f"{demo_name}: length mismatch T actions={actions.shape[0]} obs={obs.shape[0]} cond={condition.shape[0]}"
        )

    pose = obs[:, 0:6].astype(np.float32)
    pose[:, 2] += POSE_Z_OFFSET
    twist = obs[:, 6:12].astype(np.float32)
    jp_raw = obs[:, 12:30].astype(np.float32)
    if jp_raw.shape[1] != JOINT_POS_OFFSET.shape[0]:
        raise ValueError(
            f"{demo_name}: joint block must have {JOINT_POS_OFFSET.shape[0]} dims, got {jp_raw.shape[1]}"
        )
    joint_pos = _apply_joint_reorder_and_direction(jp_raw + JOINT_POS_OFFSET)
    joint_vel = _apply_joint_reorder_and_direction(obs[:, 30:48].astype(np.float32))
    command = condition[:, 0:2].astype(np.float32)
    progress = condition[:, 2:3].astype(np.float32)

    actions_raw = actions.astype(np.float32)
    if actions_raw.shape[1] != JOINT_POS_OFFSET.shape[0]:
        raise ValueError(
            f"{demo_name}: actions must be (T, {JOINT_POS_OFFSET.shape[0]}) for joint-angle offset, got {actions_raw.shape}"
        )
    actions_f = _apply_joint_reorder_and_direction(actions_raw + JOINT_POS_OFFSET)

    g = h5.require_group(f"data/{demo_name}")
    g.create_dataset("actions", data=actions_f, compression="gzip", compression_opts=4)

    og = g.require_group("obs")
    og.create_dataset("pose", data=pose, compression="gzip", compression_opts=4)
    og.create_dataset("twist", data=twist, compression="gzip", compression_opts=4)
    og.create_dataset("joint_pos", data=joint_pos, compression="gzip", compression_opts=4)
    og.create_dataset("joint_vel", data=joint_vel, compression="gzip", compression_opts=4)
    og.create_dataset("command", data=command, compression="gzip", compression_opts=4)
    og.create_dataset("progress", data=progress, compression="gzip", compression_opts=4)
    og.create_dataset("actions", data=actions_f, compression="gzip", compression_opts=4)

    init = g.require_group("initial_state/articulation/robot")
    init.create_dataset("joint_position", data=joint_pos[0:1], compression="gzip", compression_opts=4)
    init.create_dataset("joint_velocity", data=joint_vel[0:1], compression="gzip", compression_opts=4)
    init.create_dataset("root_pose", data=pose[0:1], compression="gzip", compression_opts=4)
    init.create_dataset("root_velocity", data=twist[0:1], compression="gzip", compression_opts=4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HDF5 mimic dataset from npy trajectories")
    ap.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help=f"Folder with {{i}}_action/obs/condition.npy (default: {_DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output .hdf5 (default: {_DEFAULT_OUTPUT_HDF5})",
    )
    ap.add_argument(
        "--indices",
        type=str,
        default="auto",
        help="Trajectory ids: 'auto' = all <id>_action.npy in --dir; or '1-50', '1,2,5'",
    )
    ap.add_argument(
        "--demo-prefix",
        type=str,
        default="demo_",
        help="HDF5 group name prefix (demo_ -> demo_0, demo_1, ...)",
    )
    args = ap.parse_args()

    root = args.dir.resolve()
    out = (args.output if args.output is not None else _DEFAULT_OUTPUT_HDF5).resolve()
    indices = resolve_indices(args.indices, root)

    with h5py.File(out, "w") as h5:
        h5.attrs["source"] = "build_dataset_from_npy.py"
        h5.attrs["indices"] = ",".join(str(i) for i in indices)
        h5.attrs["num_demos"] = len(indices)

        for k, idx in enumerate(indices):
            demo_name = f"{args.demo_prefix}{k}"
            act_p = root / f"{idx}_action.npy"
            obs_p = root / f"{idx}_obs.npy"
            cond_p = root / f"{idx}_condition.npy"
            for p in (act_p, obs_p, cond_p):
                if not p.exists():
                    raise FileNotFoundError(f"Missing: {p}")

            actions = np.load(act_p)
            obs = np.load(obs_p)
            condition = np.load(cond_p)
            write_demo(h5, demo_name, actions, obs, condition)

    print(f"Wrote {out} with {len(indices)} demos ({args.demo_prefix}0 .. {args.demo_prefix}{len(indices) - 1})")


if __name__ == "__main__":
    main()
