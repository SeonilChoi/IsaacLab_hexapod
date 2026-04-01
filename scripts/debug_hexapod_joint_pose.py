# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn one hexapod and verify joint targets (Isaac Sim).

**Static pose** — hold one set of angles::

  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --preset mimic-offset
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --joints 0,0.2,-0.3,...
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --user '[0.1, 0.2, 0.3, ...]'

**Playback** — step through ``data/<demo>/actions`` from the mimic HDF5 (same layout as
``build_dataset_from_npy.py``)::

  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --playback
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --playback --demo demo_1 --hdf5 /path/to/dataset.hdf5
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --playback --compare-obs

``--preset mimic-offset`` uses ``JOINT_POS_OFFSET`` from ``build_dataset_from_npy.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HDF5 = _REPO_ROOT / "source/hexapod/hexapod/mimic/dataset/hexapod/dataset.hdf5"

parser = argparse.ArgumentParser(description="Hexapod joint inspect: static pose or HDF5 action playback.")
parser.add_argument(
    "--playback",
    action="store_true",
    help="Play data/<demo>/actions from --hdf5 (one row per env step).",
)
parser.add_argument(
    "--hdf5",
    type=Path,
    default=_DEFAULT_HDF5,
    help=f"HDF5 dataset (default: {_DEFAULT_HDF5})",
)
parser.add_argument(
    "--demo",
    type=str,
    default="demo_0",
    help="Demo group name under data/, e.g. demo_0",
)
parser.add_argument(
    "--loop",
    action="store_true",
    help="With --playback: loop trajectory; otherwise hold last frame after end.",
)
parser.add_argument(
    "--compare-obs",
    action="store_true",
    help="With --playback: if obs/joint_pos exists, print max|q_sim - q_dataset| each print-every.",
)
parser.add_argument(
    "--preset",
    type=str,
    default="zero",
    choices=("zero", "mimic-offset"),
    help="Static mode: joint preset (ignored for --playback initial row uses actions[0]).",
)
parser.add_argument(
    "--joints",
    type=str,
    default=None,
    help="Static mode: comma-separated joint angles (rad). Must be 18 values for hexapod.",
)
parser.add_argument(
    "--user",
    type=str,
    default=None,
    help=(
        "Static mode: arbitrary joint vector, e.g. '[0.1, 0.2, 0.3, ...]' or '0.1,0.2,0.3' (rad). "
        "Length must match robot DOF. Quote for shells. Overrides --preset; cannot combine with --joints."
    ),
)
parser.add_argument(
    "--print-every",
    type=int,
    default=120,
    help="Print diagnostics every N env steps (0 = disable).",
)
parser.add_argument(
    "--stiffness",
    type=float,
    default=None,
    help="Optional: ImplicitActuator stiffness (default 400).",
)
parser.add_argument(
    "--damping",
    type=float,
    default=None,
    help="Optional: ImplicitActuator damping (default 40).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch

from hexapod.mimic.hexapod_joint_inspect_env import (
    HEXAPOD_JOINT_INSPECT_CFG,
    HexapodJointInspectEnv,
    HexapodJointInspectEnvCfg,
    joint_pos_tracking_error,
)
from isaaclab.actuators import ImplicitActuatorCfg


def _parse_joints(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(float(x) for x in parts)


def _parse_user_joint_vector(s: str) -> tuple[float, ...]:
    """Parse ``[a, b, c]``, ``a, b, c``, or mixed spacing into floats."""
    t = s.strip()
    if t.startswith("["):
        t = t[1:]
    if t.endswith("]"):
        t = t[:-1]
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if not parts:
        raise ValueError("--user: empty list after parsing")
    return tuple(float(x) for x in parts)


def load_demo_actions_and_optional_ref(hdf5_path: Path, demo: str) -> tuple[np.ndarray, np.ndarray | None]:
    hdf5_path = hdf5_path.resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f or demo not in f["data"]:
            raise KeyError(f"Missing data/{demo} in {hdf5_path}")
        g = f["data"][demo]
        if "actions" not in g:
            raise KeyError(f"data/{demo}/actions missing in {hdf5_path}")
        actions = np.asarray(g["actions"][...], dtype=np.float32)
        if actions.shape[0] < 1:
            raise ValueError(f"data/{demo}/actions is empty")
        joint_ref = None
        if "obs" in g and isinstance(g["obs"], h5py.Group) and "joint_pos" in g["obs"]:
            joint_ref = np.asarray(g["obs"]["joint_pos"][...], dtype=np.float32)
            if joint_ref.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"actions T={actions.shape[0]} vs obs/joint_pos T={joint_ref.shape[0]} (mismatch)"
                )
    return actions, joint_ref


def main() -> None:
    actions_traj: np.ndarray | None = None
    joint_ref: np.ndarray | None = None

    if args_cli.playback:
        actions_traj, joint_ref = load_demo_actions_and_optional_ref(args_cli.hdf5, args_cli.demo)
        joint_tuple = tuple(float(x) for x in actions_traj[0])
        print(f"[INFO] Playback: {args_cli.hdf5}  data/{args_cli.demo}/actions  T={actions_traj.shape[0]} dof={actions_traj.shape[1]}")
        if joint_ref is not None:
            print(f"[INFO] Found obs/joint_pos for optional --compare-obs (T={joint_ref.shape[0]})")
        elif args_cli.compare_obs:
            print("[WARN] --compare-obs requested but obs/joint_pos not in this demo; ignoring compare.")
    else:
        if args_cli.user is not None and args_cli.joints is not None:
            raise SystemExit("Use only one of --user or --joints (not both).")

        if args_cli.user is not None:
            try:
                joint_tuple = _parse_user_joint_vector(args_cli.user)
            except ValueError as e:
                raise SystemExit(str(e)) from e
        elif args_cli.joints is not None:
            joint_tuple = _parse_joints(args_cli.joints)
            if len(joint_tuple) != 18:
                raise SystemExit(f"--joints must provide 18 values, got {len(joint_tuple)}")
        elif args_cli.preset == "zero":
            joint_tuple = tuple([0.0] * 18)
        else:
            from hexapod.mimic.dataset.hexapod.build_dataset_from_npy import JOINT_POS_OFFSET

            joint_tuple = tuple(float(x) for x in JOINT_POS_OFFSET.tolist())

    robot_cfg = HEXAPOD_JOINT_INSPECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    if args_cli.stiffness is not None or args_cli.damping is not None:
        st = 400.0 if args_cli.stiffness is None else args_cli.stiffness
        dm = 40.0 if args_cli.damping is None else args_cli.damping
        robot_cfg = robot_cfg.replace(
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness={".*": st},
                    damping={".*": dm},
                    velocity_limit={".*": 100.0},
                ),
            },
        )

    cfg = HexapodJointInspectEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot = robot_cfg
    cfg.inspect_joint_pos = joint_tuple
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device

    use_cam = getattr(args_cli, "enable_cameras", False)
    env = HexapodJointInspectEnv(cfg, render_mode="rgb_array" if use_cam else None)
    env.reset()

    names = env.robot.data.joint_names
    print("[INFO] Joint order (index: name) — HDF5 action columns follow this order:")
    for i, n in enumerate(names):
        print(f"  {i:2d}  {n}")
    if not args_cli.playback:
        print("[INFO] Target joint pos (rad):", [round(x, 6) for x in joint_tuple])

    n_dof = len(names)
    if actions_traj is not None and actions_traj.shape[1] != n_dof:
        raise SystemExit(f"HDF5 actions dim {actions_traj.shape[1]} != robot DOFs {n_dof}")

    T = int(actions_traj.shape[0]) if actions_traj is not None else 0
    step = 0
    compare_obs = bool(args_cli.compare_obs and joint_ref is not None)

    while simulation_app.is_running():
        with torch.inference_mode():
            if actions_traj is not None:
                if args_cli.loop:
                    ti = step % T
                else:
                    ti = min(step, T - 1)
                env.set_joint_targets(actions_traj[ti])

            err = joint_pos_tracking_error(env)
            if args_cli.print_every and step % args_cli.print_every == 0:
                q = env.robot.data.joint_pos[0].cpu().tolist()
                msg = f"[step {step}] max|q-q_cmd|={err[0].item():.5f} rad  q[0:3]={[round(q[i], 4) for i in range(min(3, len(q)))]}"
                if compare_obs and actions_traj is not None:
                    ti = step % T if args_cli.loop else min(step, T - 1)
                    ref = torch.tensor(joint_ref[ti], device=env.device, dtype=torch.float32)
                    q_t = env.robot.data.joint_pos[0]
                    ds = (q_t - ref).abs().max().item()
                    msg += f"  max|q_sim-q_dataset|={ds:.5f}"
                print(msg)

            actions = torch.zeros((1, n_dof), device=env.device)
            env.step(actions)
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
