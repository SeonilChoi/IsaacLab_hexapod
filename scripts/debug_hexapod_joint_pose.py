# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn one hexapod and verify joint targets (Isaac Sim).

**Static pose** — hold one set of angles::

  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --joints 0,0.2,-0.3,...
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --user '[0.1, 0.2, 0.3, ...]'

**Playback** — step through ``data/<demo>/actions`` from the mimic HDF5 (same layout as
``build_dataset_from_npy.py``)::

  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --playback
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --playback --demo demo_1 --hdf5 /path/to/dataset.hdf5
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --playback --compare-obs

**Target joint npy** — play ``target_joint_position.npy`` (``T×18`` rad, **sim joint order** as in
``robot.data.joint_names``)::

  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --target-joint-npy path/to/target_joint_position.npy
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --target-joint-npy path/to/raw_data/1
  ./isaaclab.sh -p scripts/debug_hexapod_joint_pose.py --target-joint-npy .../1 --compare-demo-joints

With ``--compare-demo-joints``, loads ``joint_position.npy`` from the same folder and prints
``max|q_sim - q_demo|`` (raw demo joints vs sim tracking of targets).

When playing from a trajectory folder, if ``target_contact.npy`` exists (``T×6``), its row is printed
each ``--print-every`` together with sim foot contact from the env's :class:`~isaaclab.sensors.ContactSensor`
(net normal force on the six bodies; default names ``L13,L23,L33,L43,L53,L63``; override with
``--contact-foot-bodies``), using ``|F|>`` ``--contact-force-threshold`` N. Requires
``activate_contact_sensors`` on spawn (``HEXAPOD_JOINT_INSPECT_CFG``).

"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HDF5 = _REPO_ROOT / "source/hexapod/hexapod/mimic/dataset/hexapod/dataset.hdf5"
_DEFAULT_TARGET_JOINT_NPY = (
    _REPO_ROOT / "source/hexapod/hexapod/mimic/dataset/hexapod/raw_data/1/target_joint_position.npy"
)

parser = argparse.ArgumentParser(description="Hexapod joint inspect: static pose, HDF5 playback, or target npy.")
parser.add_argument(
    "--target-joint-npy",
    type=Path,
    default=None,
    metavar="PATH",
    help=(
        "Play target_joint_position.npy (T,18) rad in sim joint order, or a folder containing that file "
        f"(example: {_DEFAULT_TARGET_JOINT_NPY}). Mutually exclusive with --playback and --user/--joints."
    ),
)
parser.add_argument(
    "--compare-demo-joints",
    action="store_true",
    help="With --target-joint-npy: if joint_position.npy exists beside it, print max|q_sim-q_demo| each print-every.",
)
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
    choices=("zero",),
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
parser.add_argument(
    "--contact-force-threshold",
    type=float,
    default=1.0,
    help="Sim foot contact: |net normal contact force| from ContactSensor exceeds this (N). Default 1.0.",
)
parser.add_argument(
    "--contact-foot-bodies",
    type=str,
    default="L13,L23,L33,L43,L53,L63",
    help="Comma-separated robot link names (6) matched on ContactSensor for sim_c / |F| (hexapod tip default).",
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


def _body_names_list(robot) -> list[str]:
    raw = robot.data.body_names
    if raw is None:
        return []
    try:
        first = raw[0]
        if isinstance(first, str):
            return [str(x) for x in raw]
        return [str(x) for x in first]
    except (TypeError, IndexError):
        return [str(x) for x in raw]


def _parse_contact_foot_body_names(spec: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in spec.split(",") if p.strip())
    if len(parts) != 6:
        raise ValueError(f"--contact-foot-bodies: need 6 names, got {len(parts)}: {spec!r}")
    return parts


def _foot_body_indices_for_names(
    robot, want_names: tuple[str, ...]
) -> tuple[list[int], list[str]] | tuple[None, None]:
    """Resolve six body indices by exact name, then case-insensitive match."""
    names = _body_names_list(robot)
    lower_map: dict[str, int] = {}
    for i, nm in enumerate(names):
        key = nm.lower()
        if key not in lower_map:
            lower_map[key] = i
    idx: list[int] = []
    resolved: list[str] = []
    missing: list[str] = []
    for want in want_names:
        i = None
        try:
            j = names.index(want)
            i = j
        except ValueError:
            j = lower_map.get(want.lower())
            if j is not None:
                i = j
        if i is None:
            missing.append(want)
        else:
            idx.append(i)
            resolved.append(names[i])
    if missing:
        return None, None
    return idx, resolved


def _sim_foot_contact_row(
    contact_sensor,
    foot_sensor_indices: list[int],
    threshold_n: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary (6,) and net normal contact force norms (N); env index 0. Uses :class:`ContactSensor` data."""
    data = contact_sensor.data
    forces = getattr(data, "net_forces_w", None)
    if forces is not None:
        idx_t = torch.tensor(foot_sensor_indices, device=forces.device, dtype=torch.long)
        f = forces[0, idx_t, :]
    else:
        hist = getattr(data, "net_forces_w_history", None)
        if hist is None:
            z = np.zeros(6, dtype=np.float32)
            return z, z
        idx_t = torch.tensor(foot_sensor_indices, device=hist.device, dtype=torch.long)
        f = hist[0, 0, idx_t, :]
    mag = torch.linalg.vector_norm(f, dim=-1).detach().cpu().numpy().astype(np.float32)
    binary = (mag > float(threshold_n)).astype(np.float32)
    return binary, mag


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


def _resolve_target_joint_npy_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "target_joint_position.npy"
    if not path.is_file():
        raise FileNotFoundError(f"target_joint_position.npy not found: {path}")
    return path


def load_target_joint_position_npy(path: Path) -> np.ndarray:
    path = _resolve_target_joint_npy_path(path)
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 18:
        raise ValueError(f"{path}: expected (T, 18), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"{path}: empty time dimension")
    return np.asarray(arr, dtype=np.float32)


def load_target_contact_npy(folder: Path, t_len: int) -> np.ndarray | None:
    p = folder / "target_contact.npy"
    if not p.is_file():
        return None
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"{p}: expected (T, 6), got {arr.shape}")
    if int(arr.shape[0]) != int(t_len):
        raise ValueError(f"{p}: T={arr.shape[0]} != target_joint T={t_len}")
    return np.asarray(arr, dtype=np.float32)


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
    contact_traj: np.ndarray | None = None
    compare_joint_ref = False

    if args_cli.playback and args_cli.target_joint_npy is not None:
        raise SystemExit("Use only one of --playback or --target-joint-npy.")
    if args_cli.target_joint_npy is not None and (
        args_cli.user is not None or args_cli.joints is not None
    ):
        raise SystemExit("Do not combine --target-joint-npy with --user or --joints.")

    if args_cli.playback:
        actions_traj, joint_ref = load_demo_actions_and_optional_ref(args_cli.hdf5, args_cli.demo)
        joint_tuple = tuple(float(x) for x in actions_traj[0])
        print(f"[INFO] Playback: {args_cli.hdf5}  data/{args_cli.demo}/actions  T={actions_traj.shape[0]} dof={actions_traj.shape[1]}")
        if joint_ref is not None:
            print(f"[INFO] Found obs/joint_pos for optional --compare-obs (T={joint_ref.shape[0]})")
        elif args_cli.compare_obs:
            print("[WARN] --compare-obs requested but obs/joint_pos not in this demo; ignoring compare.")
        compare_joint_ref = bool(args_cli.compare_obs and joint_ref is not None)
    elif args_cli.target_joint_npy is not None:
        tpath = _resolve_target_joint_npy_path(args_cli.target_joint_npy)
        actions_traj = load_target_joint_position_npy(args_cli.target_joint_npy)
        joint_tuple = tuple(float(x) for x in actions_traj[0])
        print(
            f"[INFO] target_joint_position.npy: {tpath}  T={actions_traj.shape[0]} dof={actions_traj.shape[1]} "
            "(rad, sim / robot joint order)"
        )
        if args_cli.compare_demo_joints:
            demo_np = tpath.parent / "joint_position.npy"
            if demo_np.is_file():
                joint_ref = np.load(demo_np).astype(np.float32)
                if joint_ref.shape != actions_traj.shape:
                    raise SystemExit(
                        f"joint_position.npy shape {joint_ref.shape} != target_joint_position.npy {actions_traj.shape}"
                    )
                print(f"[INFO] compare-demo: joint_position.npy from {demo_np}")
                compare_joint_ref = True
            else:
                print(f"[WARN] --compare-demo-joints but missing {demo_np}; ignoring compare.")
        contact_traj = load_target_contact_npy(tpath.parent, int(actions_traj.shape[0]))
        if contact_traj is not None:
            print(f"[INFO] Loaded target_contact.npy (T,6) from {tpath.parent}")
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
        else:
            joint_tuple = tuple([0.0] * 18)

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

    try:
        foot_body_spec = _parse_contact_foot_body_names(args_cli.contact_foot_bodies)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    foot_idx, foot_resolved = _foot_body_indices_for_names(env.robot, foot_body_spec)
    if foot_idx is not None:
        print("[INFO] Articulation body_names indices (reference):", list(zip(foot_idx, foot_resolved)))
    else:
        bn = _body_names_list(env.robot)
        print(
            "[WARN] Could not resolve all --contact-foot-bodies "
            f"{list(foot_body_spec)} on articulation body_names; body_names={bn}"
        )

    foot_sensor_ids: list[int] | None = None
    if len(foot_body_spec) == 6:
        sids, snames = env._contact_sensor.find_bodies(list(foot_body_spec), preserve_order=True)
        if len(sids) == 6:
            foot_sensor_ids = list(sids)
            print(
                "[INFO] ContactSensor body indices (sim_c / |F|_N vs --contact-force-threshold):",
                list(zip(foot_body_spec, snames, sids)),
            )
        else:
            print(
                f"[WARN] ContactSensor matched only {len(sids)} bodies for {list(foot_body_spec)!r}; "
                f"got ids={list(sids)} names={list(snames)} — sim_c will be zeros."
            )
    else:
        print(f"[WARN] Expected 6 foot names for contact debug, got {len(foot_body_spec)}; sim_c disabled.")

    names = env.robot.data.joint_names
    print("[INFO] Joint order (index: name) — HDF5 action columns follow this order:")
    for i, n in enumerate(names):
        print(f"  {i:2d}  {n}")
    if actions_traj is None:
        print("[INFO] Target joint pos (rad):", [round(x, 6) for x in joint_tuple])

    n_dof = len(names)
    if actions_traj is not None and actions_traj.shape[1] != n_dof:
        raise SystemExit(f"HDF5 actions dim {actions_traj.shape[1]} != robot DOFs {n_dof}")

    T = int(actions_traj.shape[0]) if actions_traj is not None else 0
    step = 0

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
                ti = 0
                if actions_traj is not None:
                    ti = step % T if args_cli.loop else min(step, T - 1)
                if compare_joint_ref and actions_traj is not None and joint_ref is not None:
                    ref = torch.tensor(joint_ref[ti], device=env.device, dtype=torch.float32)
                    q_t = env.robot.data.joint_pos[0]
                    ds = (q_t - ref).abs().max().item()
                    label = "q_demo" if args_cli.target_joint_npy is not None else "q_dataset"
                    msg += f"  max|q_sim-{label}|={ds:.5f}"
                if actions_traj is not None:
                    if contact_traj is not None:
                        tc = contact_traj[ti]
                        tc_b = (tc >= 0.5).astype(np.float32)
                        msg += f"  tgt_c={tc_b.astype(int).tolist()}"
                    if foot_sensor_ids is not None:
                        sim_b, fmag = _sim_foot_contact_row(
                            env._contact_sensor,
                            foot_sensor_ids,
                            args_cli.contact_force_threshold,
                        )
                        msg += f"  sim_c={sim_b.astype(int).tolist()} |F|_N={np.round(fmag, 1).tolist()}"
                        if contact_traj is not None:
                            msg += f"  c_match={int((tc_b == sim_b).sum())}/6"
                print(msg)

            actions = torch.zeros((1, n_dof), device=env.device)
            env.step(actions)
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
