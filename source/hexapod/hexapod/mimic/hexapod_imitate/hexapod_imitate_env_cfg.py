# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RAW_DATA = os.path.normpath(os.path.join(_PACKAGE_DIR, "..", "dataset", "hexapod", "raw_data"))
_DEFAULT_TARGET_DIR = os.path.join(_DEFAULT_RAW_DATA, "1")
_DEFAULT_JOINT_LIMITS_NPZ = os.path.normpath(
    os.path.join(_PACKAGE_DIR, "..", "dataset", "hexapod", "raw_data", "imitation_joint_limits.npz")
)

from hexapod.mimic.hexapod_joint_inspect_env import HEXAPOD_JOINT_INSPECT_CFG


@configclass
class HexapodImitateEnvCfg(DirectRLEnvCfg):
    """Hexapod imitation (PPO): position targets from policy in [-1,1] mapped to demo ``q_min..q_max``."""

    episode_length_s = 20.0
    decimation = 2

    # 3 pos + 3 rpy + 3 vlin + 3 wang + 18 jp + 18 jv + 18 a_t + 18 a_{t-1} + 2 cmd + 1 prog
    observation_space = 87
    action_space = 18
    state_space = 0

    imitation_joint_limits_npz: str = _DEFAULT_JOINT_LIMITS_NPZ
    """``.npz`` with ``q_min``, ``q_max`` (18,) from ``compute_imitation_joint_limits.py``."""

    imitate_target_dirs: tuple[str, ...] = (_DEFAULT_TARGET_DIR,)
    """Used when ``imitate_target_scan_parent`` is empty: trajectory folders (full ``target_*.npy`` each)."""

    imitate_target_scan_parent: str | None = _DEFAULT_RAW_DATA
    """Default: scan this directory for subfolders that contain every ``target_*.npy`` (skips incomplete dirs). Set to empty string / disable in Hydra to use ``imitate_target_dirs`` only."""

    target_match_w_progress: float = 1.0
    target_match_w_pos_xy: float = 0.0
    target_match_w_direction: float = 0.0
    """Weights for nearest-row lookup (increase pos/dir to use current pose + command more)."""

    progress_cycle_steps: int = 100
    """Episode step modulo for synthetic progress: 0.01 * ((buf % cycle) + 1), wraps 1.0 → 0.01."""

    # Imitation reward shaping (see ``HexapodImitateEnv._get_rewards``).
    rew_exp_pose_coef: float = 200.0
    rew_exp_ori_coef: float = 20.0
    rew_exp_vlin_coef: float = 8.0
    rew_exp_vang_coef: float = 2.0
    rew_action_penalty_scale: float = 1.0e-3
    """Penalty on ``||clip(action,-1,1)||`` (replaces former torque penalty)."""
    rew_joint_acc_penalty_scale: float = 1.0e-6
    rew_survival: float = 1.0

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )
    """Per-link net contact forces; ``prim_path`` must list rigid-body prims (see joint-inspect env). Spawn uses ``activate_contact_sensors``."""

    imitation_contact_body_indices: tuple[int, ...] = ()
    """If length 6, use these as body indices into :class:`ContactSensor` data (overrides names)."""

    imitation_contact_body_names: tuple[str, ...] = ("L13", "L23", "L33", "L43", "L53", "L63")
    """Six tip link names matched on the contact sensor when indices are empty. Empty tuple disables contact term."""

    imitation_contact_force_threshold: float = 1.0
    """Norm of net contact force (N) above which the foot counts as in contact."""

    termination_height: float = -0.3
    """Episode ends when base (reference body) world Z is below this value."""

    max_abs_base_roll: float = 0.8
    max_abs_base_pitch: float = 0.8
    """Terminate if |roll| or |pitch| (rad, from base quat wxyz) exceeds this."""

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot: ArticulationCfg = HEXAPOD_JOINT_INSPECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
