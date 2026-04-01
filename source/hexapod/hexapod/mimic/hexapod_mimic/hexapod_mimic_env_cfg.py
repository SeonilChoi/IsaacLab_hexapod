# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from hexapod.mimic.hexapod_joint_inspect_env import HEXAPOD_JOINT_INSPECT_CFG

# mimic/dataset/hexapod/ — one level up from hexapod_mimic/
_MIMIC_DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "hexapod")
)
_DEFAULT_MOTION_FILE = os.path.join(_MIMIC_DATASET_DIR, "dataset.hdf5")


@configclass
class HexapodAmpMimicEnvCfg(DirectRLEnvCfg):
    """Hexapod AMP-style environment using HDF5 demos (pose/twist + joints in ``obs``)."""

    episode_length_s = 10.0
    decimation = 2

    observation_space = 49
    action_space = 18
    state_space = 0

    # Policy outputs (e.g. tanh in [-1, 1]): target_joint = default_joint_pos + scale * action
    action_position_scale: float = 0.15
    """Radians per unit action added to default joint positions (then clamped to soft limits)."""

    num_amp_observations = 2
    amp_observation_space = 49

    motion_file: str = _DEFAULT_MOTION_FILE
    """Path to HDF5 with ``obs/joint_pos``, ``joint_vel``, ``pose`` (T,6), ``twist`` (T,6)."""

    motion_fps: float = 100.0
    """Must match demo recording rate (e.g. 1000 frames / 10 s → 100 Hz). Used for HDF5 time ↔ frame index."""

    reference_body: str = ""

    key_body_names: tuple[str, ...] = ()

    reset_strategy: str = "random"

    reset_z_offset: float = 0.0
    """Extra world Z on root when using motion-based spawn (0 keeps root at env origin XY with motion pose)."""

    early_termination: bool = True
    termination_height: float = -0.35
    """Episode ends (reset) when reference body world Z is below this value."""

    max_abs_base_roll: float = 0.8
    """Terminate if |base roll| (rad, from root quat wxyz) exceeds this."""

    max_abs_base_pitch: float = 0.8
    """Terminate if |base pitch| (rad, from root quat wxyz) exceeds this."""

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

    # Same implicit PD as ``hexapod_joint_inspect_env.HEXAPOD_JOINT_INSPECT_CFG`` (400 / 40 / vel 100).
    robot: ArticulationCfg = HEXAPOD_JOINT_INSPECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
