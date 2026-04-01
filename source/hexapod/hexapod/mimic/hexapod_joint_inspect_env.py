# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

"""Single-hexapod scene to verify joint ordering and absolute joint targets (no HDF5 / AMP)."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from hexapod.assets.robots.hexapod import HEXAPOD_CFG

# Stiff enough for ``set_joint_position_target`` to hold a pose under gravity during inspection.
HEXAPOD_JOINT_INSPECT_CFG = HEXAPOD_CFG.replace(
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={".*": 400.0},
            damping={".*": 40.0},
            velocity_limit={".*": 100.0},
        ),
    },
)


@configclass
class HexapodJointInspectEnvCfg(DirectRLEnvCfg):
    """One env, plane, absolute joint targets (rad) each step."""

    episode_length_s = 1.0e6
    decimation = 2

    observation_space = 1
    action_space = 18
    state_space = 0

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
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot: ArticulationCfg = HEXAPOD_JOINT_INSPECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    inspect_joint_pos: tuple[float, ...] = tuple([0.0] * 18)
    """Absolute joint targets (rad), same order as ``Articulation.data.joint_names``."""


class HexapodJointInspectEnv(DirectRLEnv):
    cfg: HexapodJointInspectEnvCfg

    def __init__(self, cfg: HexapodJointInspectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self._dof_lower = dof_lower.unsqueeze(0)
        self._dof_upper = dof_upper.unsqueeze(0)

        n_dof = len(self.robot.data.joint_names)
        if len(cfg.inspect_joint_pos) != n_dof:
            raise ValueError(
                f"inspect_joint_pos has {len(cfg.inspect_joint_pos)} values but robot has {n_dof} DOFs"
            )
        t = torch.tensor(cfg.inspect_joint_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._joint_target = torch.clamp(t, self._dof_lower, self._dof_upper)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        self.robot.set_joint_position_target(self._joint_target)

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = self.scene.env_origins[env_ids]
        joint_pos = self._joint_target.expand(len(env_ids), -1).clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def set_joint_targets(self, positions_rad: list[float] | np.ndarray | torch.Tensor) -> None:
        """Update clamped targets (same DOF order as ``joint_names``)."""
        t = torch.as_tensor(positions_rad, dtype=torch.float32, device=self.device).view(1, -1)
        n = len(self.robot.data.joint_names)
        if t.shape[1] != n:
            raise ValueError(f"Expected {n} joint values, got {t.shape[1]}")
        self._joint_target = torch.clamp(t, self._dof_lower, self._dof_upper)


def joint_pos_tracking_error(env: HexapodJointInspectEnv) -> torch.Tensor:
    """L2 error per env between commanded and measured joint positions."""
    return (env.robot.data.joint_pos - env._joint_target).abs().max(dim=-1).values
