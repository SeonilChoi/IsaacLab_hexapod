# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod AMP training environment driven by HDF5 reference motion (see ``HexapodHdf5MotionLoader``)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .hexapod_hdf5_motion_loader import HexapodHdf5MotionLoader
from .hexapod_mimic_env_cfg import HexapodAmpMimicEnvCfg


def _synthetic_progress_u(fi: torch.Tensor) -> torch.Tensor:
    """Scalar progress in ``[0, 1]``: step 0 → 0; then 0.01 … 1.0 cycling (after 1.0 next is 0.01).

    For integer index ``s``: 0 → 0; ``s>=1`` → ``(((s-1) mod 100) + 1) * 0.01``.
    """
    out = torch.zeros(fi.shape[0], 1, device=fi.device, dtype=torch.float32)
    m = fi > 0
    if m.any().item():
        fv = fi[m].long()
        out[m] = (((fv - 1) % 100) + 1).unsqueeze(-1).float() * 0.01
    return out


def _roll_pitch_rad_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll and pitch (rad) from body quaternion ``(w, x, y, z)`` per row."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return roll, pitch


class HexapodAmpMimicEnv(DirectRLEnv):
    cfg: HexapodAmpMimicEnvCfg

    def __init__(self, cfg: HexapodAmpMimicEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.key_body_names:
            raise ValueError(
                "HexapodAmpMimicEnv: key_body_names must be empty — the HDF5 loader only stores a single root "
                "track; extend HexapodHdf5MotionLoader before enabling key bodies."
            )

        if cfg.playback_mode:
            step_dt = float(cfg.sim.dt) * int(cfg.decimation)
            cfg.episode_length_s = float(cfg.playback_episode_length_steps) * step_dt

        super().__init__(cfg, render_mode, **kwargs)

        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self._dof_lower = dof_lower_limits.unsqueeze(0)
        self._dof_upper = dof_upper_limits.unsqueeze(0)

        self._motion_loader = HexapodHdf5MotionLoader(
            motion_file=self.cfg.motion_file, device=self.device, fps=self.cfg.motion_fps
        )

        ref_name = self.cfg.reference_body or self.robot.data.body_names[0]
        self.ref_body_index = self.robot.data.body_names.index(ref_name)
        self.key_body_indexes = [self.robot.data.body_names.index(n) for n in self.cfg.key_body_names]

        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([ref_name])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(list(self.cfg.key_body_names))

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        self._amp_prog_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._demo_cmd_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

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
        default_pos = self.robot.data.default_joint_pos
        target = default_pos + self.cfg.action_position_scale * self.actions
        target = torch.clamp(target, self._dof_lower, self._dof_upper)
        self.robot.set_joint_position_target(target)

    def _key_body_positions_world(self) -> torch.Tensor:
        if not self.key_body_indexes:
            return torch.zeros((self.num_envs, 0, 3), device=self.device)
        return self.robot.data.body_pos_w[:, self.key_body_indexes]

    def _get_observations(self) -> dict:
        if self.cfg.playback_mode:
            cx, cy = self.cfg.playback_fixed_command_xy
            cmd = torch.tensor(
                [cx, cy], device=self.device, dtype=torch.float32
            ).unsqueeze(0).expand(self.num_envs, -1)
            prog_u = _synthetic_progress_u(self._amp_prog_step)
        else:
            prog_u = _synthetic_progress_u(self._amp_prog_step)
            if self.cfg.amp_command_mode.lower() == "random":
                lo = self.cfg.amp_random_command_xy_low
                hi = self.cfg.amp_random_command_xy_high
                cmd = torch.rand(self.num_envs, 2, device=self.device, dtype=torch.float32) * (hi - lo) + lo
            else:
                t_np = self._demo_cmd_time.detach().cpu().numpy()
                *_, cmd, _ = self._motion_loader.sample(num_samples=self.num_envs, times=t_np)

        obs = compute_hexapod_amp_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self._key_body_positions_world(),
            cmd,
            prog_u,
        )
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        self._amp_prog_step += 1
        if not self.cfg.playback_mode:
            dur = float(self._motion_loader.duration)
            if dur > 0.0:
                self._demo_cmd_time = torch.remainder(self._demo_cmd_time + self.step_dt, dur)
            else:
                self._demo_cmd_time.zero_()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Learning signal is discriminator-only (task_reward_weight 0); env return unused for policy gradient.
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            z = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
            died_fall = z < self.cfg.termination_height
            quat = self.robot.data.body_quat_w[:, self.ref_body_index]
            roll, pitch = _roll_pitch_rad_from_quat_wxyz(quat)
            died_tilt = (torch.abs(roll) > self.cfg.max_abs_base_roll) | (
                torch.abs(pitch) > self.cfg.max_abs_base_pitch
            )
            died = died_fall | died_tilt
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._amp_prog_step[env_ids] = 0

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # Root at env origin (0, 0, 0 in env frame → world = env_origins).
        root_state[:, 0:3] = self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self._demo_cmd_time[env_ids] = 0.0
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(self, env_ids: torch.Tensor, start: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
            _cmd_reset,
            _fi0,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = self.scene.env_origins[env_ids]
        root_state[:, 2] += self.cfg.reset_z_offset
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]

        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        self._demo_cmd_time[env_ids] = torch.tensor(times, device=self.device, dtype=torch.float32)

        return root_state, dof_pos, dof_vel

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
            cmd_rows,
            frame_index_0,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        if self.motion_key_body_indexes:
            key_pos = body_positions[:, self.motion_key_body_indexes]
        else:
            key_pos = body_positions.new_zeros((body_positions.shape[0], 0, 3))

        prog_ref = _synthetic_progress_u(frame_index_0)
        amp_observation = compute_hexapod_amp_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            key_pos,
            cmd_rows,
            prog_ref,
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_hexapod_amp_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    command_xy: torch.Tensor,
    progress_u: torch.Tensor,
) -> torch.Tensor:
    n_env = dof_positions.shape[0]
    if key_body_positions.shape[1] == 0:
        key_flat = dof_positions.new_zeros((n_env, 0))
    else:
        key_flat = (key_body_positions - root_positions.unsqueeze(-2)).reshape(n_env, -1)
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            key_flat,
            command_xy,
            progress_u,
        ),
        dim=-1,
    )
    return obs
