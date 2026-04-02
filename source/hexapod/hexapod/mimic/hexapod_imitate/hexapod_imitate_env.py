# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod imitation env: policy actions in [-1,1] → joint targets via demo ``q_min``/``q_max``; PD tracking."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .hexapod_imitate_env_cfg import HexapodImitateEnvCfg
from .imitation_target_table import (
    REQUIRED_IMITATION_TARGET_NPY,
    ImitationTargetTable,
    discover_imitation_target_dirs,
)

_LOGGER = logging.getLogger(__name__)


def _euler_xyz_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Roll-pitch-yaw (XYZ intrinsic / Tait–Bryan) in radians, shape (N, 3); quat is (w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


class HexapodImitateEnv(DirectRLEnv):
    cfg: HexapodImitateEnvCfg

    def __init__(self, cfg: HexapodImitateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.ref_body_index = 0
        scan = cfg.imitate_target_scan_parent
        if scan is not None and str(scan).strip() != "":
            discovered = discover_imitation_target_dirs(Path(scan).expanduser())
            if not discovered:
                req = ", ".join(REQUIRED_IMITATION_TARGET_NPY)
                raise ValueError(
                    f"imitate_target_scan_parent={scan!r}: no trajectory folder (or this directory itself) contains "
                    f"all of: {req}. "
                    "Run target_generator on each id subfolder. "
                    "Or set imitate_target_scan_parent to '' and imitate_target_dirs to one complete folder."
                )
            target_dirs = tuple(str(p) for p in discovered)
        else:
            target_dirs = tuple(cfg.imitate_target_dirs)
            if not target_dirs:
                raise ValueError("imitate_target_dirs is empty and imitate_target_scan_parent is not set")

        self._target_table = ImitationTargetTable(
            target_dirs,
            self.device,
            weight_progress=cfg.target_match_w_progress,
            weight_pos_xy=cfg.target_match_w_pos_xy,
            weight_direction=cfg.target_match_w_direction,
        )

        na = int(cfg.action_space)
        self._cmd_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._act_prev = torch.zeros(self.num_envs, na, device=self.device, dtype=torch.float32)
        self._joint_vel_prev = torch.zeros(self.num_envs, na, device=self.device, dtype=torch.float32)

        self._contact_feet_sensor_ids: list[int] | None = None
        idx = tuple(cfg.imitation_contact_body_indices)
        if len(idx) == 6:
            self._contact_feet_sensor_ids = list(idx)
            _LOGGER.info("Imitation contact feet: ContactSensor body indices %s", self._contact_feet_sensor_ids)
        elif len(idx) not in (0, 6):
            raise ValueError("imitation_contact_body_indices must be empty (disabled) or length 6")
        else:
            names_cfg = tuple(cfg.imitation_contact_body_names)
            if len(names_cfg) == 6:
                sids, snames = self._contact_sensor.find_bodies(list(names_cfg), preserve_order=True)
                if len(sids) == 6:
                    self._contact_feet_sensor_ids = list(sids)
                    _LOGGER.info("Imitation contact feet: %s", list(zip(names_cfg, snames, sids)))
                else:
                    warnings.warn(
                        f"imitation_contact_body_names={names_cfg!r} matched only {sids!r} on contact sensor; "
                        "contact reward term disabled.",
                        stacklevel=1,
                    )
            elif len(names_cfg) != 0:
                raise ValueError("imitation_contact_body_names must be empty (disabled) or length 6")

        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self._dof_lower = dof_lower.unsqueeze(0)
        self._dof_upper = dof_upper.unsqueeze(0)

        lim_path = Path(self.cfg.imitation_joint_limits_npz).expanduser()
        if lim_path.is_file():
            lim = np.load(lim_path)
            qn = np.asarray(lim["q_min"], dtype=np.float32).reshape(-1)
            qx = np.asarray(lim["q_max"], dtype=np.float32).reshape(-1)
            if qn.shape != (na,) or qx.shape != (na,):
                raise ValueError(f"q_min/q_max must be shape ({na},), got {qn.shape}, {qx.shape}")
            self._q_demo_min = torch.tensor(qn, device=self.device, dtype=torch.float32).unsqueeze(0)
            self._q_demo_max = torch.tensor(qx, device=self.device, dtype=torch.float32).unsqueeze(0)
        else:
            msg = (
                f"imitation_joint_limits_npz not found ({lim_path}); "
                "using URDF soft joint limits for [-1,1] → q mapping. "
                "For demo-tight ranges run: "
                "python compute_imitation_joint_limits.py (mimic/dataset/hexapod/)"
            )
            warnings.warn(msg, stacklevel=1)
            _LOGGER.warning(msg)
            self._q_demo_min = self._dof_lower.clone()
            self._q_demo_max = self._dof_upper.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        a = self.actions.clamp(-1.0, 1.0)
        span = self._q_demo_max - self._q_demo_min
        q_des = self._q_demo_min + 0.5 * (a + 1.0) * span
        q_des = torch.clamp(q_des, self._dof_lower, self._dof_upper)
        self.robot.set_joint_position_target(q_des)

    def _progress_from_episode_buf(self) -> torch.Tensor:
        c = max(int(self.cfg.progress_cycle_steps), 1)
        idx = (self.episode_length_buf % c) + 1
        return idx.to(dtype=torch.float32) * 0.01

    def _get_observations(self) -> dict:
        rb = self.ref_body_index
        root_pos = self.robot.data.body_pos_w[:, rb]
        root_quat = self.robot.data.body_quat_w[:, rb]
        rpy = _euler_xyz_from_quat_wxyz(root_quat)
        vlin = self.robot.data.body_lin_vel_w[:, rb]
        wang = self.robot.data.body_ang_vel_w[:, rb]
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel

        prog = self._progress_from_episode_buf().unsqueeze(-1)

        # Action history: normalized commands in ~[-1,1] (same as sent to mapping).
        obs = torch.cat(
            (
                root_pos,
                rpy,
                vlin,
                wang,
                jp,
                jv,
                self.actions,
                self._act_prev,
                self._cmd_xy,
                prog,
            ),
            dim=-1,
        )

        prog_flat = prog.squeeze(-1)
        targets = self._target_table.query_nearest(root_pos, self._cmd_xy, prog_flat)
        self.extras["imitation_targets"] = {k: v for k, v in targets.items() if k != "table_index"}
        self.extras["imitation_table_index"] = targets["table_index"]

        self._act_prev.copy_(self.actions)
        return {"policy": obs}

    def _contact_flags(self) -> torch.Tensor:
        """(N, 6) binary {0,1} from :class:`ContactSensor` net normal force norm (world frame)."""
        n = self.num_envs
        if self._contact_feet_sensor_ids is None:
            return torch.zeros(n, 6, device=self.device, dtype=torch.float32)
        nf_hist = self._contact_sensor.data.net_forces_w_history
        if nf_hist is None:
            return torch.zeros(n, 6, device=self.device, dtype=torch.float32)
        recent = nf_hist[:, 0, :, :]
        idx_t = torch.tensor(self._contact_feet_sensor_ids, device=self.device, dtype=torch.long)
        f = recent[:, idx_t, :]
        mag = torch.linalg.vector_norm(f, dim=-1)
        return (mag > float(self.cfg.imitation_contact_force_threshold)).to(dtype=torch.float32)

    def _get_rewards(self) -> torch.Tensor:
        rb = self.ref_body_index
        root_pos = self.robot.data.body_pos_w[:, rb]
        root_quat = self.robot.data.body_quat_w[:, rb]
        rpy = _euler_xyz_from_quat_wxyz(root_quat)
        vlin = self.robot.data.body_lin_vel_w[:, rb]
        wang = self.robot.data.body_ang_vel_w[:, rb]
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel

        prog = self._progress_from_episode_buf()
        q = self._target_table.query_nearest(root_pos, self._cmd_xy, prog)
        tg = {k: v for k, v in q.items() if k != "table_index"}

        tp = tg["target_position"]
        to = tg["target_orientation"]
        tv = tg["target_linear_velocity"]
        tw = tg["target_angular_velocity"]
        tj = tg["target_joint_position"]
        tjv = tg["target_joint_velocity"]
        tc = tg["target_contact"]

        n_pos = torch.linalg.vector_norm(root_pos - tp, dim=-1)
        r_pose = torch.exp(-float(self.cfg.rew_exp_pose_coef) * n_pos)

        d_ori = torch.atan2(torch.sin(rpy - to), torch.cos(rpy - to))
        n_ori = torch.linalg.vector_norm(d_ori, dim=-1)
        r_ori = torch.exp(-float(self.cfg.rew_exp_ori_coef) * n_ori)

        n_vlin = torch.linalg.vector_norm(vlin - tv, dim=-1)
        r_vlin = torch.exp(-float(self.cfg.rew_exp_vlin_coef) * n_vlin)

        n_vang = torch.linalg.vector_norm(wang - tw, dim=-1)
        r_vang = torch.exp(-float(self.cfg.rew_exp_vang_coef) * n_vang)

        r_jp = -torch.linalg.vector_norm(jp - tj, dim=-1) * 15.0
        r_jv = -torch.linalg.vector_norm(jv - tjv, dim=-1) * 0.001

        cur_c = self._contact_flags()
        tc_b = (tc >= 0.5).float()
        cur_b = (cur_c >= 0.5).float()
        r_contact = (cur_b == tc_b).float().sum(dim=-1)

        a_c = self.actions.clamp(-1.0, 1.0)
        r_act = -float(self.cfg.rew_action_penalty_scale) * torch.linalg.vector_norm(a_c, dim=-1)

        dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        qdd = (jv - self._joint_vel_prev) / max(dt, 1e-8)
        r_qdd = -float(self.cfg.rew_joint_acc_penalty_scale) * torch.linalg.vector_norm(qdd, dim=-1)
        self._joint_vel_prev.copy_(jv)

        r_surv = torch.full((self.num_envs,), float(self.cfg.rew_survival), device=self.device, dtype=torch.float32)

        return (
            r_pose
            + r_ori
            + r_vlin
            + r_vang
            + r_jp
            + r_jv
            + r_contact
            + r_act
            + r_qdd
            + r_surv
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        rb = self.ref_body_index
        z = self.robot.data.body_pos_w[:, rb, 2]
        died_fall = z < float(self.cfg.termination_height)
        quat = self.robot.data.body_quat_w[:, rb]
        rpy = _euler_xyz_from_quat_wxyz(quat)
        roll, pitch = rpy[:, 0], rpy[:, 1]
        died_tilt = (torch.abs(roll) > float(self.cfg.max_abs_base_roll)) | (
            torch.abs(pitch) > float(self.cfg.max_abs_base_pitch)
        )
        died = died_fall | died_tilt
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = env_ids.shape[0]
        self._act_prev[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        ang = torch.rand(n, device=self.device, dtype=torch.float32) * (2.0 * torch.pi)
        self._cmd_xy[env_ids, 0] = torch.cos(ang)
        self._cmd_xy[env_ids, 1] = torch.sin(ang)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._joint_vel_prev[env_ids] = joint_vel
