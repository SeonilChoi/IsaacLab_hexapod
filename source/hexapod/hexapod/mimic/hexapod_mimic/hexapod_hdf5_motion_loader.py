# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

"""Load concatenated AMP reference motion from Isaac Lab-style HDF5 (hexapod mimic exports)."""

from __future__ import annotations

import os
import re

import h5py
import numpy as np
import torch

from isaaclab.utils.math import quat_from_euler_xyz


def _sorted_demo_keys(h5: h5py.File) -> list[str]:
    data = h5["data"]
    keys = [k for k in data.keys() if k.startswith("demo_")]
    return sorted(keys, key=lambda s: int(re.search(r"demo_(\d+)$", s).group(1)) if re.search(r"demo_(\d+)$", s) else s)


class HexapodHdf5MotionLoader:
    """Motion clip built by concatenating all ``data/demo_*`` trajectories in one HDF5 file.

    Expects per-demo groups (e.g. from ``build_dataset_from_npy.py``)::

        data/demo_*/obs/joint_pos   (T, D)
        data/demo_*/obs/joint_vel   (T, D)
        data/demo_*/obs/pose        (T, 6)   # position xyz + roll-pitch-yaw
        data/demo_*/obs/twist       (T, 6)   # linear vel xyz + angular vel xyz
        data/demo_*/obs/command     (T, 2)   # optional; zeros if missing
    """

    def __init__(self, motion_file: str, device: torch.device, *, fps: float = 120.0) -> None:
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        self.device = device
        self.dt = 1.0 / fps
        self._file_num_demos_attr: int | None = None

        joint_pos_chunks: list[np.ndarray] = []
        joint_vel_chunks: list[np.ndarray] = []
        pose_chunks: list[np.ndarray] = []
        twist_chunks: list[np.ndarray] = []
        command_chunks: list[np.ndarray] = []

        with h5py.File(motion_file, "r") as f:
            raw_attr = f.attrs.get("num_demos")
            if raw_attr is not None:
                self._file_num_demos_attr = int(raw_attr)
            demo_names = _sorted_demo_keys(f)
            for name in demo_names:
                g = f["data"][name]
                if "obs" not in g:
                    raise KeyError(f"{name}: missing 'obs' group")
                obs = g["obs"]
                if "joint_pos" not in obs or "joint_vel" not in obs:
                    raise KeyError(f"{name}/obs: need joint_pos and joint_vel datasets")
                if "pose" not in obs or "twist" not in obs:
                    raise KeyError(f"{name}/obs: need pose (T,6) and twist (T,6) for hexapod AMP")

                jp = np.asarray(obs["joint_pos"][...], dtype=np.float32)
                jv = np.asarray(obs["joint_vel"][...], dtype=np.float32)
                pose = np.asarray(obs["pose"][...], dtype=np.float32)
                twist = np.asarray(obs["twist"][...], dtype=np.float32)
                if jp.shape != jv.shape or jp.shape[0] != pose.shape[0] or jp.shape[0] != twist.shape[0]:
                    raise ValueError(f"{name}: length or dof mismatch among obs arrays")
                if pose.shape[-1] != 6 or twist.shape[-1] != 6:
                    raise ValueError(f"{name}: pose and twist must have last dim 6")

                joint_pos_chunks.append(jp)
                joint_vel_chunks.append(jv)
                pose_chunks.append(pose)
                twist_chunks.append(twist)
                if "command" in obs:
                    cmd = np.asarray(obs["command"][...], dtype=np.float32)
                    if cmd.shape != (jp.shape[0], 2):
                        raise ValueError(f"{name}/obs/command: expected shape (T, 2), got {cmd.shape}")
                    command_chunks.append(cmd)
                else:
                    command_chunks.append(np.zeros((jp.shape[0], 2), dtype=np.float32))

        self.dof_positions = torch.tensor(np.concatenate(joint_pos_chunks, axis=0), device=device)
        self.dof_velocities = torch.tensor(np.concatenate(joint_vel_chunks, axis=0), device=device)
        poses = np.concatenate(pose_chunks, axis=0)
        twists = np.concatenate(twist_chunks, axis=0)

        self.body_positions = torch.tensor(poses[:, :3], device=device).unsqueeze(1)
        euler = torch.tensor(poses[:, 3:6], device=device)
        q = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.body_rotations = q.unsqueeze(1)
        self.body_linear_velocities = torch.tensor(twists[:, :3], device=device).unsqueeze(1)
        self.body_angular_velocities = torch.tensor(twists[:, 3:6], device=device).unsqueeze(1)
        self.commands = torch.tensor(np.concatenate(command_chunks, axis=0), device=device)

        self.num_frames = self.dof_positions.shape[0]
        self.num_dofs = self.dof_positions.shape[1]
        self.num_bodies = 1
        self._body_names = ["root"]
        self._dof_names: list[str] | None = None
        self.duration = self.dt * (self.num_frames - 1)
        n_demos = len(demo_names)
        demo_preview = ", ".join(demo_names[:12]) + (" ..." if n_demos > 12 else "")
        attr_note = ""
        if self._file_num_demos_attr is not None and self._file_num_demos_attr != n_demos:
            attr_note = f" (HDF5 attr num_demos={self._file_num_demos_attr} != group count)"
        print(
            f"Motion loaded ({motion_file}): {n_demos} demo(s) [{demo_preview}], "
            f"duration: {self.duration:.4f} s, frames: {self.num_frames}, dofs: {self.num_dofs}{attr_note}"
        )

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        duration = self.duration if duration is None else duration
        if duration > self.duration:
            raise AssertionError(f"duration ({duration}) > motion duration ({self.duration})")
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None, duration: float | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend_t = torch.tensor(blend, dtype=torch.float32, device=self.device)
        frame_index_0 = torch.tensor(index_0, dtype=torch.long, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend_t, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend_t, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend_t, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend_t, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend_t, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend_t, start=index_0, end=index_1),
            self._interpolate(self.commands, blend=blend_t, start=index_0, end=index_1),
            frame_index_0,
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        if self._dof_names is not None:
            return [self._dof_names.index(n) for n in dof_names]
        if len(dof_names) != self.num_dofs:
            raise ValueError(
                f"Motion has {self.num_dofs} dofs but robot reports {len(dof_names)} names. "
                "Set joint name metadata in HDF5 or align export order with the robot."
            )
        return list(range(self.num_dofs))

    def get_body_index(self, body_names: list[str]) -> list[int]:
        if not body_names:
            return []
        return [0] * len(body_names)
