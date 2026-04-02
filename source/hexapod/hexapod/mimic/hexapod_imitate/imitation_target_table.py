# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load ``target_generator`` outputs and pick rows from (world position, planar direction, progress)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

_TARGET_NPY = (
    "position.npy",
    "direction.npy",
    "progress.npy",
    "target_position.npy",
    "target_orientation.npy",
    "target_linear_velocity.npy",
    "target_angular_velocity.npy",
    "target_joint_position.npy",
    "target_joint_velocity.npy",
    "target_contact.npy",
)

# Public alias (error messages / docs): every file must exist per trajectory folder.
REQUIRED_IMITATION_TARGET_NPY: tuple[str, ...] = _TARGET_NPY

def _folder_has_all_npy(folder: Path) -> bool:
    return all((folder / name).is_file() for name in _TARGET_NPY)


def discover_imitation_target_dirs(raw_root: Path) -> list[Path]:
    """Trajectory dirs under ``raw_root``: either ``raw_root`` itself if it holds all npy, else each direct child dir that does (sorted by name)."""
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        return []
    if _folder_has_all_npy(raw_root):
        return [raw_root]
    out: list[Path] = []
    for p in sorted(raw_root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and _folder_has_all_npy(p):
            out.append(p)
    return out


def _load_one_folder(folder: Path) -> dict[str, np.ndarray]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    out: dict[str, np.ndarray] = {}
    for name in _TARGET_NPY:
        fp = folder / name
        if not fp.is_file():
            need = ", ".join(_TARGET_NPY)
            raise FileNotFoundError(
                f"Missing {fp}. Each trajectory folder needs: {need}. "
                f"Run ``target_generator`` on this folder's parent with ``--root``."
            )
        out[name] = np.load(fp).astype(np.float64)
    t = int(out["position.npy"].shape[0])
    for name, arr in out.items():
        if arr.shape[0] != t:
            raise ValueError(f"{folder}: {name} has length {arr.shape[0]}, expected {t} like position.npy")
    return out


def _merge_folders(folders: Sequence[Path]) -> dict[str, np.ndarray]:
    if not folders:
        raise ValueError("ImitationTargetTable: empty folder list")
    blocks = [_load_one_folder(p) for p in folders]
    return {name: np.concatenate([b[name] for b in blocks], axis=0) for name in _TARGET_NPY}


class ImitationTargetTable:
    """Reference table from one or more trajectory folders (rows concatenated along time).

    ``target_*`` arrays are expected already in the same frame as the sim (fix offsets at export).
    """

    def __init__(
        self,
        target_dirs: Sequence[str | Path] | str | Path,
        device: torch.device,
        *,
        weight_progress: float = 1.0,
        weight_pos_xy: float = 0.0,
        weight_direction: float = 0.0,
    ) -> None:
        self.device = device
        self.weight_progress = float(weight_progress)
        self.weight_pos_xy = float(weight_pos_xy)
        self.weight_direction = float(weight_direction)

        if isinstance(target_dirs, (str, Path)):
            seq: tuple[str | Path, ...] = (target_dirs,)
        else:
            seq = tuple(target_dirs)
        if not seq:
            raise ValueError("ImitationTargetTable: target_dirs is empty")

        seen: set[Path] = set()
        paths: list[Path] = []
        for raw in seq:
            p = Path(raw).expanduser().resolve()
            if p in seen:
                continue
            seen.add(p)
            paths.append(p)

        if not paths:
            raise ValueError("ImitationTargetTable: no directories after deduplication")

        merged = _merge_folders(paths)

        self.pos_ref = torch.tensor(merged["position.npy"], device=device, dtype=torch.float32)
        self.dir_ref = torch.tensor(merged["direction.npy"], device=device, dtype=torch.float32)
        self.prog_ref = torch.tensor(merged["progress.npy"], device=device, dtype=torch.float32).reshape(-1)

        self.tgt_pos = torch.tensor(merged["target_position.npy"], device=device, dtype=torch.float32)
        self.tgt_ori = torch.tensor(merged["target_orientation.npy"], device=device, dtype=torch.float32)
        self.tgt_vlin = torch.tensor(merged["target_linear_velocity.npy"], device=device, dtype=torch.float32)
        self.tgt_vang = torch.tensor(merged["target_angular_velocity.npy"], device=device, dtype=torch.float32)
        self.tgt_jp = torch.tensor(merged["target_joint_position.npy"], device=device, dtype=torch.float32)
        self.tgt_jv = torch.tensor(merged["target_joint_velocity.npy"], device=device, dtype=torch.float32)
        self.tgt_contact = torch.tensor(merged["target_contact.npy"], device=device, dtype=torch.float32)

        t = self.pos_ref.shape[0]
        for name, x in [
            ("dir", self.dir_ref),
            ("tgt_pos", self.tgt_pos),
            ("tgt_ori", self.tgt_ori),
            ("tgt_vlin", self.tgt_vlin),
            ("tgt_vang", self.tgt_vang),
            ("tgt_jp", self.tgt_jp),
            ("tgt_jv", self.tgt_jv),
            ("tgt_contact", self.tgt_contact),
        ]:
            if x.shape[0] != t:
                raise ValueError(f"{name} length {x.shape[0]} != merged position length {t}")

        self._dir_ref_u = self.dir_ref / (torch.linalg.vector_norm(self.dir_ref, dim=-1, keepdim=True).clamp(min=1e-8))

    @property
    def num_rows(self) -> int:
        return int(self.pos_ref.shape[0])

    def query_nearest(
        self,
        root_pos_w: torch.Tensor,
        direction_xy: torch.Tensor,
        progress: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Nearest row by weighted (circular progress, planar XY, direction cosine).

        Args:
            root_pos_w: (N, 3) world root position.
            direction_xy: (N, 2) command direction (not necessarily unit).
            progress: (N,) or (N, 1) scalar in ~[0.01, 1.0].

        Returns:
            Dict of (N, D) tensors aligned with ``target_*`` files.
        """
        n = root_pos_w.shape[0]
        prog = progress.reshape(-1).clamp(0.0, 1.0)
        t = self.num_rows

        pr = self.prog_ref.unsqueeze(0).expand(n, -1)
        pe = prog.unsqueeze(1).expand(-1, t)
        dp = (pr - pe).abs()
        dp = torch.minimum(dp, 1.0 - dp)

        pos_xy = root_pos_w[:, :2].unsqueeze(1)
        ref_xy = self.pos_ref[:, :2].unsqueeze(0)
        dxy = torch.sum((ref_xy - pos_xy) ** 2, dim=-1)

        u = direction_xy / (torch.linalg.vector_norm(direction_xy, dim=-1, keepdim=True).clamp(min=1e-8))
        cos = (self._dir_ref_u.unsqueeze(0) * u.unsqueeze(1)).sum(dim=-1)
        cos = cos.clamp(-1.0, 1.0)

        cost = self.weight_progress * dp + self.weight_pos_xy * dxy + self.weight_direction * (1.0 - cos)
        idx = cost.argmin(dim=1)

        return {
            "target_position": self.tgt_pos[idx],
            "target_orientation": self.tgt_ori[idx],
            "target_linear_velocity": self.tgt_vlin[idx],
            "target_angular_velocity": self.tgt_vang[idx],
            "target_joint_position": self.tgt_jp[idx],
            "target_joint_velocity": self.tgt_jv[idx],
            "target_contact": self.tgt_contact[idx],
            "table_index": idx,
        }
