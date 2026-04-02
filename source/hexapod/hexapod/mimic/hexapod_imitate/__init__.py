# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod imitation (BC-style targets + PPO)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Template-Hexapod-Imitate-Direct-v0",
    entry_point=f"{__name__}.hexapod_imitate_env:HexapodImitateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_imitate_env_cfg:HexapodImitateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodImitatePPORunnerCfg",
    },
)
