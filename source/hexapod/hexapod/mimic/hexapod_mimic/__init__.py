"""Hexapod mimic / AMP — Gym registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Template-Hexapod-AMP-Mimic-Direct-v0",
    entry_point=f"{__name__}.hexapod_mimic_env:HexapodAmpMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_mimic_env_cfg:HexapodAmpMimicEnvCfg",
        # Isaac Lab train.py uses skrl_cfg_entry_point when --algorithm PPO (default); AMP YAML is the only agent for this env.
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_hexapod_amp_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_hexapod_amp_cfg.yaml",
    },
)
