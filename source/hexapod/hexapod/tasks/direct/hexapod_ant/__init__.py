import gymnasium as gym

from . import agents

gym.register(
    id="Template-HexapodAnt-Direct-v0",
    entry_point=f"{__name__}.hexapod_ant_env:HexapodAntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_ant_env:HexapodAntEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)