import gymnasium as gym

from . import agents

gym.register(
    id="Template-HexapodWalk-Direct-v0",
    entry_point=f"{__name__}.hexapod_walk_env:HexapodWalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_walk_env:HexapodWalkEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodWalkPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)