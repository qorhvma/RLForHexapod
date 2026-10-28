import gymnasium as gym

_PACKAGE = __name__

gym.register(
    id="Hexapod-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{_PACKAGE}.env_cfg:HexapodEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:HexapodStatePPORunnerCfg"
    },
)
