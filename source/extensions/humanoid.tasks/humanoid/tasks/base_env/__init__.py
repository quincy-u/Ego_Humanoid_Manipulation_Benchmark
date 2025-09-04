"""
Humanoid base env.
"""

import gymnasium as gym

from .base_env import BaseEnv, BaseEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Base-v0",
    entry_point="humanoid.tasks.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaseEnvCfg
    },
)
