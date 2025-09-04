"""
Humanoid push box environment.
"""

import gymnasium as gym

from .sort_cans_env import SortCansEnv, SortCansEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Sort-Cans-v0",
    entry_point="humanoid.tasks.sort_cans:SortCansEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SortCansEnvCfg
    },
)
