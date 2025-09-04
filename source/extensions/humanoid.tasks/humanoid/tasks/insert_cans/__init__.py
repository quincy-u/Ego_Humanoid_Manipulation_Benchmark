"""
Humanoid push box environment.
"""

import gymnasium as gym

from .insert_cans_env import InsertCansEnv, InsertCansEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Insert-Cans-v0",
    entry_point="humanoid.tasks.insert_cans:InsertCansEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": InsertCansEnvCfg
    },
)
