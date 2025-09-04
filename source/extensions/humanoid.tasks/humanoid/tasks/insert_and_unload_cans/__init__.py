"""
Humanoid push box environment.
"""

import gymnasium as gym

from .insert_and_unload_cans import InsertAndUnloadCansEnv, InsertAndUnloadCansEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Insert-And-Unload-Cans-v0",
    entry_point="humanoid.tasks.insert_and_unload_cans:InsertAndUnloadCansEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": InsertAndUnloadCansEnvCfg
    },
)
