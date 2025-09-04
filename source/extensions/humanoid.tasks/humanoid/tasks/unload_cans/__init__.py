"""
Humanoid push box environment.
"""

import gymnasium as gym

from .unload_cans_env import UnloadCansEnv, UnloadCansEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Unload-Cans-v0",
    entry_point="humanoid.tasks.unload_cans:UnloadCansEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnloadCansEnvCfg
    },
)
