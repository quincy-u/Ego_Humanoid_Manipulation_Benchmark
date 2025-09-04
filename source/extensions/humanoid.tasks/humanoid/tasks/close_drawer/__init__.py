"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .close_drawer import CloseDrawerEnv, CloseDrawerEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Close-Drawer-v0",
    entry_point="humanoid.tasks.close_drawer:CloseDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CloseDrawerEnvCfg
    },
)
