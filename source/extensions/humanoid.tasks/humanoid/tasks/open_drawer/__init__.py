"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .open_drawer import OpenDrawerEnv, OpenDrawerEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Open-Drawer-v0",
    entry_point="humanoid.tasks.open_drawer:OpenDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenDrawerEnvCfg
    },
)
