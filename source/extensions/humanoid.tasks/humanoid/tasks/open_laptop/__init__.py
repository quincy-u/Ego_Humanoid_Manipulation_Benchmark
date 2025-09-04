"""
Humanoid heat food environment.
"""

import gymnasium as gym

from .open_laptop import OpenLaptopEnv, OpenLaptopEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Open-Laptop-v0",
    entry_point="humanoid.tasks.open_laptop:OpenLaptopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenLaptopEnvCfg
    },
)
