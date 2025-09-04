"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .stack_can_into_drawer import StackCanIntoDrawerEnv, StackCanIntoDrawerEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Stack-Can-Into-Drawer-v0",
    entry_point="humanoid.tasks.stack_can_into_drawer:StackCanIntoDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StackCanIntoDrawerEnvCfg
    },
)
