"""
Humanoid stack items environment.
"""

import gymnasium as gym

from .stack_can import StackCanEnv, StackCanEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Stack-Can-v0",
    entry_point="humanoid.tasks.stack_can:StackCanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StackCanEnvCfg
    },
)
