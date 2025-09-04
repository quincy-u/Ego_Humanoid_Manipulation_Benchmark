"""
Humanoid push box environment.
"""

import gymnasium as gym

from .push_box_env import PushBoxEnv, PushBoxEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Push-Box-v0",
    entry_point="humanoid.tasks.push_box:PushBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PushBoxEnvCfg
    },
)
