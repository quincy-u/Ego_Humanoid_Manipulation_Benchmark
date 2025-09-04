"""
Humanoid heat food environment.
"""

import gymnasium as gym

from .flip_mug import FlipMugEnv, FlipMugEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Flip-Mug-v0",
    entry_point="humanoid.tasks.flip_mug:FlipMugEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlipMugEnvCfg
    },
)
