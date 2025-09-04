"""
Humanoid heat food environment.
"""

import gymnasium as gym

from .pour_balls import PourBallsEnv, PourBallsEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-Pour-Balls-v0",
    entry_point="humanoid.tasks.pour_balls:PourBallsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PourBallsEnvCfg
    },
)
