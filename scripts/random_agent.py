"""Script to run a humanoid environment with random action agent."""
"""Script is derived from IsaacLab random agent example."""

import importlib
import sys
if "omni.isaac.lab" not in sys.modules:
    sys.modules["omni.isaac.lab"] = importlib.import_module("isaaclab")
    
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent for Ego_Humanoid_Manipulation_Benchmark environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

if "omni.isaac.lab_tasks" not in sys.modules:
    sys.modules["omni.isaac.lab"] = importlib.import_module("isaaclab")
    sys.modules["omni.isaac.lab_tasks"] = importlib.import_module("isaaclab_tasks")
import omni.isaac.lab_tasks  # noqa: F401
import humanoid.tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
