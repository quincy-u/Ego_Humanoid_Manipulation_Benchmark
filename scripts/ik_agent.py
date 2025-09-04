"""Script to run a humanoid environment with pose control IK agent."""

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
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.math import subtract_frame_transforms


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 5
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    env = env.unwrapped  # type: ignore
    # IK controllers configs
    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="svd")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.num_envs, device=env.sim.device)
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="svd")
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.num_envs, device=env.sim.device)
    left_jacobin_idx = env.left_ee_idx-1
    right_jacobin_idx = env.right_ee_idx-1

    # Create buffers to store actions
    left_ik_commands_world = torch.zeros(env.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    action = torch.zeros((env.num_envs, 50), device=env.robot.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy() # type: ignore
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # we only visualize the right end-effector for simplicity, can visualize both arms if needed
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    left_ee_goal = [0.2521, 0.2212,  1.2,  0.0586,  0.6937, -0.7154, -0.0589]
    left_ee_goal = torch.tensor(left_ee_goal, device=env.device)
    right_ee_goal = [0.2521, -0.2212,  1.2,  0.0586,  0.6937, -0.7154, -0.0589]
    right_ee_goal = torch.tensor(right_ee_goal, device=env.device)

    left_ik_controller.reset()
    right_ik_controller.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # obtain quantities from simulation
            robot_pose_w = env.robot.data.root_state_w[:, 0:7]
            left_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, left_jacobin_idx, :, env.cfg.left_arm_cfg.joint_ids]
            left_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.left_arm_cfg.body_ids[0], 0:7]
            left_joint_pos = env.robot.data.joint_pos[:, env.cfg.left_arm_cfg.joint_ids]
            right_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, right_jacobin_idx, :, env.cfg.right_arm_cfg.joint_ids]
            right_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.right_arm_cfg.body_ids[0], 0:7]
            right_joint_pos = env.robot.data.joint_pos[:, env.cfg.right_arm_cfg.joint_ids]
            # prepare IK 
            left_ee_curr_pose_robot, left_ee_curr_quat_robot = subtract_frame_transforms(
                robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ee_curr_pose_world[:, 0:3], left_ee_curr_pose_world[:, 3:7]
            )
            right_ee_curr_pos_robot, right_ee_curr_quat_robot = subtract_frame_transforms(
                robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ee_curr_pose_world[:, 0:3], right_ee_curr_pose_world[:, 3:7]
            )
            left_ik_commands_world[:, 0:7] = left_ee_goal[0:7]
            left_ik_commands_robot[:, 0:3], left_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
                robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ik_commands_world[:, 0:3], left_ik_commands_world[:, 3:7]
            )
            right_ik_commands_world[:, 0:7] = right_ee_goal[0:7]
            right_ik_commands_robot[:, 0:3], right_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
                robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ik_commands_world[:, 0:3], right_ik_commands_world[:, 3:7]
            )
            left_ik_controller.set_command(left_ik_commands_robot, left_ee_curr_pose_robot, left_ee_curr_quat_robot)
            right_ik_controller.set_command(right_ik_commands_robot, right_ee_curr_pos_robot, right_ee_curr_quat_robot)
            # compute the joint commands
            left_joint_pos_des = left_ik_controller.compute(left_ee_curr_pose_robot, left_ee_curr_quat_robot, left_arm_jacobian, left_joint_pos)
            right_joint_pos_des = right_ik_controller.compute(right_ee_curr_pos_robot, right_ee_curr_quat_robot, right_arm_jacobian, right_joint_pos)
            action[:, :] = 0
            action[:, env.cfg.left_arm_cfg.joint_ids] = left_joint_pos_des
            action[:, env.cfg.right_arm_cfg.joint_ids] = right_joint_pos_des
            env.step(action)

            # update markers
            right_ee_curr_pose_world = env.robot.data.body_state_w[:,  env.cfg.right_arm_cfg.body_ids[0], 0:7]
            ee_marker.visualize(right_ee_curr_pose_world[:, 0:3]+ env.scene.env_origins, right_ee_curr_pose_world[:, 3:7])
            goal_marker.visualize(right_ik_commands_world[:, 0:3] + env.scene.env_origins, right_ik_commands_world[:, 3:7])

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
