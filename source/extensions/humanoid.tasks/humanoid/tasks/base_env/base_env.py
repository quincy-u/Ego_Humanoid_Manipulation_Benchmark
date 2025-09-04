from __future__ import annotations
import time

import torch
from collections.abc import Sequence
from humanoid.tasks.data.h1 import (H1_INSPIRE_CFG, H1_INSPIRE_LEFT_ARM_CFG, H1_INSPIRE_LEFT_HAND_CFG,
                                    H1_INSPIRE_RIGHT_ARM_CFG, H1_INSPIRE_RIGHT_HAND_CFG)
from humanoid.tasks.data.scene import ROOM_CFG
from humanoid.tasks.data.table import TABLE_CFG
from omni.isaac.lab.sensors.contact_sensor.contact_sensor import ContactSensor
from omni.isaac.lab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.camera.camera import Camera
from omni.isaac.lab.sensors.camera.camera_cfg import CameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files import spawn_ground_plane
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_util

@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=10.0, replicate_physics=False)
    # robot
    robot: ArticulationCfg = H1_INSPIRE_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore
    left_arm_cfg = H1_INSPIRE_LEFT_ARM_CFG
    right_arm_cfg = H1_INSPIRE_RIGHT_ARM_CFG
    left_hand_cfg = H1_INSPIRE_LEFT_HAND_CFG
    right_hand_cfg = H1_INSPIRE_RIGHT_HAND_CFG
    # table
    table = TABLE_CFG.replace(prim_path="/World/envs/env_.*/Table")  # type: ignore
    table.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.7, 0, 0), rot=(0.70711, 0, 0, 0.70711))
    # background
    room = ROOM_CFG.replace(prim_path="/World/envs/env_.*/Room")  # type: ignore
    # camera
    left_eye_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/left_eye_camera",
        height=360,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=8.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.09, 0.033, 0.65), rot=(0.66446, 0.24184, -0.24184, -0.664464), convention="opengl"),
    )
    right_eye_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/right_eye_camera",
        height=360,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=8.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.09, -0.033, 0.65), rot=(0.66446, 0.24184, -0.24184, -0.664464), convention="opengl"),
    )
    main_camera = CameraCfg(
        prim_path="/World/envs/env_.*/main_camera",
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=8.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.09, 0.0, 1.7), rot=(0.66446, 0.24184, -0.24184, -0.664464), convention="opengl"),
    )
    left_hand_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/L_hand_base_link/left_hand_camera",
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=8.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.1, 0.04, 0.0), rot=(-0.17705, -0.17705, 0.68458, 0.68458), convention="opengl"),
    )
    right_hand_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/R_hand_base_link/right_hand_camera",
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.FisheyeCameraCfg(
            focal_length=8.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.1, 0.04, 0.0), rot=(0.68458, -0.68458, -0.17705, 0.17705), convention="opengl"),
    )
    # contact sensors
    left_hand_contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/L_index_intermediate")
    right_hand_contact_sensor = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/R_index_intermediate")
    # env
    episode_length_s = 0.5
    decimation = 4
    action_scale = 1
    num_actions = 50
    num_observations = 99
    # IsaacLab updated variable name for some reason
    observation_space = num_observations
    action_space = num_actions
    spawn_table = True
    spawn_background = False
    room_idx = 1
    table_idx = 2
    seed = 0
    randomize = True
    randomize_range = 1.0
    randomize_idx = -1

class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # unit tensors
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # set up arm cfg for ee pos data
        self.cfg.left_arm_cfg.resolve(self.scene)
        self.cfg.right_arm_cfg.resolve(self.scene)
        self.cfg.left_hand_cfg.resolve(self.scene)
        self.cfg.right_hand_cfg.resolve(self.scene)
        self.left_ee_idx = self.cfg.left_arm_cfg.body_ids[0]  # type: ignore
        self.right_ee_idx = self.cfg.right_arm_cfg.body_ids[0]  # type: ignore
        self.left_finger_tips_idx = self.cfg.left_hand_cfg.body_ids  # type: ignore
        self.right_finger_tips_idx = self.cfg.right_hand_cfg.body_ids  # type: ignore
        # buffers
        self.left_ee_target_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.right_ee_target_pose = torch.zeros((self.num_envs, 7), device=self.device)

    def _update_cfg_mesh(self, cfg, mesh_idx):
        # replacing last 7 characters, e.g., '001.usd' -> '002.usd'
        new_mesh_suffix = (3 - len(str(mesh_idx))) * "0" + str(mesh_idx)
        cfg.spawn.usd_path = cfg.spawn.usd_path[:-7] + new_mesh_suffix + ".usd"

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add sensors
        self.left_eye_camera = Camera(self.cfg.left_eye_camera)
        self.right_eye_camera = Camera(self.cfg.right_eye_camera)
        self.main_camera = Camera(self.cfg.main_camera)
        self.left_hand_camera = Camera(self.cfg.left_hand_camera)
        self.right_hand_camera = Camera(self.cfg.right_hand_camera)
        self.left_hand_contact_sensor = ContactSensor(self.cfg.left_hand_contact_sensor)
        self.right_hand_contact_sensor = ContactSensor(self.cfg.right_hand_contact_sensor)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        if self.cfg.spawn_table:
            self._update_cfg_mesh(self.cfg.table, self.cfg.table_idx)
            self.cfg.table.spawn.func(
                self.cfg.table.prim_path,
                self.cfg.table.spawn,
                translation=self.cfg.table.init_state.pos,
                orientation=self.cfg.table.init_state.rot,
            )
        if self.cfg.spawn_background:
            self._update_cfg_mesh(self.cfg.room, self.cfg.room_idx)
            self.cfg.room.spawn.func(
                self.cfg.room.prim_path,
                self.cfg.room.spawn,
                translation=self.cfg.room.init_state.pos,
                orientation=self.cfg.room.init_state.rot,
            )
        self._register_scene()

    def _register_scene(self):
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # register 
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["left_eye_camera"] = self.left_eye_camera
        self.scene.sensors["right_eye_camera"] = self.right_eye_camera
        self.scene.sensors["fixed_camera"] = self.main_camera
        self.scene.sensors["left_hand_camera"] = self.left_hand_camera
        self.scene.sensors["right_hand_camera"] = self.right_hand_camera
        self.scene.sensors["left_hand_contact_sensor"] = self.left_hand_contact_sensor
        self.scene.sensors["right_hand_contact_sensor"] = self.right_hand_contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.actions *= self.cfg.action_scale
        self.actions[:] = torch.clamp(self.actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # self.robot.set_joint_position_target(self.actions)
        self.robot.write_joint_state_to_sim(self.actions, torch.zeros_like(self.actions))

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore
        self.robot.reset(env_ids=env_ids)
        # reset robot
        root_state = self.robot.data.default_root_state[env_ids, :7]
        root_state[:, 0:3] += self.scene.env_origins[env_ids, :]
        root_vel = torch.zeros([len(env_ids), 6]).to(self.device)  # type: ignore
        self.robot.write_root_pose_to_sim(root_pose=root_state, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_velocity=root_vel, env_ids=env_ids)
        # robot joint state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids, :] = 0
        # print("Camera intrinsic matrices:")
        # print(self.camera.data.intrinsic_matrices)
    
    def _get_target_ee_pose(self):
        left_arm_jacobian = self.robot.root_physx_view.get_jacobians()[:, self.left_ee_idx-1, :, self.cfg.left_arm_cfg.joint_ids]
        right_arm_jacobian = self.robot.root_physx_view.get_jacobians()[:, self.right_ee_idx-1, :, self.cfg.right_arm_cfg.joint_ids]
        left_delta_joint_pose = self.actions[:, self.cfg.left_arm_cfg.joint_ids] - self.robot.data.joint_pos[:, self.cfg.left_arm_cfg.joint_ids]
        right_delta_joint_pose = self.actions[:, self.cfg.right_arm_cfg.joint_ids] - self.robot.data.joint_pos[:, self.cfg.right_arm_cfg.joint_ids]
        left_delta_ee_pose = (left_arm_jacobian.double() @ left_delta_joint_pose.double().unsqueeze(-1)).squeeze(-1)
        right_delta_ee_pose = (right_arm_jacobian.double() @ right_delta_joint_pose.double().unsqueeze(-1)).squeeze(-1)
        self.left_ee_target_pose[:, 0:3] = self.robot.data.body_state_w[:, self.left_ee_idx, 0:3] + left_delta_ee_pose[:, 0:3]
        self.right_ee_target_pose[:, 0:3] = self.robot.data.body_state_w[:, self.right_ee_idx, 0:3] + right_delta_ee_pose[:, 0:3]
        left_delta_ee_quat = math_util.quat_from_euler_xyz(left_delta_ee_pose[:, 3], left_delta_ee_pose[:, 4], left_delta_ee_pose[:, 5])
        self.left_ee_target_pose[:, 3:7] = math_util.quat_mul(self.robot.data.body_state_w[:, self.left_ee_idx, 3:7], left_delta_ee_quat)
        right_delta_ee_quat = math_util.quat_from_euler_xyz(right_delta_ee_pose[:, 3], right_delta_ee_pose[:, 4], right_delta_ee_pose[:, 5])
        self.right_ee_target_pose[:, 3:7] = math_util.quat_mul(self.robot.data.body_state_w[:, self.right_ee_idx, 3:7], right_delta_ee_quat)
        
    def _get_observations(self) -> dict:
        # get image obs for VR
        left_eye_obs = self.left_eye_camera.data.output["rgb"][:, :, :, :3]
        right_eye_obs = self.right_eye_camera.data.output["rgb"][:, :, :, :3]
        rgb = torch.cat((left_eye_obs, right_eye_obs), dim=2)
        left_ee_pose = self.robot.data.body_state_w[:, self.left_ee_idx, 0:7]
        right_ee_pose = self.robot.data.body_state_w[:, self.right_ee_idx, 0:7]
        left_finger_tips = self.robot.data.body_state_w[:, self.left_finger_tips_idx, 0:3]
        right_finger_tips = self.robot.data.body_state_w[:, self.right_finger_tips_idx, 0:3]
        # convert to local env frame
        left_ee_pose[:, 0:3] -= self.scene.env_origins
        right_ee_pose[:, 0:3] -= self.scene.env_origins
        # shape of finger tips is (num_envs, num_fingers, 3)
        # need to extend env origins to match the shape of finger tips
        env_origins_extended = self.scene.env_origins.unsqueeze(1)
        left_finger_tips -= env_origins_extended
        right_finger_tips -= env_origins_extended
        self.left_finger_pos_mean = left_finger_tips.mean(dim=1)
        self.right_finger_pos_mean = right_finger_tips.mean(dim=1)
        left_hand_contact_force = torch.norm(self.left_hand_contact_sensor.data.net_forces_w, dim=-1)
        right_hand_contact_force = torch.norm(self.right_hand_contact_sensor.data.net_forces_w, dim=-1)
        self._get_target_ee_pose()

        return {
            "rgb": rgb,
            "fixed_rgb": self.main_camera.data.output["rgb"][:, :, :, :3],
            "fixed_d": self.main_camera.data.output["distance_to_image_plane"][:, :, :],
            "left_hand_rgb": self.left_hand_camera.data.output["rgb"][:, :, :, :3],
            "right_hand_rgb": self.right_hand_camera.data.output["rgb"][:, :, :, :3],
            "qpos": self.robot.data.joint_pos,
            "qvel": self.robot.data.joint_vel,
            "action": self.actions,
            "left_ee_pose": left_ee_pose,
            "right_ee_pose": right_ee_pose,
            "left_target_ee_pose": self.left_ee_target_pose,
            "right_target_ee_pose": self.right_ee_target_pose,
            "left_finger_tip_pos": left_finger_tips,
            "right_finger_tip_pos": right_finger_tips,
            "left_hand_contact_force": left_hand_contact_force,
            "right_hand_contact_force": right_hand_contact_force,
        }

    def _get_rewards(self) -> None:
        pass
