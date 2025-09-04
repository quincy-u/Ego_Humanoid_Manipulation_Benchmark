from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.drawer.drawer import DRAWER_CFG, DRAWER_JOINT_CFG
from humanoid.tasks.data.can import CAN_FANTA_CFG
from humanoid.tasks.data.h1.h1_inspire import H1_INSPIRE_CFG
from humanoid.tasks.data.plate import PLATE_CFG
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.sim.simulation_cfg import SimulationCfg
from omni.isaac.lab.utils import configclass

import torch

@configclass
class StackCanIntoDrawerEnvCfg(BaseEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=8,
    )
    can: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/Can")  # type: ignore
    can.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.0, 1.02), rot=(1, 0, 0, 0))
    plate: RigidObjectCfg = PLATE_CFG.replace(prim_path="/World/envs/env_.*/Plate")  # type: ignore
    plate.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 1.17), rot=(1, 0, 0, 0))
    plate.spawn.scale = (0.6, 0.6, 0.6) # type: ignore
    # drawer
    drawer = DRAWER_CFG.replace(prim_path="/World/envs/env_.*/Drawer")  # type: ignore
    drawer.spawn.scale = (0.75, 0.75, 0.6)
    drawer.init_state.pos = (0.53, 0.0, 1.4017)
    drawer.init_state.rot = (0, 0, 0, 1)
    drawer_mesh_idx = 1
    drawer_init_state = 'open' # "open", "close"
    drawer_joint_cfg = DRAWER_JOINT_CFG
    # need to override robot init joint state to avoid collision with drawer door
    robot: ArticulationCfg = H1_INSPIRE_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore
    robot.init_state.joint_pos[".*_shoulder_yaw_joint"] = 1.5
    # fine tune robot initial joint pos so that it doesnt not collide with drawer door
    decimation = 8

class StackCanIntoDrawerEnv(BaseEnv):
    cfg: StackCanIntoDrawerEnvCfg

    def __init__(self, cfg: StackCanIntoDrawerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.plate_radius = 0.09
        self.success = torch.zeros(self.num_envs, dtype=torch.int32)
        # get drawer top and bottom joints idx
        self.cfg.drawer_joint_cfg.resolve(self.scene)
        if self.cfg.drawer_joint_cfg.joint_ids == slice(None): # isaaclab set joint_ids to Slice(None) if joint_names same as in cfg
            self.drawer_top_idx = 0
            self.drawer_bottom_idx = 1
        else:
            self.drawer_top_idx = self.cfg.drawer_joint_cfg.joint_ids[0] # type: ignore
            self.drawer_bottom_idx = self.cfg.drawer_joint_cfg.joint_ids[1] # type: ignore
        self.drawer_handle_offset = torch.tensor([-0.08, -0.122, -0.05], device=self.device)
        self.is_drawer_close = torch.zeros(self.num_envs, device=self.device)
        self.reach_drawer_success = torch.zeros(self.num_envs, device=self.device)
        self.move_door_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_can_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_success = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.can = RigidObject(self.cfg.can)
        self.plate = RigidObject(self.cfg.plate)
        self._update_cfg_mesh(self.cfg.drawer, self.cfg.drawer_mesh_idx)
        self.drawer = Articulation(self.cfg.drawer)
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects['can'] = self.can
        self.scene.rigid_objects['plate'] = self.plate
        self.scene.articulations["drawer"] = self.drawer
    
    def _get_joints_data(self) -> None:
        self.is_drawer_close[:] = 0
        drawer_bottom_upper = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 1]
        self.is_drawer_close[torch.where(self.drawer.data.joint_pos[:, self.drawer_bottom_idx] > drawer_bottom_upper * 0.9)] = 1

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        # reset drawer joints
        joint_pos = self.drawer.data.default_joint_pos[env_ids]
        bottom_joint_bound = 1 if self.cfg.drawer_init_state == "close" else 0
        joint_pos[:, self.drawer_top_idx] = self.drawer.data.joint_limits[env_ids, self.drawer_top_idx, 1].squeeze()
        joint_pos[:, self.drawer_bottom_idx] = self.drawer.data.joint_limits[env_ids, self.drawer_bottom_idx, bottom_joint_bound].squeeze()
        joint_pos = torch.clamp(joint_pos, self.drawer.data.soft_joint_pos_limits[0, :, 0], self.drawer.data.soft_joint_pos_limits[0, :, 1])
        joint_vel = torch.zeros_like(joint_pos)
        self.drawer.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # reset drawer, plate and can positions
        can_root_state = self.can.data.default_root_state[env_ids, :]
        plate_root_state = self.plate.data.default_root_state[env_ids, :]
        drawer_root_state = self.drawer.data.default_root_state[env_ids, :]
        if self.cfg.randomize: 
            if self.cfg.randomize_idx < 0:
                drawer_noise = (self.cfg.randomize_range * 0.1) * torch.rand((self.num_envs, 2), device=self.device) - (0.05 * self.cfg.randomize_range) 
                drawer_root_state[:, 0:2] += drawer_noise
                can_root_state[:, 0:2] += drawer_noise
                plate_root_state[:, 0:2] += (self.cfg.randomize_range * 0.1) * torch.rand((self.num_envs, 2), device=self.device) - (0.05 * self.cfg.randomize_range) 
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                drawer_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                plate_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                can_root_state[:, 1] += 0.1 - (0.2/ 99) * row_idx
                can_root_state[:, 0] += 0.05 - (0.1 / 99) * column_idx
        self.drawer.write_root_state_to_sim(drawer_root_state, env_ids=env_ids)
        self.can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
        self.plate.write_root_state_to_sim(plate_root_state, env_ids=env_ids)

        self.reach_drawer_success[:] = 0
        self.move_door_success[:] = 0
        self.reach_can_success[:] = 0
        self.lift_success[:] = 0
        
    def _get_subtask(self):
        drawer_handle_pos_w = self.drawer.data.root_state_w[:, 0:3] + self.drawer_handle_offset
        left_hand_handle_dist = torch.norm(self.left_finger_pos_mean - drawer_handle_pos_w, dim=-1)
        self.reach_drawer_success[torch.where(left_hand_handle_dist < 0.12)] = 1
        right_hand_can_dist = torch.norm(self.right_finger_pos_mean - self.can.data.root_state_w[:, 0:3], dim=-1)
        self.reach_can_success[torch.where(right_hand_can_dist < 0.12)] = 1
        drawer_bottom_upper = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 1]
        drawer_bottom_lower = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 0]
        drawer_non_close_limit = drawer_bottom_upper - 0.1 * (drawer_bottom_upper - drawer_bottom_lower)
        self.move_door_success[torch.where(self.drawer.data.joint_pos[:, self.drawer_bottom_idx] < drawer_non_close_limit)] = 1
        self.lift_success[torch.where(self.can.data.root_state_w[:, 2] > 1.08)] = 1

    def _get_success(self):
        self.success[:] = 0
        dist_horziontal = torch.norm(self.can.data.root_state_w[:, :2] - self.plate.data.root_state_w[:, :2], p=2, dim=-1)
        dist_vertical = torch.abs(self.can.data.root_state_w[:, 2] - self.plate.data.root_state_w[:, 2])
        self.success[torch.where(dist_horziontal < self.plate_radius and dist_vertical < 0.02 and self.is_drawer_close)] = 1
        
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_joints_data()
        self._get_subtask()
        self._get_success()
        obs["success"] = self.success
        obs["reach_drawer_success"] = self.reach_drawer_success
        obs["reach_can_success"] = self.reach_can_success
        obs["lift_success"] = self.lift_success
        return obs