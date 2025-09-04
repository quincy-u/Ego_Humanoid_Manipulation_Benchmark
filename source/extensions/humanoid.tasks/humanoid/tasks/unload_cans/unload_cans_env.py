from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.can import CAN_FANTA_CFG
from humanoid.tasks.data.container import CONTAINER_2X3_CFG
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab.utils.math as math_util
import torch


@configclass
class UnloadCansEnvCfg(BaseEnvCfg):
    # cans
    can_fanta_1: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/CanFanta1")  # type: ignore
    can_fanta_1.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.38, -0.045, 1.08), rot=(1, 0.0, 0.0, 0.0))
    can_fanta_2: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/CanFanta2")  # type: ignore
    can_fanta_2.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.045, 1.08), rot=(1, 0.0, 0.0, 0.0))
    # container
    container: RigidObjectCfg = CONTAINER_2X3_CFG.replace(prim_path="/World/envs/env_.*/Container")  # type: ignore
    container.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.48, 0.0, 1.06), rot=(1, 0.0, 0.0, 0.0))


class UnloadCansEnv(BaseEnv):
    cfg: UnloadCansEnvCfg

    def __init__(self, cfg: UnloadCansEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cans = [self.can_fanta_1, self.can_fanta_2]
        self.success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.container_xy_upper = torch.tensor([0.64, 0.1], device=self.device)
        self.container_xy_lower = torch.tensor([0.32, -0.1], device=self.device)
        self.curr_episode_offset = torch.zeros(2, device=self.device)
        self.reach_fanta1_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_fanta2_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_fanta1_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_fanta2_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_success = torch.zeros(self.num_envs, device=self.device)
        self.unload_success = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.can_fanta_1 = RigidObject(self.cfg.can_fanta_1)
        self.can_fanta_2 = RigidObject(self.cfg.can_fanta_2)
        self.container = RigidObject(self.cfg.container)
        super()._setup_scene()

    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects["can_fanta_1"] = self.can_fanta_1
        self.scene.rigid_objects["can_fanta_2"] = self.can_fanta_2
        self.scene.rigid_objects["container"] = self.container

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self.curr_episode_offset = torch.zeros(2, device=self.device)
        if self.cfg.randomize: 
            if self.cfg.randomize_idx < 0:
                self.curr_episode_offset = (self.cfg.randomize_range*0.2) * torch.rand((2), device=self.device) - (self.cfg.randomize_range*0.1)
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                self.curr_episode_offset[1] += 0.1 - (0.2 / 99) * row_idx
                self.curr_episode_offset[0] += 0.05 - (0.1 / 99) * column_idx
        for can in self.cans:
            can_root_state = can.data.default_root_state[env_ids]
            can_root_state[:, :2] += self.curr_episode_offset
            can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
        container_root_state = self.container.data.default_root_state[env_ids]
        container_root_state[:, :2] += self.curr_episode_offset
        self.container.write_root_state_to_sim(container_root_state, env_ids=env_ids)
        self.reach_fanta1_success[env_ids] = 0
        self.reach_fanta2_success[env_ids] = 0
        self.lift_fanta1_success[env_ids] = 0
        self.lift_fanta2_success[env_ids] = 0
        self.reach_success[env_ids] = 0
        self.lift_success[env_ids] = 0
        self.unload_success[env_ids] = 0

    def _get_success(self):
        self.unload_success[:] = 0
        container_xy_upper = self.container_xy_upper + self.curr_episode_offset
        container_xy_lower = self.container_xy_lower + self.curr_episode_offset
        for env_id in range(self.num_envs):
            all_can_unloaded = True
            all_can_upwards = True
            for can in [self.can_fanta_1,self.can_fanta_2]:
                curr_can_unloaded = True
                if container_xy_lower[0] < can.data.root_state_w[env_id, 0]\
                    and can.data.root_state_w[env_id, 0] < container_xy_upper[0]\
                    and container_xy_lower[1] < can.data.root_state_w[env_id, 1]\
                    and can.data.root_state_w[env_id, 1] < container_xy_upper[1]:
                        all_can_unloaded = False
                        curr_can_unloaded = False
                z_unit_can_frame = math_util.quat_apply(torch.tensor(can.data.root_quat_w, device=self.device), torch.tensor((0, 0, 1), device=self.device, dtype=torch.float32))
                if z_unit_can_frame[2] < 0.5 or can.data.root_state_w[env_id, 2] < 1: # check if can is pointing upwards above table
                    all_can_upwards = False
                    curr_can_unloaded = False
                if curr_can_unloaded:
                    self.unload_success[env_id] += 1
            # self.success[env_id] = 1 if all_can_unloaded and all_can_upwards else 0
            self.success[env_id] = 1 if all_can_unloaded else 0
    
    def _get_subtask(self):
        self.reach_fanta1_success[torch.where(torch.norm(self.right_finger_pos_mean - self.can_fanta_1.data.root_state_w[:, 0:3], dim=-1) < 0.12)] = 1
        self.reach_fanta2_success[torch.where(torch.norm(self.left_finger_pos_mean - self.can_fanta_2.data.root_state_w[:, 0:3], dim=-1) < 0.12)] = 1
        self.reach_success = self.reach_fanta1_success + self.reach_fanta2_success
        self.lift_fanta1_success[torch.where(self.can_fanta_1.data.root_state_w[:, 2] > 1.14)] = 1
        self.lift_fanta2_success[torch.where(self.can_fanta_2.data.root_state_w[:, 2] > 1.14)] = 1
        self.lift_success = self.lift_fanta1_success + self.lift_fanta2_success

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        can_poses = []
        env_origins_extended = self.scene.env_origins.unsqueeze(1)
        for can in self.cans:
            can_pose = can.data.body_state_w[:, :, 0:7]
            can_pose[:, :, 0:3] -= env_origins_extended
            can_poses.append(can_pose)
        can_poses = torch.cat(can_poses, dim=1)
        obs["object_pose"] = can_poses
        self._get_success()
        self._get_subtask()
        obs['success'] = self.success
        obs['reach_success'] = self.reach_success
        obs['lift_success'] = self.lift_success
        obs['unload_success'] = self.unload_success
        print(self.lift_success)
        print(self.unload_success)
        return obs
