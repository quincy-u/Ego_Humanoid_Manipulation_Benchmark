from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.can import CAN_FANTA_CFG
from humanoid.tasks.data.container import CONTAINER_2X3_CFG
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.utils import configclass

import torch


@configclass
class InsertCansEnvCfg(BaseEnvCfg):
    # cans
    can_fanta_1: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/CanFanta1")  # type: ignore
    can_fanta_1.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.37, -0.3, 1.02), rot=(1, 0.0, 0.0, 0.0))
    can_fanta_2: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/CanFanta2")  # type: ignore
    can_fanta_2.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.37, 0.3, 1.02), rot=(1, 0.0, 0.0, 0.0))
    # container
    container: RigidObjectCfg = CONTAINER_2X3_CFG.replace(prim_path="/World/envs/env_.*/Container")  # type: ignore
    container.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.48, 0.0, 1.06), rot=(1, 0.0, 0.0, 0.0))


class InsertCansEnv(BaseEnv):
    cfg: InsertCansEnvCfg

    def __init__(self, cfg: InsertCansEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cans = [self.can_fanta_1, self.can_fanta_2]
        self.success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.container_xy_upper = torch.tensor([0.64, 0.1], device=self.device)
        self.container_xy_lower = torch.tensor([0.32, -0.1], device=self.device)
        self.reach_fanta1_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_fanta2_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_fanta1_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_fanta2_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_success = torch.zeros(self.num_envs, device=self.device)
        self.insert_success = torch.zeros(self.num_envs, device=self.device)

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
        for can_idx, can in enumerate(self.cans):
            can_root_state = can.data.default_root_state[env_ids]
            if self.cfg.randomize:
                if self.cfg.randomize_idx < 0:
                    can_root_state[:, 0:2] += (self.cfg.randomize_range*0.12) * torch.rand((self.num_envs, 2), device=self.device) - (self.cfg.randomize_range*0.06)
                else:
                    left_can_idx = self.cfg.randomize_idx // 100
                    left_can_column_idx = left_can_idx // 10
                    left_can_row_idx = left_can_idx % 10
                    right_can_idx = self.cfg.randomize_idx % 100
                    right_can_column_idx = right_can_idx // 10
                    right_can_row_idx = right_can_idx % 10
                    if can_idx == 0: # can_idx 0 is can_fanta_1, which is on the right side
                        can_root_state[:, 1] += 0.06 - (0.12 / 9) * right_can_row_idx
                        can_root_state[:, 0] += 0.06 - (0.12 / 9) * right_can_column_idx
                    elif can_idx == 1: # can_idx 1 is can_fanta_2, which is on the left side
                        can_root_state[:, 1] += 0.06 - (0.12 / 9) * left_can_row_idx
                        can_root_state[:, 0] += 0.06 - (0.12 / 9) * left_can_column_idx
            can_root_state[:, 0:3] += self.scene.env_origins[env_ids, :]
            can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
        container_root_state = self.container.data.default_root_state[env_ids]
        container_root_state[:, 0:3] += self.scene.env_origins[env_ids, :]
        self.container.write_root_state_to_sim(container_root_state, env_ids=env_ids)
        self.reach_fanta1_success[env_ids] = 0
        self.reach_fanta2_success[env_ids] = 0
        self.lift_fanta1_success[env_ids] = 0
        self.lift_fanta2_success[env_ids] = 0
        self.reach_success[env_ids] = 0
        self.lift_success[env_ids] = 0
        self.insert_success[env_ids] = 0

    def _get_success(self):
        self.insert_success[:] = 0
        for env_id in range(self.num_envs):
            all_can_inserted = True
            for can in [self.can_fanta_1,self.can_fanta_2]:
                curr_can_inserted = True
                if can.data.root_state_w[env_id, 2] > 1.065:
                    all_can_inserted = False
                    curr_can_inserted = False
                if can.data.root_state_w[env_id, 0] < self.container_xy_lower[0]\
                    or can.data.root_state_w[env_id, 1] < self.container_xy_lower[1]\
                    or can.data.root_state_w[env_id, 0] > self.container_xy_upper[0]\
                    or can.data.root_state_w[env_id, 1] > self.container_xy_upper[1]:
                        all_can_inserted = False
                        curr_can_inserted = False
                if curr_can_inserted:
                    self.insert_success[env_id] += 1
            self.success[env_id] = 1 if all_can_inserted else 0
            
    def _get_subtask(self):
        hand_can_dist = torch.norm(self.right_finger_pos_mean - self.can_fanta_1.data.root_state_w[:, 0:3], dim=-1)
        self.reach_fanta1_success[torch.where(hand_can_dist < 0.12)] = 1
        hand_can_dist = torch.norm(self.left_finger_pos_mean - self.can_fanta_2.data.root_state_w[:, 0:3], dim=-1)
        self.reach_fanta2_success[torch.where(hand_can_dist < 0.12)] = 1
        self.reach_success = self.reach_fanta1_success + self.reach_fanta2_success
        self.lift_fanta1_success[torch.where(self.can_fanta_1.data.root_state_w[:, 2] > 1.08)] = 1
        self.lift_fanta2_success[torch.where(self.can_fanta_2.data.root_state_w[:, 2] > 1.08)] = 1
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
        obs['insert_success'] = self.insert_success
        print(self.lift_success)
        print(self.insert_success)
        return obs
