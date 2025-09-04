from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.can import CAN_SPRITE_CFG, CAN_FANTA_CFG
from humanoid.tasks.data.container import CONTAINER_PLASTIC_CFG
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.utils import configclass

import torch

@configclass
class SortCansEnvCfg(BaseEnvCfg):
    # cans
    can_sprite_1: RigidObjectCfg = CAN_SPRITE_CFG.replace(prim_path="/World/envs/env_.*/CanSprite1")  # type: ignore
    can_sprite_1.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.42, -0.33, 1.02), rot=(1, 0.0, 0.0, 0.0))
    can_sprite_2: RigidObjectCfg = CAN_SPRITE_CFG.replace(prim_path="/World/envs/env_.*/CanSprite2")  # type: ignore
    can_sprite_2.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.42, 0.33, 1.02), rot=(1, 0.0, 0.0, 0.0))
    can_fanta_1: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/CanRed1")  # type: ignore
    can_fanta_1.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.52, -0.33, 1.02), rot=(1, 0.0, 0.0, 0.0))
    can_fanta_2: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/CanRed2")  # type: ignore
    can_fanta_2.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.33, 1.02), rot=(1, 0.0, 0.0, 0.0))
    # container
    container_1: AssetBaseCfg = CONTAINER_PLASTIC_CFG.replace(prim_path="/World/envs/env_.*/Container1")  # type: ignore
    container_1.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.085, 1.1), rot=(0.70711, 0.0, 0.0, -0.70711))
    container_2: AssetBaseCfg = CONTAINER_PLASTIC_CFG.replace(prim_path="/World/envs/env_.*/Container2")  # type: ignore
    container_2.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.6, -0.085, 1.1), rot=(0.70711, 0.0, 0.0, -0.70711))

class SortCansEnv(BaseEnv):
    cfg: SortCansEnvCfg

    def __init__(self, cfg: SortCansEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # constants
        self.right_container_top_left_pos = torch.tensor([0.72, -0.015, 1.15], device=self.device)
        self.right_container_bot_right_pos = torch.tensor([0.47, -0.16, 1.02], device=self.device)
        self.left_container_top_left_pos = torch.tensor([0.72, 0.1592, 1.15], device=self.device)
        self.left_container_bot_right_pos = torch.tensor([0.47, 0.007, 1.02], device=self.device)
        self.success = torch.zeros(self.num_envs, device=self.device)
        self.reach_fanta1_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_fanta2_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_sprite1_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_sprite2_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_fanta1_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_fanta2_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_sprite1_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_sprite2_success = torch.zeros(self.num_envs, device=self.device)
        self.reach_success = torch.zeros(self.num_envs, device=self.device)
        self.lift_success = torch.zeros(self.num_envs, device=self.device)
        self.sort_success = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.can_sprite_1 = RigidObject(self.cfg.can_sprite_1)
        self.can_sprite_2 = RigidObject(self.cfg.can_sprite_2)
        self.can_fanta_1 = RigidObject(self.cfg.can_fanta_1)
        self.can_fanta_2 = RigidObject(self.cfg.can_fanta_2)
        # add containers
        self.cfg.container_1.spawn.func( # type: ignore
            self.cfg.container_1.prim_path,
            self.cfg.container_1.spawn,
            translation=self.cfg.container_1.init_state.pos,
            orientation=self.cfg.container_1.init_state.rot,
        )
        self.cfg.container_2.spawn.func( # type: ignore
            self.cfg.container_2.prim_path,
            self.cfg.container_2.spawn,
            translation=self.cfg.container_2.init_state.pos,
            orientation=self.cfg.container_2.init_state.rot,
        )
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects["can_sprite_1"] = self.can_sprite_1
        self.scene.rigid_objects["can_sprite_2"] = self.can_sprite_2
        self.scene.rigid_objects["can_fanta_1"] = self.can_fanta_1
        self.scene.rigid_objects["can_fanta_2"] = self.can_fanta_2

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        # reset object
        for can_idx, can in enumerate([self.can_sprite_1, self.can_sprite_2, self.can_fanta_1, self.can_fanta_2]):
            can_root_state = can.data.default_root_state[env_ids]
            if self.cfg.randomize:
                if self.cfg.randomize_idx < 0:
                    can_root_state[:, 0:2] += (self.cfg.randomize_range *0.12) * torch.rand((self.num_envs, 2), device=self.device) - (0.06 * self.cfg.randomize_range)
                else:
                    left_cans_idx = self.cfg.randomize_idx // 100
                    left_cans_column_idx = left_cans_idx // 10
                    left_cans_row_idx = left_cans_idx % 10
                    right_cans_idx = self.cfg.randomize_idx % 100
                    right_cans_column_idx = right_cans_idx // 10
                    right_cans_row_idx = right_cans_idx % 10
                    if can_idx == 0 or can_idx == 2: # can_sprite_1 and can_fanta_1 are on the right side
                        can_root_state[:, 1] += 0.06 - (0.12 / 9) * right_cans_row_idx
                        can_root_state[:, 0] += 0.06 - (0.12 / 9) * right_cans_column_idx
                    elif can_idx == 1 or can_idx == 3: # can_sprite_2 and can_fanta_2 are on the left side
                        can_root_state[:, 1] += 0.06 - (0.12 / 9) * left_cans_row_idx
                        can_root_state[:, 0] += 0.06 - (0.12 / 9) * left_cans_column_idx
            can_root_state[:, 0:3] += self.scene.env_origins[env_ids, :]
            can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
        self.reach_fanta1_success[env_ids] = 0
        self.reach_fanta2_success[env_ids] = 0
        self.reach_sprite1_success[env_ids] = 0
        self.reach_sprite2_success[env_ids] = 0
        self.lift_fanta1_success[env_ids] = 0
        self.lift_fanta2_success[env_ids] = 0
        self.lift_sprite1_success[env_ids] = 0
        self.lift_sprite2_success[env_ids] = 0
        self.reach_success[env_ids] = 0
        self.lift_success[env_ids] = 0
        self.sort_success[env_ids] = 0
    
    def _get_subtask(self):
        hand_can_dist = torch.norm(self.right_finger_pos_mean - self.can_fanta_1.data.root_state_w[:, 0:3], dim=-1)
        self.reach_fanta1_success[torch.where(hand_can_dist < 0.12)] = 1
        hand_can_dist = torch.norm(self.right_finger_pos_mean - self.can_sprite_1.data.root_state_w[:, 0:3], dim=-1)
        self.reach_sprite1_success[torch.where(hand_can_dist < 0.12)] = 1
        hand_can_dist = torch.norm(self.left_finger_pos_mean - self.can_fanta_2.data.root_state_w[:, 0:3], dim=-1)
        self.reach_fanta2_success[torch.where(hand_can_dist < 0.12)] = 1
        hand_can_dist = torch.norm(self.left_finger_pos_mean - self.can_sprite_2.data.root_state_w[:, 0:3], dim=-1)
        self.reach_sprite2_success[torch.where(hand_can_dist < 0.12)] = 1
        self.reach_success = self.reach_fanta1_success + self.reach_fanta2_success + self.reach_sprite1_success + self.reach_sprite2_success
        self.lift_fanta1_success[torch.where(self.can_fanta_1.data.root_state_w[:, 2] > 1.08)] = 1
        self.lift_sprite1_success[torch.where(self.can_sprite_1.data.root_state_w[:, 2] > 1.08)] = 1
        self.lift_fanta2_success[torch.where(self.can_fanta_2.data.root_state_w[:, 2] > 1.08)] = 1
        self.lift_sprite2_success[torch.where(self.can_sprite_2.data.root_state_w[:, 2] > 1.08)] = 1
        self.lift_success = self.lift_fanta1_success + self.lift_fanta2_success + self.lift_sprite1_success + self.lift_sprite2_success

    def _get_success(self) -> None:
        self.sort_success[:] = 0
        for env_id in range(self.num_envs):
            fanta_sorted = True
            sprite_sorted = True
            for fanta in [self.can_fanta_1, self.can_fanta_2]:
                curr_fanta_sorted = True
                fanta_pos = fanta.data.root_state_w[env_id, :3] - self.scene.env_origins[env_id, :]
                if (len(fanta_pos.shape) == 1):
                    fanta_pos = fanta_pos.unsqueeze(0)
                for i in range(3):
                    if self.right_container_bot_right_pos[i] > fanta_pos[env_id, i] or fanta_pos[env_id, i] > self.right_container_top_left_pos[i]:
                        fanta_sorted = False
                        curr_fanta_sorted = False
                if curr_fanta_sorted:
                    self.sort_success[env_id] += 1
            for sprite in [self.can_sprite_1, self.can_sprite_2]:
                curr_sprite_sorted = True
                sprite_pos = sprite.data.root_state_w[env_id, :3] - self.scene.env_origins[env_id, :]
                if (len(sprite_pos.shape) == 1):
                    sprite_pos = sprite_pos.unsqueeze(0)
                for i in range(3):
                    if self.left_container_bot_right_pos[i] > sprite_pos[env_id, i] or sprite_pos[env_id, i] > self.left_container_top_left_pos[i]:
                        sprite_sorted = False
                        curr_sprite_sorted = False
                if curr_sprite_sorted:
                    self.sort_success[env_id] += 1
            if fanta_sorted and sprite_sorted:
                self.success[env_id] = 1
            else:
                self.success[env_id] = 0
            
        
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_success()
        self._get_subtask()
        obs['success'] = self.success
        obs['reach_success'] = self.reach_success
        obs['lift_success'] = self.lift_success
        obs['sort_success'] = self.sort_success
        return obs