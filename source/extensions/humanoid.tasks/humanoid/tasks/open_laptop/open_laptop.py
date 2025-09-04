from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.laptop import LAPTOP_CFG
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg, RigidObject
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg, Articulation
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab.utils.math as math_util 
import omni.isaac.lab.sim as sim_utils
import torch

@configclass
class OpenLaptopEnvCfg(BaseEnvCfg):
    laptop: ArticulationCfg = LAPTOP_CFG.replace(prim_path="/World/envs/env_.*/laptop") # type: ignore
    laptop.init_state.pos = (0.6, 0.0, 1.03)
    laptop.init_state.rot = (0.5, 0.5, 0.5, 0.5)
    decimation = 4

class OpenLaptopEnv(BaseEnv):
    cfg: OpenLaptopEnvCfg 

    def __init__(self, cfg: OpenLaptopEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.success = torch.zeros([self.num_envs], device=self.device)
        self.move_lid_success = torch.zeros([self.num_envs], device=self.device)
            
    def _setup_scene(self):
        self.laptop = Articulation(self.cfg.laptop)
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        self.scene.articulations['laptop'] = self.laptop

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        laptop_root_state = self.laptop.data.default_root_state[env_ids] # type: ignore
        laptop_root_state[:, 0:3] += self.scene.env_origins[env_ids, :]
        if self.cfg.randomize:
            if self.cfg.randomize_idx < 0:
                laptop_root_state[:, 0:2] += (0.20 * self.cfg.randomize_range) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range)
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                laptop_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                laptop_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx
        self.laptop.write_root_state_to_sim(laptop_root_state, env_ids=env_ids)

        joint_pos = self.laptop.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.laptop.data.soft_joint_pos_limits[0, :, 0], self.laptop.data.soft_joint_pos_limits[0, :, 1]) # type: ignore
        joint_vel = torch.zeros_like(joint_pos)
        self.laptop.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.move_lid_success[env_ids] = 0
        
    def _get_subtask(self):
        laptop_upper = self.laptop.data.joint_limits[:, 0, 1]
        self.move_lid_success[torch.where(self.laptop.data.joint_pos[:, 0] > laptop_upper * 0.15)] = 1
        
    def _get_success(self):
        laptop_upper = self.laptop.data.joint_limits[:, 0, 1]
        self.success[torch.where(self.laptop.data.joint_pos[:, 0] > laptop_upper * 0.7)] = 1

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_success()
        self._get_subtask()
        obs['success'] = self.success
        obs['move_lid_success'] = self.move_lid_success
        print(self.move_lid_success)
        return obs