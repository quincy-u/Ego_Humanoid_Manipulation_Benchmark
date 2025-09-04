from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.can import CAN_FANTA_CFG
from humanoid.tasks.data.plate import PLATE_CFG
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import torch
import omni.isaac.lab.sim as sim_utils

@configclass
class StackCanEnvCfg(BaseEnvCfg):
    can: RigidObjectCfg = CAN_FANTA_CFG.replace(prim_path="/World/envs/env_.*/Can")  # type: ignore
    can.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.25, 1.02), rot=(1, 0, 0, 0))
    plate: RigidObjectCfg = PLATE_CFG.replace(prim_path="/World/envs/env_.*/Plate")  # type: ignore
    plate.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 1.02), rot=(1, 0, 0, 0))
    plate.spawn.scale = (0.6, 0.6, 0.6) # type: ignore

class StackCanEnv(BaseEnv):
    cfg: StackCanEnvCfg

    def __init__(self, cfg: StackCanEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.plate_radius = 0.09
        self.success = torch.zeros(self.num_envs, device=self.device)
        self.place_success = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.can = RigidObject(self.cfg.can)
        self.plate = RigidObject(self.cfg.plate)
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects['can'] = self.can
        self.scene.rigid_objects['plate'] = self.plate

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        can_root_state = self.can.data.default_root_state[env_ids, :]
        plate_root_state = self.plate.data.default_root_state[env_ids, :]
        if self.cfg.randomize:
            if self.cfg.randomize_idx < 0:
                plate_root_state[:, 0:2] += (self.cfg.randomize_range * 0.2) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range) 
                can_root_state[:, 0:2] += (self.cfg.randomize_range * 0.2) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range)
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                plate_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                plate_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx
                can_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                can_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx
        self.can.write_root_state_to_sim(can_root_state, env_ids=env_ids)
        self.plate.write_root_state_to_sim(plate_root_state, env_ids=env_ids)
        self.place_success[env_ids] = 0
        
    def _get_subtask(self):
        dist_horziontal = torch.norm(self.can.data.root_state_w[:, :2] - self.plate.data.root_state_w[:, :2], p=2, dim=-1)
        dist_vertical = torch.abs(self.can.data.root_state_w[:, 2] - self.plate.data.root_state_w[:, 2])
        self.place_success[torch.where(dist_horziontal < self.plate_radius and dist_vertical < 0.05)] = 1

    def _get_success(self):
        self.success[:] = 0
        dist_horziontal = torch.norm(self.can.data.root_state_w[:, :2] - self.plate.data.root_state_w[:, :2], p=2, dim=-1)
        dist_vertical = torch.abs(self.can.data.root_state_w[:, 2] - self.plate.data.root_state_w[:, 2])
        self.success[torch.where(dist_horziontal < self.plate_radius and dist_vertical < 0.02)] = 1
        
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_success()
        self._get_subtask()
        obs["success"] = self.success
        obs['place_success'] = self.place_success
        return obs