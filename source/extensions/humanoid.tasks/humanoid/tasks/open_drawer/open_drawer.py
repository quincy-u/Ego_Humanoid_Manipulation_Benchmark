from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.drawer.drawer import DRAWER_CFG, DRAWER_JOINT_CFG
from humanoid.tasks.data.cube import CUBE_CFG
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.sim.simulation_cfg import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import torch
import omni.isaac.lab.sim as sim_utils

@configclass
class OpenDrawerEnvCfg(BaseEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=8,
        physx=sim_utils.PhysxCfg(
        )
    )
    # drawer
    drawer = DRAWER_CFG.replace(prim_path="/World/envs/env_.*/Drawer")  # type: ignore
    drawer.spawn.scale = (0.75, 0.75, 0.6)
    drawer.init_state.pos = (0.6, 0.0, 1.0)
    drawer.init_state.rot = (0, 0, 0, 1)
    drawer_mesh_idx = 1
    drawer_init_state = 'close' # "open", "close"
    drawer_joint_cfg = DRAWER_JOINT_CFG
    decimation = 8

class OpenDrawerEnv(BaseEnv):
    cfg: OpenDrawerEnvCfg

    def __init__(self, cfg: OpenDrawerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.success = torch.zeros(self.num_envs, dtype=torch.int32)
        # get drawer top and bottom joints idx
        self.cfg.drawer_joint_cfg.resolve(self.scene)
        if self.cfg.drawer_joint_cfg.joint_ids == slice(None): # isaaclab set joint_ids to Slice(None) if joint_names same as in cfg
            self.drawer_top_idx = 0
            self.drawer_bottom_idx = 1
        else:
            self.drawer_top_idx = self.cfg.drawer_joint_cfg.joint_ids[0] # type: ignore
            self.drawer_bottom_idx = self.cfg.drawer_joint_cfg.joint_ids[1] # type: ignore
        self.is_drawer_full_open = torch.zeros(self.num_envs, device=self.device)
        self.drawer_handle_offset = torch.tensor([-0.08, -0.122, -0.05], device=self.device)
        self.reach_success = torch.zeros(self.num_envs, device=self.device)
        self.move_door_success = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._update_cfg_mesh(self.cfg.drawer, self.cfg.drawer_mesh_idx)
        self.drawer = Articulation(self.cfg.drawer)
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        self.scene.articulations["drawer"] = self.drawer
    
    def _get_joints_data(self) -> None:
        self.is_drawer_full_open[:] = 0
        
        drawer_bottom_upper = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 1]
        drawer_bottom_lower = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 0]
        drawer_full_open_limit = drawer_bottom_lower + 0.1 * (drawer_bottom_upper - drawer_bottom_lower)
        self.is_drawer_full_open[torch.where(self.drawer.data.joint_pos[:, self.drawer_bottom_idx] < drawer_full_open_limit)] = 1
        
    def _get_subtask(self):
        drawer_handle_pos_w = self.drawer.data.root_state_w[:, 0:3] + self.drawer_handle_offset
        hand_handle_dist = torch.norm(self.left_finger_pos_mean - drawer_handle_pos_w, dim=-1)
        self.reach_success[torch.where(hand_handle_dist < 0.12)] = 1
        drawer_bottom_upper = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 1]
        drawer_bottom_lower = self.drawer.data.joint_limits[:, self.drawer_bottom_idx, 0]
        drawer_non_close_limit = drawer_bottom_upper - 0.1 * (drawer_bottom_upper - drawer_bottom_lower)
        self.move_door_success[torch.where(self.drawer.data.joint_pos[:, self.drawer_bottom_idx] < drawer_non_close_limit)] = 1
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        joint_pos = self.drawer.data.default_joint_pos[env_ids]
        bottom_joint_bound = 1 if self.cfg.drawer_init_state == "close" else 0
        joint_pos[:, self.drawer_top_idx] = self.drawer.data.joint_limits[env_ids, self.drawer_top_idx, 1].squeeze()
        joint_pos[:, self.drawer_bottom_idx] = self.drawer.data.joint_limits[env_ids, self.drawer_bottom_idx, bottom_joint_bound].squeeze()
        joint_pos = torch.clamp(joint_pos, self.drawer.data.soft_joint_pos_limits[0, :, 0], self.drawer.data.soft_joint_pos_limits[0, :, 1])
        joint_vel = torch.zeros_like(joint_pos)
        self.drawer.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        drawer_root_state = self.drawer.data.default_root_state[env_ids].clone()
        drawer_root_state[:, 2] = 1.4017
        if self.cfg.randomize:
            if self.cfg.randomize_idx < 0:
                drawer_root_state[:, 0:2] += (0.2 * self.cfg.randomize_range) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range)
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                drawer_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                drawer_root_state[:, 0] += 0.05 - (0.1 / 99) * column_idx
        self.drawer.write_root_state_to_sim(drawer_root_state, env_ids=env_ids)
        self.reach_success[:] = 0
        self.move_door_success[:] = 0
        
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_joints_data()
        self._get_subtask()
        obs["success"] = self.is_drawer_full_open
        obs["reach_success"] = self.reach_success
        obs["move_door_success"] = self.move_door_success
        return obs