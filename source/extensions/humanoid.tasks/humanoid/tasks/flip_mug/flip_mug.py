from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.mug import MUG_CFG
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_util
import torch

BALL_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.006,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.0, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.001,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.35, 1.1)),
) 

@configclass
class FlipMugEnvCfg(BaseEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240, 
        render_interval=8,
        physx=PhysxCfg(
            enable_ccd=True
        )
    )
    # mug that balls are poured into
    mug: RigidObjectCfg = MUG_CFG.replace(prim_path="/World/envs/env_.*/mug")  # type: ignore
    mug.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 1.05), rot=(0.6065, -0.3590,  0.5960, -0.3849))
    mug.spawn.scale = (1.7, 1.7, 1.0) # type: ignore
    mug_mesh_idx = 1
    decimation = 8

class FlipMugEnv(BaseEnv):
    cfg: FlipMugEnvCfg 

    def __init__(self, cfg: FlipMugEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.reach_mug_success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
            
    def _setup_scene(self):
        self._update_cfg_mesh(self.cfg.mug, self.cfg.mug_mesh_idx)
        self.mug = RigidObject(self.cfg.mug)
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects['mug'] = self.mug

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        mug_root_state = self.mug.data.default_root_state[env_ids, :]
        if self.cfg.randomize:
            if self.cfg.randomize_idx < 0:
                mug_root_state[:, 0:2] += (0.2 * self.cfg.randomize_range) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range) 
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                mug_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                mug_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx
        mug_root_state[env_ids, 0:3] += self.scene.env_origins[env_ids, 0:3]
        self.mug.write_root_state_to_sim(mug_root_state, env_ids=env_ids)
        self.reach_mug_success[env_ids] = 0
        
    def _get_subtask(self):
        left_hand_mug_dist = torch.norm(self.left_finger_pos_mean - self.mug.data.root_state_w[:, 0:3], dim=-1)
        self.reach_mug_success[torch.where(left_hand_mug_dist < 0.12)] = 1
        
    def _get_success(self):
        self.success[:] = 0
        for env_id in range(self.num_envs):
            # check if mug is flipped upwards up
            z_unit_mug_frame = math_util.quat_apply(torch.tensor(self.mug.data.root_quat_w, device=self.device), torch.tensor((0, 0, 1), device=self.device, dtype=torch.float32))
            mug_flipped = True if z_unit_mug_frame[2] > 0.5 else False
            self.success[env_id] = 1 if mug_flipped else 0

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_success()
        self._get_subtask()
        obs['success'] = self.success
        obs['reach_success'] = self.reach_mug_success
        return obs