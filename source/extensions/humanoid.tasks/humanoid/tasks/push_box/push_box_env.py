from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.marker import MARKER_CFG
from omni.isaac.lab.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import  UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import sample_uniform

import torch
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.envs.mdp as mdp


@configclass
class PushBoxEnvCfg(BaseEnvCfg):
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.3, 1.04), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    # goal object
    goal_cfg: VisualizationMarkersCfg = MARKER_CFG.replace(prim_path="/Visuals/goal") # type: ignore
    goal_default_pos = (0.5, 0.0, 1.021)

class PushBoxEnv(BaseEnv):
    cfg: PushBoxEnvCfg

    def __init__(self, cfg: PushBoxEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.int32)
        self.reach_success = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.box = RigidObject(self.cfg.box)
        self.goal_marker = VisualizationMarkers(self.cfg.goal_cfg)
        super()._setup_scene()
        
    def _register_scene(self):
        super()._register_scene()
        self.scene.rigid_objects["box"] = self.box

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        box_root_state = self.box.data.default_root_state[env_ids, :]
        if self.cfg.randomize:
            self._reset_goal_pos(env_ids)
            if self.cfg.randomize_idx < 0:
                box_root_state[:, 0:2] += (self.cfg.randomize_range * 0.2) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range) 
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                box_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                box_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx
        box_root_state[:, 0:3] += self.scene.env_origins[env_ids, 0:3]
        self.box.write_root_state_to_sim(box_root_state, env_ids=env_ids)
        self.reach_success[env_ids] = 0
        
    def _reset_goal_pos(self, env_ids):
        if self.cfg.randomize_idx < 0:
            offset_x = sample_uniform(-0.03, 0.03, (len(env_ids)), device=self.device)
            offset_y = sample_uniform(-0.1, 0.1, (len(env_ids)), device=self.device)
        else:
            column_idx = (9999 - self.cfg.randomize_idx) // 100
            row_idx = (9999 - self.cfg.randomize_idx) % 100
            offset_x = 0.03 - (0.06 / 99) * row_idx
            offset_y = 0.1 - (0.2 / 99) * column_idx
        self.goal_pos_w[env_ids, :] = torch.tensor(self.cfg.goal_default_pos, device=self.device) + self.scene.env_origins[env_ids]
        self.goal_pos_w[env_ids, 0] += offset_x
        self.goal_pos_w[env_ids, 1] += offset_y 
        self.goal_marker.visualize(self.goal_pos_w)
    
    def _get_subtask(self):
        right_hand_box_dist = torch.norm(self.right_finger_pos_mean - self.box.data.root_state_w[:, 0:3], dim=-1)
        self.reach_success[torch.where(right_hand_box_dist < 0.13)] = 1

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self.success[:] = 0
        box_goal_dist = torch.norm(self.box.data.root_state_w[:, :2] - self.goal_pos_w[:, :2], p=2, dim=-1)
        self.success[torch.where(box_goal_dist < 0.08)] = 1
        box_pose = self.box.data.root_state_w[:, 0:7]
        box_pose[:, 0:3] -= self.scene.env_origins
        self._get_subtask()
        obs['object_pose'] = box_pose
        obs['success'] = self.success
        obs['reach_success'] = self.reach_success
        return obs