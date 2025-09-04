from __future__ import annotations
from collections.abc import Sequence

from humanoid.tasks.base_env.base_env import BaseEnv, BaseEnvCfg
from humanoid.tasks.data.bowl import BOWL_CFG
from humanoid.tasks.data.glassware import GLASSWARE_CFG
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab.utils.math as math_util 
import omni.isaac.lab.sim as sim_utils
import torch

BALL_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.006,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.0, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=0.5,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.35, 1.1)),
) 

@configclass
class PourBallsEnvCfg(BaseEnvCfg):
    # bottle that initially contains balls
    bottle: RigidObjectCfg = GLASSWARE_CFG.replace(prim_path="/World/envs/env_.*/bottle") # type: ignore
    bottle.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.3, 1.025), rot=(1, 0, 0, 0))
    # bowl that balls are poured into
    bowl: RigidObjectCfg = BOWL_CFG.replace(prim_path="/World/envs/env_.*/bowl")  # type: ignore
    bowl.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.54, 0.0, 1.025), rot=(1, 0, 0, 0))
    bowl.spawn.scale = (0.6, 0.6, 0.6) # type: ignore
    # balls
    ball1: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball1")  # type: ignore
    ball1.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.31, 1.1))
    ball2: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball2")  # type: ignore
    ball2.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.29, 1.1))
    ball3: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball3")  # type: ignore
    ball3.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.31, 1.1))
    ball4: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball4")  # type: ignore
    ball4.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.29, 1.1))
    ball5: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball5")  # type: ignore
    ball5.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.31, 1.07))
    ball6: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball6")  # type: ignore
    ball6.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.46, -0.29, 1.07))
    ball7: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball7")  # type: ignore
    ball7.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.31, 1.07))
    ball8: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/ball8")  # type: ignore
    ball8.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.44, -0.29, 1.07))

class PourBallsEnv(BaseEnv):
    cfg: PourBallsEnvCfg 

    def __init__(self, cfg: PourBallsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.bowl_surface_center_bowlframe = torch.tensor([0.0, 0.0, 0.075], device=self.device)
        self.bowl_surface_radius = 0.085
        self.reach_bottle_success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.lift_success = torch.zeros([self.num_envs], dtype=torch.bool, device=self.device)
        self.ball_pass_surface = torch.zeros([self.num_envs, 8], dtype=torch.bool, device=self.device)
        self.poured_balls_count = torch.zeros([self.num_envs], dtype=torch.int, device=self.device)
            
    def _setup_scene(self):
        self.ball1 = RigidObject(self.cfg.ball1)
        self.ball2 = RigidObject(self.cfg.ball2)
        self.ball3 = RigidObject(self.cfg.ball3)
        self.ball4 = RigidObject(self.cfg.ball4)
        self.ball5 = RigidObject(self.cfg.ball5)
        self.ball6 = RigidObject(self.cfg.ball6)
        self.ball7 = RigidObject(self.cfg.ball7)
        self.ball8 = RigidObject(self.cfg.ball8)
        self.balls = [self.ball1, self.ball2, self.ball3, self.ball4, self.ball5, self.ball6, self.ball7, self.ball8]
        self.bottle = RigidObject(self.cfg.bottle)
        self.bowl = RigidObject(self.cfg.bowl)
        super()._setup_scene()
    
    def _register_scene(self):
        super()._register_scene()
        for i, ball in enumerate(self.balls):
            self.scene.rigid_objects[f"ball{i+1}"] = ball
        self.scene.rigid_objects['bottle'] = self.bottle
        self.scene.rigid_objects['bowl'] = self.bowl

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        for ball in self.balls:
            ball_root_state = ball.data.default_root_state[env_ids, :]
            ball_root_state[env_ids, 0:3] += self.scene.env_origins[env_ids, 0:3]
            ball.write_root_state_to_sim(ball_root_state, env_ids=env_ids)
        bottle_root_state = self.bottle.data.default_root_state[env_ids, :]
        bottle_root_state[env_ids, 0:3] += self.scene.env_origins[env_ids, 0:3]
        self.bottle.write_root_state_to_sim(bottle_root_state, env_ids=env_ids)
        bowl_root_state = self.bowl.data.default_root_state[env_ids, :]
        if self.cfg.randomize:
            if self.cfg.randomize_idx < 0:
                bowl_root_state[:, 0:2] += (self.cfg.randomize_range * 0.2) * torch.rand((self.num_envs, 2), device=self.device) - (0.1 * self.cfg.randomize_range) 
            else:
                column_idx = self.cfg.randomize_idx // 100
                row_idx = self.cfg.randomize_idx % 100
                bowl_root_state[:, 1] += 0.1 - (0.2 / 99) * row_idx
                bowl_root_state[:, 0] += 0.1 - (0.2 / 99) * column_idx
        bowl_root_state[env_ids, 0:3] += self.scene.env_origins[env_ids, 0:3]
        self.bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
        self.reach_bottle_success[env_ids] = 0
        self.lift_success[env_ids] = 0
        self.ball_pass_surface[env_ids, :] = 0
        self.poured_balls_count[env_ids] = 0
        
    def _get_subtask(self):
        right_hand_bottle_dist = torch.norm(self.right_finger_pos_mean - self.bottle.data.root_state_w[:, 0:3], dim=-1)
        self.reach_bottle_success[torch.where(right_hand_bottle_dist < 0.12)] = 1
        self.lift_success[torch.where(self.bottle.data.root_state_w[:, 2] > 1.085)] = 1
        
    def _get_success(self):
        rot_matrix = math_util.matrix_from_quat(self.bowl.data.root_state_w[:, 3:7])
        bowl_surface_center = self.bowl.data.root_state_w[:, 0:3] + torch.matmul(rot_matrix, self.bowl_surface_center_bowlframe)
        bowl_surface_normal = bowl_surface_center - self.bowl.data.root_state_w[:, 0:3]
        bowl_surface_normal /= torch.norm(bowl_surface_normal, dim=-1, keepdim=True)
        self.success[:] = 0
        # the normal of the bowl surface is calculated first, then we calcualte the line passing through
        # the ball and the center of the bowl surface, and check if the angle between the line and the normal
        # if the angle is 90 degrees and distance between the ball and surface center is less than radius,
        # then the ball is passing through the surface, and we count it as poured into the bowl
        for env_id in range(self.num_envs):
            for ball_idx, ball in enumerate(self.balls):
                ball_pos = ball.data.root_state_w[env_id, 0:3] - self.scene.env_origins[env_id, 0:3]
                ball_bowl_surface_center_vector = ball_pos - bowl_surface_center[env_id]
                ball_bowl_surface_center_vector /= torch.norm(ball_bowl_surface_center_vector)
                cos_theta = torch.dot(ball_bowl_surface_center_vector, bowl_surface_normal[env_id])
                angle = torch.acos(cos_theta) * 180 / torch.pi
                
                ball_surface_center_dist = torch.norm((bowl_surface_center - ball_pos), dim=-1)
                if (not self.ball_pass_surface[env_id, ball_idx]) and ball_surface_center_dist < self.bowl_surface_radius and 80 < angle and angle < 100:
                    self.ball_pass_surface[env_id, ball_idx] = 1
                    self.poured_balls_count[env_id] += 1
            self.success[env_id] = 1 if self.poured_balls_count >= 3 else 0

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._get_success()
        self._get_subtask()
        obs['success'] = self.success
        obs['reach_bottle_success'] = self.reach_bottle_success
        obs['lift_success'] = self.lift_success
        return obs