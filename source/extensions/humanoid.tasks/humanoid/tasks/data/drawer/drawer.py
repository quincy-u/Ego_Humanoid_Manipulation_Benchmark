import os
import random
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
import omni.isaac.lab.sim as sim_utils

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

MESH_COUNT = 1

DRAWER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/drawer",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/003.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_angular_velocity=1.0,
            max_linear_velocity=1.0,
            max_depenetration_velocity=1.0,
            kinematic_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1, 0, 0),
        joint_pos={
            ".*bottom_joint": 0.0,
            ".*top_joint": 0.0,
        }
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={},
)

DRAWER_JOINT_CFG = SceneEntityCfg(
    "drawer",
    joint_names=[
        "top_joint",
        "bottom_joint",
    ],
    preserve_order=True,
)
