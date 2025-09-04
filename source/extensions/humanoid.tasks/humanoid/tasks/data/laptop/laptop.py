import os
import random
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

MESH_COUNT = 1

LAPTOP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/laptop",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/notebook.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=10.0,
            max_linear_velocity=50.0,
            max_depenetration_velocity=50.0,
            kinematic_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1, 0, 0),
        joint_pos={
            ".*joint": 0.088
        }
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={},
)

def randomize_mesh():
    mesh_idx = random.randint(1, MESH_COUNT)
    file_name = (3 - len(str(mesh_idx))) * "0" + str(mesh_idx)
    return sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/{file_name}.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=10.0,
            max_linear_velocity=50.0,
            max_depenetration_velocity=50.0,
            kinematic_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
