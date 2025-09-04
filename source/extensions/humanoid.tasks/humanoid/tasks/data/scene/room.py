import os
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
import omni.isaac.lab.sim as sim_utils

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

ROOM_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/room",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/001.usd",
        activate_contact_sensors=False,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)),
)