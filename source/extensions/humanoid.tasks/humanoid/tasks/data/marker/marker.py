import os
from omni.isaac.lab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cross": sim_utils.UsdFileCfg(
            usd_path=f"{parent_dir_path}/marker.usd",
            scale=(1.0, 1.0, 1.0),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
