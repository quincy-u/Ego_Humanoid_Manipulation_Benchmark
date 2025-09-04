import os
from omni.isaac.lab.managers.scene_entity_cfg import SceneEntityCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(current_file_path)

stiffness = 2000
damping = 75

H1_INSPIRE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{parent_dir_path}/h1_inspire_convex_decomp.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=100.0,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=10,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=4
        ),
        semantic_tags=[('focus', 'true'), ('category', 'robot')],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            # "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.5,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_pitch_joint": -1,
            ".*_elbow_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
            ".*_index_proximal_joint": 0.0,
            ".*_ring_proximal_joint": 0.0,
            ".*_index_intermediate_joint": 0.0,
            ".*_middle_intermediate_joint": 0.0,
            ".*_pinky_intermediate_joint": 0.0,
            ".*_ring_intermediate_joint": 0.0,
            ".*_thumb_intermediate_joint": 0.0,
            ".*_thumb_proximal_yaw_joint": 0.0,
            ".*_thumb_proximal_pitch_joint": 0.0,
            ".*_thumb_distal_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint", ".*_elbow_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit=100,
            velocity_limit=1.0,
            stiffness={
                ".*_shoulder_pitch_joint": stiffness,
                ".*_shoulder_roll_joint": stiffness,
                ".*_shoulder_yaw_joint": stiffness,
                ".*_elbow_pitch_joint": stiffness,
                ".*_elbow_roll_joint": stiffness,
                ".*_wrist_pitch_joint": stiffness,
                ".*_wrist_yaw_joint": stiffness,
            },
            damping={
                ".*_shoulder_pitch_joint": damping,
                ".*_shoulder_roll_joint": damping,
                ".*_shoulder_yaw_joint": damping,
                ".*_elbow_pitch_joint": damping,
                ".*_elbow_roll_joint": damping,
                ".*_wrist_pitch_joint": damping,
                ".*_wrist_yaw_joint": damping,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_index_proximal_joint", ".*_middle_proximal_joint", ".*_pinky_proximal_joint",
                              ".*_ring_proximal_joint", ".*_index_intermediate_joint", ".*_middle_intermediate_joint",
                              ".*_pinky_intermediate_joint", ".*_ring_intermediate_joint", ".*_thumb_intermediate_joint",
                              ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint"  , ".*_thumb_distal_joint"],
            effort_limit=10,
            velocity_limit=1.0,
            stiffness={
                ".*_index_proximal_joint": stiffness,
                ".*_middle_proximal_joint": stiffness,
                ".*_pinky_proximal_joint": stiffness,
                ".*_ring_proximal_joint": stiffness,
                ".*_index_intermediate_joint": stiffness,
                ".*_middle_intermediate_joint": stiffness,
                ".*_pinky_intermediate_joint": stiffness,
                ".*_ring_intermediate_joint": stiffness,
                ".*_thumb_intermediate_joint": stiffness,
                ".*_thumb_proximal_yaw_joint": stiffness,
                ".*_thumb_proximal_pitch_joint": stiffness,
                ".*_thumb_distal_joint": stiffness,
            },
            damping={
                ".*_index_proximal_joint": damping,
                ".*_middle_proximal_joint": damping,
                ".*_pinky_proximal_joint": damping,
                ".*_ring_proximal_joint": damping,
                ".*_index_intermediate_joint": damping,
                ".*_middle_intermediate_joint": damping,
                ".*_pinky_intermediate_joint": damping,
                ".*_ring_intermediate_joint": damping,
                ".*_thumb_intermediate_joint": damping,
                ".*_thumb_proximal_yaw_joint": damping,
                ".*_thumb_proximal_pitch_joint": damping,
                ".*_thumb_distal_joint": damping,
            }
        ),
    }
)

H1_INSPIRE_LEFT_ARM_CFG: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    body_names=["L_hand_base_link"],
    preserve_order=True
)
    
H1_INSPIRE_RIGHT_ARM_CFG: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=[
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    body_names=["R_hand_base_link"],
    preserve_order=True
)

H1_INSPIRE_LEFT_HAND_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint",
        "L_thumb_distal_joint"
    ],
    body_names=[
        "L_thumb_tip",
        "L_index_tip",
        "L_middle_tip",
        "L_ring_tip",
        "L_pinky_tip",
    ],
    preserve_order=True,
)

H1_INSPIRE_RIGHT_HAND_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint"
    ],
    body_names=[
        "R_thumb_tip",
        "R_index_tip",
        "R_middle_tip",
        "R_ring_tip",
        "R_pinky_tip",
    ],
    preserve_order=True,
)
"""Configuration for the Unitree H1 Inspire Humanoid robot with hand."""
