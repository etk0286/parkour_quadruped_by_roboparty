"""Configuration for BigReddog parkour with adversarial motion priors."""
import math
import os
import copy
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg, ObservationsCfg as AmpObservationsCfg
from robolab.tasks.manager_based.parkour.managers import MultiRewardCfg
import robolab.tasks.manager_based.parkour.mdp as mdp
import robolab.terrains as terrain_gen
from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.bigreddog import BIG_REDDOG_CFG, BIG_REDDOG_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import (
    ParkourEnvCfg,
    ROUGH_TERRAINS_CFG,
)
from robolab.sensors import Grid3dPointsGeneratorCfg, NoisyGroupedRayCasterCameraCfg, VolumePointsCfg
from robolab.terrains import GreedyconcatEdgeCylinderCfg, TerrainImporterCfg
from robolab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    PerlinNoiseCfg,
    PixelFailureNoiseCfg,
    RandomGaussianNoiseCfg,
    RandomConvNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    ScaleRandomizationNoiseCfg,
    StereoFusionNoiseCfg,
)
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG


AMP_NUM_STEPS = 3

# AMP-task-only reward scales. Edit this table to tune
# BigReddog-Parkour-AMP without changing the non-AMP BigReddog task.
BIG_REDDOG_AMP_PARKOUR_REWARD_WEIGHTS = {
    # Task tracking
    "track_lin_vel_xy_exp": 5.0,
    "track_ang_vel_z_exp": 5.0,
    "heading_error": -1.0,
    "dont_wait": -0.5,
    "is_alive": 3.0,
    "lin_vel_z_l2": -5.0,
    "stand_still": -4.0, #1.5
    # Contact and stability
    "feet_slide": -1.0,
    "ang_vel_xy_l2": -0.1,
    "flat_orientation_l2": -3.0,
    "sound_suppression": -5.0e-4,
    "undesired_contacts": -1.0,
    "feet_stumble": -1.0,
    # Actuation and regularization
    "dof_torques_l2": -1.0e-5,
    "dof_acc_l2": -2.5e-7,
    "dof_vel_l2": -1.0e-4,
    "joint_regularization": -1.0e-4,
    "action_rate_l2": -0.01,
    "energy": -5.0e-5,
    # Safety limits
    "dof_pos_limits": -1.0,
    "dof_vel_limits": -1.0,
    "torque_limits": -0.01,
}

# Curriculum changes these two weights during training. Edit their final values
# here; their initial values come from the table above.
BIG_REDDOG_AMP_PARKOUR_CURRICULUM_FINAL_WEIGHTS = {
    "feet_stumble": -10.0,
    "undesired_contacts": -10.0,
}

BIG_REDDOG_AMP_KEY_BODY_NAMES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]
BIG_REDDOG_MOTION_DIR = os.path.join(
    ROBOLAB_ROOT_DIR,
    "data",
    "motions",
    "bigreddog",
    "walk",
)


def _discover_motion_weights() -> dict[str, float]:
    """Give every BigReddog pickle in the motion directory equal sampling weight."""
    if not os.path.isdir(BIG_REDDOG_MOTION_DIR):
        return {}

    motion_weights = {}
    lfs_pointers = []
    for filename in sorted(os.listdir(BIG_REDDOG_MOTION_DIR)):
        if not filename.endswith(".pkl"):
            continue
        motion_path = os.path.join(BIG_REDDOG_MOTION_DIR, filename)
        with open(motion_path, "rb") as motion_file:
            if motion_file.read(42) == b"version https://git-lfs.github.com/spec/v1":
                lfs_pointers.append(filename)
                continue
        motion_weights[os.path.splitext(filename)[0]] = 1.0

    if lfs_pointers:
        raise RuntimeError(
            f"Found {len(lfs_pointers)} Git LFS pointer files instead of motion pickle data in "
            f"{BIG_REDDOG_MOTION_DIR}. Download the Git LFS objects from the motion-data repository "
            "before starting BigReddog-Parkour-AMP."
        )
    return motion_weights


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.values():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class BigReddogParkourAmpEnvCfg(ParkourEnvCfg):
    """Training configuration for BigReddog parkour with AMP."""

    def __post_init__(self):
        super().__post_init__()

        # BigReddog robot and robot-mounted sensors.
        self.scene.robot = BIG_REDDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.camera.ray_alignment = "base"
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(BIG_REDDOG_LINKS))
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.left_height_scanner = None
        self.scene.right_height_scanner = None
        self.scene.leg_volume_points = None
        self.scene.knee_volume_points = None
        self.scene.camera.offset.pos = (0.28, 0.0, 0.1214)
        self.scene.camera.offset.rot = (0.939693,0.0,0.342020,0.0)#40 degree
        self.sim.physx.gpu_collision_stack_size = 2**31

        # AMP motion data and matching discriminator history.
        self.motion_data.motion_dataset.motion_data_dir = BIG_REDDOG_MOTION_DIR
        self.motion_data.motion_dataset.motion_data_weights = _discover_motion_weights()
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.key_body_pos_b.params["asset_cfg"] = SceneEntityCfg(
            name="robot",
            body_names=BIG_REDDOG_AMP_KEY_BODY_NAMES,
            preserve_order=True,
        )

        # Disable humanoid-only sensors, rewards, events, and curricula.
        rewards = self.rewards.rewards
        rewards.rpo_thigh_yaw_joint_sign_penalty = None
        rewards.volume_points_penetration_feet = None
        rewards.volume_points_penetration_knee = None
        rewards.joint_deviation_upper_body = None
        rewards.freeze_upper_torso = None
        rewards.pelvis_orientation_l2 = None
        rewards.feet_flat_ori = None
        rewards.feet_at_plane = None

        # Apply AMP-only task reward scales.
        for term_name, weight in BIG_REDDOG_AMP_PARKOUR_REWARD_WEIGHTS.items():
            getattr(rewards, term_name).weight = weight
        for term_name, final_weight in BIG_REDDOG_AMP_PARKOUR_CURRICULUM_FINAL_WEIGHTS.items():
            curriculum_term = getattr(self.curriculum, f"{term_name}_weight")
            curriculum_term.params["init_weight"] = BIG_REDDOG_AMP_PARKOUR_REWARD_WEIGHTS[term_name]
            curriculum_term.params["final_weight"] = final_weight

        rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
        rewards.feet_slide.params["asset_cfg"].body_names = ".*_foot"
        rewards.sound_suppression.params["sensor_cfg"].body_names = ".*_foot"
        rewards.feet_stumble.params["sensor_cfg"].body_names = ".*_foot"
        rewards.undesired_contacts.params["sensor_cfg"].body_names = "(?!.*_foot).*"

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        self.terminations.root_height.params["minimum_height"] = 0.15
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["base_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["(?:FL|FR|RL|RR)_.*_link"]
        self.events.register_virtual_obstacles = None
        self.events.register_virtual_obstacles_knee = None
        self.curriculum.volume_points_penetration_weight_feet = None
        self.curriculum.volume_points_penetration_weight_knee = None

class SceneCfg(InteractiveSceneCfg):
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.03,
                min_points=2,
            ),
        },
    )
    # robots
    
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=[
            "/World/ground",
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
        ],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy
            width=64,
            height=48,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.0875,
                0.01,
                0.20568,
            ),
            rot=(
                0.866,
                0.0,
                0.5,
                0.0,
            ),
            convention="world",
        ),
        min_distance=0.1,
        # noise
        noise_pipeline={
            # --- conservative augmentations (applied on raw metric depth, before normalization) ---
            "scale_randomization": ScaleRandomizationNoiseCfg(
                apply_probability=0.5,
                scale_min=0.97,
                scale_max=1.03,
            ),
            "stereo_fusion": StereoFusionNoiseCfg(
                apply_probability=0.4,
                disparity_grad_threshold=0.10,
                texture_var_threshold=3e-4,
                hole_probability=0.02,
                hole_kernel_size=1,
                hole_value=2.5,  # treat holes as max-range (2.5 m) before normalization
            ),
            "random_conv": RandomConvNoiseCfg(
                apply_probability=0.3,
                kernel_std=0.05,
                center_weight=1.0,
            ),
            "perlin_noise": PerlinNoiseCfg(
                apply_probability=0.5,
                octaves=3,
                base_frequency=8.0,
                lacunarity=2.0,
                persistence=0.5,
                amplitude=1.0,
                noise_std=0.01,  # ~1 cm at 1 m, relative to 2.5 m range
            ),
            "pixel_failures": PixelFailureNoiseCfg(
                apply_probability=0.5,
                dead_pixel_prob=5e-4,
                saturated_pixel_prob=5e-4,
                dead_value=0.0,
                saturated_value=2.5,  # saturated = max-range before normalization
            ),
            # --- fixed preprocessing (keep last) ---
            "crop_and_resize": CropAndResizeCfg(crop_region=(30, 0, 16, 16)),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )
    
        

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        straight_target_prob=0.8, # 80% chance to force the target y to 0 for straight walking.
        ranges=mdp.PoseVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
        #random_velocity_terrain=None["mesh_boxes"],
        velocity_ranges={
            # "mesh_boxes": {"lin_vel_x": (0.2, 0.6), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "boxes": {"lin_vel_x": (0.2, 0.6), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            # "stepping_stones": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough_walk": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "perlin_rough_trun": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "square_gaps": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "training_steps": {"lin_vel_x": (0.3, 0.7), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
            "pyramid_stairs_32": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_30": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_28": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_32": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_30": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_28": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "hf_pyramid_slope_inv": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        },
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )







@configclass
class BigReddogParkourAmpEnvCfg_PLAY(BigReddogParkourAmpEnvCfg):
    """Small deterministic AMP scene used to play BigReddog checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        self.scene.num_envs = 10
        self.episode_length_s = 10.0

        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True

        terrain_name = "pyramid_stairs_inv_32"
        #perlin_rough
        # perlin_rough_walk
        # perlin_rough_trun
        # perlin_rough_stand
        # square_gaps
        # pyramid_stairs_32
        # pyramid_stairs_30
        # pyramid_stairs_28
        # pyramid_stairs_inv_32
        # pyramid_stairs_inv_30
        # pyramid_stairs_inv_28
        # hf_pyramid_slope_inv
        # mesh_boxes
        # boxes

        terrain_generator = self.scene.terrain.terrain_generator
        selected_terrain = terrain_generator.sub_terrains[terrain_name]
        selected_terrain.proportion = 1.0
        terrain_generator.sub_terrains = {terrain_name: selected_terrain}
        terrain_generator.num_rows = 1
        terrain_generator.num_cols = 1

        self.commands.base_velocity.velocity_ranges = {
            terrain_name: {
                "lin_vel_x": (0.8, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            }
        }
        self.commands.base_velocity.random_velocity_terrain = None
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
