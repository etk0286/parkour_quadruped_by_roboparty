"""Configuration for BigReddog parkour without AMP motion data."""

import copy

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from robolab.assets.robots.bigreddog import BIG_REDDOG_CFG, BIG_REDDOG_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    ROUGH_TERRAINS_CFG,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


# BigReddog-only task reward scales. Both the PPO and AMP parkour tasks use
# this table, so tuning here does not change the humanoid RPO task.
BIG_REDDOG_PARKOUR_REWARD_WEIGHTS = {
    # Task tracking
    "track_lin_vel_xy_exp": 5.0,
    "track_ang_vel_z_exp": 5.0,
    "heading_error": -1.0,
    "dont_wait": -0.5,
    "is_alive": 3.0,
    "lin_vel_z_l2": -5.0,
    "stand_still": -1.3,
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

# These two reward weights are changed dynamically by the curriculum.
BIG_REDDOG_PARKOUR_CURRICULUM_FINAL_WEIGHTS = {
    "feet_stumble": -10.0,
    "undesired_contacts": -10.0,
}


def apply_bigreddog_parkour_reward_weights(cfg) -> None:
    """Apply all BigReddog task and reward-curriculum scales."""
    rewards = cfg.rewards.rewards
    for term_name, weight in BIG_REDDOG_PARKOUR_REWARD_WEIGHTS.items():
        getattr(rewards, term_name).weight = weight

    for term_name, final_weight in BIG_REDDOG_PARKOUR_CURRICULUM_FINAL_WEIGHTS.items():
        curriculum_term = getattr(cfg.curriculum, f"{term_name}_weight")
        curriculum_term.params["init_weight"] = BIG_REDDOG_PARKOUR_REWARD_WEIGHTS[term_name]
        curriculum_term.params["final_weight"] = final_weight


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.values():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class BigReddogParkourEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration specialized for BigReddog's four legs."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Control at 50 Hz while simulating physics at 200 Hz.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**29
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Robot and robot-mounted terrain sensors.
        self.scene.robot = BIG_REDDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(BIG_REDDOG_LINKS))
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # These sensors describe two humanoid shoes/knees. They are not used by
        # the initial quadruped reward set.
        self.scene.left_height_scanner = None
        self.scene.right_height_scanner = None
        self.scene.leg_volume_points = None
        self.scene.knee_volume_points = None

        self.scene.camera.offset.pos = (0.28, 0.0, 0.12)
        self.scene.camera.offset.rot =(0.966, 0.0, 0.259, 0.0)
        # A normal PPO policy only consumes policy and critic observations.
        self.observations.disc = None
        self.observations.disc_demo = None

        rewards = self.rewards.rewards

        # Humanoid-only rewards.
        rewards.rpo_thigh_yaw_joint_sign_penalty = None
        rewards.volume_points_penetration_feet = None
        rewards.volume_points_penetration_knee = None
        rewards.joint_deviation_upper_body = None
        rewards.freeze_upper_torso = None
        rewards.pelvis_orientation_l2 = None
        rewards.feet_flat_ori = None
        rewards.feet_at_plane = None

        # Apply the complete BigReddog-only reward scale table above.
        apply_bigreddog_parkour_reward_weights(self)

        # Generic contact rewards, redirected from humanoid ankles to all four
        # fixed foot links.
        rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
        rewards.feet_slide.params["asset_cfg"].body_names = ".*_foot"
        rewards.sound_suppression.params["sensor_cfg"].body_names = ".*_foot"
        rewards.feet_stumble.params["sensor_cfg"].body_names = ".*_foot"
        rewards.undesired_contacts.params["sensor_cfg"].body_names = "(?!.*_foot).*"

        # Terminate when the trunk hits terrain or falls below a quadruped-sized
        # minimum height.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        self.terminations.root_height.params["minimum_height"] = 0.15

        # BigReddog uses base_link and FL/FR/RL/RR link naming instead of the
        # humanoid torso/left/right naming.
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["base_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = [
            "(?:FL|FR|RL|RR)_.*_link"
        ]
        self.events.register_virtual_obstacles = None
        self.events.register_virtual_obstacles_knee = None

        # Curricula tied to the removed humanoid volume-point rewards.
        self.curriculum.volume_points_penetration_weight_feet = None
        self.curriculum.volume_points_penetration_weight_knee = None


@configclass
class BigReddogParkourEnvCfg_PLAY(BigReddogParkourEnvCfg):
    """Small deterministic scene used by the play script."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.debug_vis = True

        if self.scene.terrain.terrain_generator is not None:
            terrain_name = "square_gaps"
            terrain_generator = self.scene.terrain.terrain_generator
            selected_terrain = terrain_generator.sub_terrains[terrain_name]
            selected_terrain.proportion = 1.0

            terrain_generator.sub_terrains = {
                terrain_name: selected_terrain,
            }

            self.commands.base_velocity.velocity_ranges = {
                "square_gaps": {
                    "lin_vel_x": (0.4, 0.8), 
                    "lin_vel_y": (0.0, 0.0), 
                    "ang_vel_z": (-1.0, 1.0) 
                },
            }
            self.commands.base_velocity.random_velocity_terrain = None

            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
