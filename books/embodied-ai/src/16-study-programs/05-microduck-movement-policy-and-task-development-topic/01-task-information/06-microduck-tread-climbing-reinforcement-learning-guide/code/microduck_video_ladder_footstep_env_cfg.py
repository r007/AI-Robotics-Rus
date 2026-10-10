"""V3 physical foothold-planning task for the MicroDuck video ladder.

V2 rewarded root translation and could therefore learn to push its body into
the ladder.  V3 keeps the same discrete ladder geometry but makes foothold
support the progress variable: a rung is completed only after the alternating
foot has a real MuJoCo contact and holds it briefly.
"""

from __future__ import annotations

from copy import deepcopy
import os

import mujoco

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_video_ladder_env_cfg import (
    LADDER_RAIL_Y,
    LADDER_RUNG_HALF_WIDTH,
    LADDER_START_X,
    LADDER_START_Z,
    LADDER_END_X,
    LADDER_TOP_Z,
    TABLE_TOP_Z,
    _add_video_ladder_copy,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


RUNG_COUNT = 14
DOUBLE_SUPPORT_FOOTSTEPS = os.environ.get(
    "MICRODUCK_LADDER_DOUBLE_SUPPORT", "1"
) != "0"
# The lower timber is a real raised entry foothold.  Two alternating
# placements on it are part of the plan before rung 0; this removes the
# impossible floor-to-rung jump while keeping the final scene rung-only.
ENTRY_FOOTSTEP_COUNT = microduck_mdp.LADDER_ENTRY_FOOTSTEPS
FOOTHOLD_COUNT = ENTRY_FOOTSTEP_COUNT + RUNG_COUNT * (
    2 if DOUBLE_SUPPORT_FOOTSTEPS else 1
)
# A foothold exchange is deliberately split into three physical intervals:
# transfer the body over the support foot, swing the other foot, then settle.
LADDER_CYCLE_TIME = 1.90
LADDER_PRE_SWING_TIME = 0.45
LADDER_SWING_TIME = 1.10
STAGE_SCHEDULE = (
    (0, 0),
    (200 * 24, 1),   # A -> B: first foot on the raised entry timber
    (700 * 24, 2),   # B2: both feet share the entry timber
    (1200 * 24, 4),  # C: first physical rung with double support
    (1800 * 24, 8),  # D: continuous half-ladder climb
    (2500 * 24, FOOTHOLD_COUNT), # E: complete double-support sequence
    (3500 * 24, FOOTHOLD_COUNT), # F: top landing is now part of the success gate
    (4500 * 24, FOOTHOLD_COUNT), # G: robustness randomization after nominal success
)
DOMAIN_RANDOMIZATION_START = 4500 * 24

# Building the full 14-rung MjSpec for a one-rung curriculum stage wastes
# startup time and memory.  This only changes which discrete rungs are
# compiled into the scene; the planner and raw-contact classifier keep the
# production ``RUNG_COUNT`` contract.  Set it back to 14 for full-ladder
# training and evaluation.
COMPILE_RUNG_COUNT = max(
    1,
    min(
        int(os.environ.get("MICRODUCK_LADDER_COMPILE_RUNGS", RUNG_COUNT)),
        RUNG_COUNT,
    ),
)

# Stage A keeps the discrete ladder in the compiled scene so smoke tests and
# later resets see the same contact geometry.  The body-impact termination is
# gated on the first active rung in mdp.ladder_body_collision_termination;
# contact force, penetration, and slip costs remain active during bootstrap.


def _add_video_ladder_footstep_scene(spec: mujoco.MjSpec) -> None:
    """Add the selected discrete curriculum ladder with real collisions."""
    origins = [
        tuple(float(value) for value in site.pos)
        for site in spec.sites
        if site.name.startswith("env_origin_")
    ]
    for env_id, origin in enumerate(origins):
        _add_video_ladder_copy(
            spec,
            env_id,
            origin,
            rung_count=COMPILE_RUNG_COUNT,
            collision_mask=(3, 3),
        )


def _make_ladder_foot_contact_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="ladder_foot_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        # secondary_policy='any' is retained for aggregate force/dist fields.
        # In mjlab, 'any' removes the secondary geom filter entirely, so this
        # sensor is never used as the exact foothold-progress signal.  Target
        # rung progress and table landing are classified from MJWarp's raw
        # contact.geom/worldid table in mdp.py.
        secondary=ContactMatch(
            mode="geom", pattern=r"^video_ladder_.*_rung_[0-9]+$"
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="any",
        track_air_time=True,
    )


def _make_ladder_body_contact_cfg() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="ladder_body_contact",
        primary=ContactMatch(
            # Match only the named bit-2 body geoms.  A subtree match would
            # also include the feet, making a valid rung landing look like a
            # body impact and terminating the rollout.
            mode="geom",
            pattern=r"^(trunk_body_collision|left_leg_body_collision|right_leg_body_collision)$",
            entity="robot",
        ),
        # The exact hard-failure classification is performed from the raw
        # MJWarp contact table in ladder_body_collision_termination. The
        # generic sensor intentionally tracks all ladder contacts so its force
        # magnitude remains available for the threshold and audit terms.
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="any",
    )


def _make_ladder_trunk_contact_cfg() -> ContactSensorCfg:
    """Track only torso impacts for the hard-failure termination.

    Leg-to-rung contact is a legitimate load path during a ladder transfer.
    It remains visible through ``ladder_body_contact`` for impact and
    penetration costs, while only a trunk hit ends the approach/climb.
    """
    return ContactSensorCfg(
        name="ladder_trunk_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^trunk_body_collision$",
            entity="robot",
        ),
        # Brushing a crossbar with the torso is allowed during a real transfer
        # and remains visible through the full-body impact/penetration costs.
        # Only side rails, entry timber, and the top/table assembly are hard
        # failure contacts for the trunk termination.
        secondary=ContactMatch(
            mode="geom",
            pattern=r"^video_ladder_.*_(left_rail|right_rail|floor_foot|table_top|table_leg_.*|top_landing_.*)$",
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="any",
    )


def _set_climbing_command(cfg) -> None:
    """Keep the old command slots and retain the learned walking prior.

    The base MicroDuck policy was trained to generate a gait from a non-zero
    forward command.  Feeding it a permanent zero command turns the warm start
    into a standing controller, so the foothold terms have to discover every
    swing from noise.  The command remains only a gait prior here: the V3
    reward does not include velocity tracking and the contact-gated state
    machine still owns ladder progress.
    """
    command = deepcopy(cfg.commands["twist"])
    command.resampling_time_range = (1.0, 1.0)
    command_x = float(os.environ.get("MICRODUCK_LADDER_COMMAND_X", "0.14"))
    if command_x < 0.0:
        raise ValueError("MICRODUCK_LADDER_COMMAND_X must be non-negative")
    command.rel_standing_envs = 0.0 if command_x > 0.0 else 1.0
    command.rel_heading_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.ranges.lin_vel_x = (command_x, command_x)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(command, "heading_command"):
        command.heading_command = False
        command.ranges.heading = None
    command.debug_vis = False
    cfg.commands["twist"] = command
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6


def _disable_nominal_randomization(cfg) -> None:
    cfg.events.pop("push_robot", None)
    for event_name in (
        "base_com",
        "randomize_com",
        "randomize_head_com",
        "randomize_mass_inertia",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_armature",
        "randomize_motor_gains",
        "randomize_base_orientation",
        "encoder_bias",
    ):
        cfg.events.pop(event_name, None)
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)


def _add_foothold_observations(cfg, foot_sensor_name: str) -> None:
    shared = {
        "rung_count": RUNG_COUNT,
        "stage_schedule": STAGE_SCHEDULE,
        "start_x": LADDER_START_X,
        "start_z": LADDER_START_Z,
        "finish_x": LADDER_END_X,
        "finish_z": LADDER_TOP_Z,
        "sensor_name": foot_sensor_name,
    }
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["ladder_step_targets"] = ObservationTermCfg(
            func=microduck_mdp.ladder_step_targets,
            params=shared | {"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
            scale=1.0,
        )
        terms["ladder_phase"] = ObservationTermCfg(
            func=microduck_mdp.ladder_phase_observation,
            params=shared | {"cycle_time": LADDER_CYCLE_TIME},
            scale=1.0,
        )
        terms["ladder_swing_foot"] = ObservationTermCfg(
            func=microduck_mdp.ladder_swing_foot_observation,
            params=shared,
            scale=1.0,
        )
        terms["ladder_foot_positions"] = ObservationTermCfg(
            func=microduck_mdp.ladder_foot_positions_observation,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
            scale=1.0,
        )
        terms["ladder_foot_contacts"] = ObservationTermCfg(
            func=microduck_mdp.ladder_foot_contacts_observation,
            params={"sensor_name": foot_sensor_name, "rung_count": RUNG_COUNT},
            scale=1.0,
        )
        terms["ladder_stage"] = ObservationTermCfg(
            func=microduck_mdp.ladder_stage_observation,
            params=shared,
            scale=1.0,
        )
        # Append new values after the complete legacy observation contract.
        # This keeps the first 218/233 columns compatible with old V3
        # checkpoints when the warm-start expansion pads the first layer.
        terms["ladder_support_state"] = ObservationTermCfg(
            func=microduck_mdp.ladder_support_state_observation,
            params=shared | {
                "asset_cfg": SceneEntityCfg(
                    "robot", site_names=("left_foot", "right_foot")
                ),
            },
            scale=1.0,
        )


def make_microduck_video_ladder_footstep_env_cfg(play: bool = False):
    """Create the V3 planned-foothold ladder environment."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.env_spacing = 4.5
    cfg.scene.spec_fn = _add_video_ladder_footstep_scene
    cfg.scene.extent = 3.4

    # Optional train-time residual teacher.  It is deliberately opt-in so
    # strict evaluation and tutorial smoke tests remain policy-only.  The
    # state file is a solved MuJoCo pose for the currently planned rung; the
    # action term interpolates toward its 14 actuated joint targets while PPO
    # learns the residual needed for the real contact dynamics.
    teacher_state = os.environ.get("MICRODUCK_LADDER_TEACHER_STATE")
    if teacher_state:
        base_action = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = (
            microduck_mdp.LadderTeacherJointPositionActionCfg(
                entity_name=base_action.entity_name,
                actuator_names=base_action.actuator_names,
                scale=base_action.scale,
                offset=base_action.offset,
                clip=base_action.clip,
                preserve_order=base_action.preserve_order,
                use_default_offset=base_action.use_default_offset,
                teacher_state_path=teacher_state,
                teacher_target_rung=int(
                    os.environ.get("MICRODUCK_LADDER_TEACHER_TARGET_RUNG", "2")
                ),
                teacher_blend=float(
                    os.environ.get("MICRODUCK_LADDER_TEACHER_BLEND", "0.85")
                ),
                teacher_duration=float(
                    os.environ.get("MICRODUCK_LADDER_TEACHER_DURATION", "1.20")
                ),
                teacher_all_joints=os.environ.get(
                    "MICRODUCK_LADDER_TEACHER_ALL_JOINTS", "0"
                )
                == "1",
            )
        )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        _make_ladder_foot_contact_cfg(),
        _make_ladder_body_contact_cfg(),
        _make_ladder_trunk_contact_cfg(),
        RayCastSensorCfg(
            name="ladder_scan",
            frame=ObjRef(type="body", name="trunk_base", entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.4, 0.8), resolution=0.1),
            max_distance=1.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            debug_vis=bool(play),
        ),
    )

    cfg.episode_length_s = 20.0
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 2.4
    cfg.viewer.elevation = -25.0
    cfg.viewer.azimuth = 78.0
    cfg.viewer.max_extra_envs = 0

    cfg.events["reset_base"].params["pose_range"] = {
        # Keep the duck in front of the lower timber while putting rung 0
        # inside one reachable step.  This is a curriculum reset, not a
        # success shortcut: the feet remain on the ground and the planner
        # still requires a real rung contact plus a hold interval.
        "x": (0.12, 0.14),
        "y": (-0.020, 0.020),
        "z": (0.12, 0.13),
        "yaw": (-0.04, 0.04),
    }
    _set_climbing_command(cfg)
    _disable_nominal_randomization(cfg)

    cfg.events["ladder_domain_randomization"] = EventTermCfg(
        func=microduck_mdp.ladder_domain_randomization,
        mode="reset",
        params={
            "start_step": DOMAIN_RANDOMIZATION_START,
            "foot_asset_cfg": SceneEntityCfg(
                "robot", geom_names=("left_foot_collision", "right_foot_collision")
            ),
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "mass_asset_cfg": SceneEntityCfg(
                "robot", body_names=("trunk_base",)
            ),
        },
    )

    bootstrap_state = os.environ.get("MICRODUCK_LADDER_BOOTSTRAP_STATE")
    if bootstrap_state:
        if not teacher_state:
            base_action = cfg.actions["joint_pos"]
            cfg.actions["joint_pos"] = (
                microduck_mdp.LadderBootstrapJointPositionActionCfg(
                    entity_name=base_action.entity_name,
                    actuator_names=base_action.actuator_names,
                    scale=base_action.scale,
                    offset=base_action.offset,
                    clip=base_action.clip,
                    preserve_order=base_action.preserve_order,
                    use_default_offset=base_action.use_default_offset,
                )
            )
        cfg.events["ladder_bootstrap_reset"] = EventTermCfg(
            func=microduck_mdp.ladder_bootstrap_reset,
            mode="reset",
            params={
                "state_path": bootstrap_state,
                # The bridge file may already contain two validated support
                # feet.  Start from the next uncompleted rung instead of
                # asking the policy to rediscover the support pose.
                "target_rung": int(
                    os.environ.get("MICRODUCK_LADDER_BOOTSTRAP_TARGET_RUNG", "1")
                ),
                "velocity_scale": 0.0,
                "joint_velocity_scale": float(
                    os.environ.get(
                        "MICRODUCK_LADDER_BOOTSTRAP_JOINT_VELOCITY_SCALE",
                        "0.5",
                    )
                ),
            },
        )

    # Contact-rich ladder scenes need more contact slots and solver work than
    # the flat walking default.  These settings are still modest for vectorized
    # MuJoCo and avoid silently dropping simultaneous body/rung contacts.
    cfg.sim.nconmax = 256
    cfg.sim.mujoco.iterations = 40
    cfg.sim.mujoco.ls_iterations = 60

    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "ladder_scan"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        scale=1.0,
    )
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "ladder_scan"},
        scale=1.0,
    )
    _add_foothold_observations(cfg, "ladder_foot_contact")

    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["upright"].weight = 3.0
    # The flat-walking pose prior is useful for approach, but it otherwise
    # dominates the bridge task and teaches the policy to keep both feet on
    # the floor.  Footstep-specific terms remain responsible for posture.
    cfg.rewards["pose"].weight = 0.25
    cfg.rewards["air_time"].weight = 0.25
    cfg.rewards["foot_clearance"].weight = -0.25
    cfg.rewards["foot_slip"].weight = -0.05
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.rewards["body_pose_tracking"].weight = 0.0

    common = {
        "rung_count": RUNG_COUNT,
        "stage_schedule": STAGE_SCHEDULE,
        "start_x": LADDER_START_X,
        "start_z": LADDER_START_Z,
        "finish_x": LADDER_END_X,
        "finish_z": LADDER_TOP_Z,
        "sensor_name": "ladder_foot_contact",
    }
    feet_common = common | {
        "asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_foot", "right_foot")
        )
    }
    cfg.rewards["ladder_target_foot"] = RewardTermCfg(
        func=microduck_mdp.ladder_target_foot_reward,
        weight=8.0,
        params=feet_common | {"distance_scale": 0.22},
    )
    cfg.rewards["ladder_target_contact"] = RewardTermCfg(
        func=microduck_mdp.ladder_target_contact_reward,
        weight=6.0,
        params=feet_common,
    )
    cfg.rewards["ladder_entry_forward_velocity"] = RewardTermCfg(
        func=microduck_mdp.ladder_entry_forward_velocity_reward,
        # The entry is the only phase where a positive approach signal is
        # useful.  It is disabled automatically after both entry targets are
        # complete, so later rung progress cannot be farmed by pushing ahead.
        weight=4.0,
        params=common | {"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["ladder_entry_position_progress"] = RewardTermCfg(
        func=microduck_mdp.ladder_entry_position_progress_reward,
        # Public step-up training uses a persistent approach command.  This
        # bounded potential gives the same entry gradient without restoring
        # root-only progress after the first raised timber.
        weight=24.0,
        params=common | {"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["ladder_foothold_contact_milestone"] = RewardTermCfg(
        func=microduck_mdp.ladder_foothold_contact_milestone,
        # One-shot real-contact credit helps PPO discover the 25 mm entry
        # timber while the hold/pose gate remains the only progress source.
        weight=100.0,
        params=feet_common | {"target_tolerance": 0.060},
    )
    cfg.rewards["ladder_landing_stability"] = RewardTermCfg(
        func=microduck_mdp.ladder_landing_stability_reward,
        # A contact tap is not progress.  This term pays for the physically
        # meaningful next step: the trunk rises above the contacted rung and
        # settles before the hard foothold gate can advance.
        weight=10.0,
        params=feet_common,
    )
    cfg.rewards["ladder_post_landing_transfer"] = RewardTermCfg(
        func=microduck_mdp.ladder_post_landing_transfer_reward,
        # Contact-gated potential for loading the newly landed foot.  It does
        # not modify the foothold state machine or award root-only progress.
        weight=18.0,
        params=feet_common,
    )
    cfg.rewards["ladder_body_alignment"] = RewardTermCfg(
        func=microduck_mdp.ladder_body_alignment_reward,
        # The term is a per-step potential difference, so it needs a larger
        # coefficient than the bounded absolute foothold score to provide a
        # visible directional signal during the approach phase.
        weight=10.0,
        params=common | {
            "asset_cfg": SceneEntityCfg("robot"),
            "distance_scale": 0.30,
        },
    )
    cfg.rewards["ladder_heading_alignment"] = RewardTermCfg(
        func=microduck_mdp.ladder_heading_alignment_reward,
        # Prevent a one-rung support state from drifting sideways before the
        # next swing.  This is a heading potential; it cannot advance the
        # contact-gated foothold state machine.
        weight=5.0,
        params=common | {"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["ladder_support_transfer"] = RewardTermCfg(
        func=microduck_mdp.ladder_support_transfer_reward,
        # Move the COM over the foot already carrying the ladder load before
        # the other foot is asked to leave the floor.
        weight=12.0,
        params=common | {
            "asset_cfg": feet_common["asset_cfg"],
            "ground_sensor_name": "feet_ground_contact",
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
            "lateral_scale": 0.045,
        },
    )
    cfg.rewards["ladder_support_pose"] = RewardTermCfg(
        func=microduck_mdp.ladder_support_pose_reward,
        # The learned walking prior already stabilizes the stance leg.  A
        # reset-pose lock here prevents the pelvis from rising and extending
        # the support leg for the next rung, which is exactly the transfer
        # needed after the first landing.  It is opt-in because later ladder
        # stages should be free to change support-leg posture, while the
        # dedicated transfer stage benefits from a small stabilizing prior.
        weight=float(os.environ.get("MICRODUCK_LADDER_SUPPORT_POSE_WEIGHT", "0.0")),
        params=common | {
            "asset_cfg": feet_common["asset_cfg"],
            "ground_sensor_name": "feet_ground_contact",
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
        },
    )
    teacher_state = os.environ.get("MICRODUCK_LADDER_TEACHER_STATE", "")
    cfg.rewards["ladder_joint_teacher"] = RewardTermCfg(
        func=microduck_mdp.ladder_joint_teacher_reward,
        # Disabled for the production V3 task.  Stage-B experiments can opt
        # in with a MuJoCo-solved joint pose while keeping contact-gated
        # progress and the collision audit unchanged.
        weight=float(os.environ.get("MICRODUCK_LADDER_TEACHER_WEIGHT", "0.0")),
        params=common | {
            "state_path": teacher_state,
            "asset_cfg": feet_common["asset_cfg"],
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
            "swing_time": LADDER_SWING_TIME,
        },
    )
    cfg.rewards["ladder_root_teacher"] = RewardTermCfg(
        func=microduck_mdp.ladder_root_teacher_reward,
        # Joint imitation alone can leave the free base behind the support
        # foot.  This optional whole-body bridge term keeps the torso over
        # the support foot while the swing-leg teacher is active; contact and
        # hold gates still decide whether a foothold is actually completed.
        weight=float(os.environ.get("MICRODUCK_LADDER_ROOT_TEACHER_WEIGHT", "0.0")),
        params=common | {
            "state_path": teacher_state,
            "asset_cfg": feet_common["asset_cfg"],
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
            "swing_time": LADDER_SWING_TIME,
            "position_scale": 0.075,
            "orientation_scale": 0.30,
        },
    )
    cfg.rewards["ladder_step_advance"] = RewardTermCfg(
        func=microduck_mdp.ladder_step_advance_reward,
        weight=40.0,
        params=common,
    )
    cfg.rewards["ladder_support_stability"] = RewardTermCfg(
        func=microduck_mdp.ladder_support_stability_reward,
        weight=2.0,
        params=common | {
            "asset_cfg": feet_common["asset_cfg"],
        },
    )
    cfg.rewards["ladder_swing_clearance"] = RewardTermCfg(
        func=microduck_mdp.ladder_swing_clearance_reward,
        weight=10.0,
        params=feet_common | {
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
            "swing_time": LADDER_SWING_TIME,
        },
    )
    cfg.rewards["ladder_footstep_trajectory"] = RewardTermCfg(
        func=microduck_mdp.ladder_footstep_trajectory_reward,
        # A phase-conditioned swing target supplies exploration guidance while
        # the contact/hold state machine remains the only progress gate.
        weight=24.0,
        params=common | {
            "asset_cfg": feet_common["asset_cfg"],
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
            "swing_time": LADDER_SWING_TIME,
            "lift_height": 0.06,
            "distance_scale": 0.09,
        },
    )
    cfg.rewards["ladder_step_up"] = RewardTermCfg(
        func=microduck_mdp.ladder_step_up_reward,
        # The signal is only for the physical first transition; contact and
        # foothold advancement remain governed by the real rung sensor.
        weight=16.0,
        params=common | {
            "asset_cfg": feet_common["asset_cfg"],
            "floor_foot_top_z": 0.025,
        },
    )
    cfg.rewards["ladder_phase_contact_velocity"] = RewardTermCfg(
        func=microduck_mdp.ladder_phase_contact_velocity_reward,
        # Ported from the planned-footstep idea: phase-conditioned support,
        # swing-foot motion, and a short double-support settling interval.
        weight=1.2,
        params=common | {
            "ground_sensor_name": "feet_ground_contact",
            "asset_cfg": feet_common["asset_cfg"],
            "cycle_time": LADDER_CYCLE_TIME,
            "pre_swing_time": LADDER_PRE_SWING_TIME,
            "swing_time": LADDER_SWING_TIME,
        },
    )
    cfg.rewards["ladder_contact_slip"] = RewardTermCfg(
        func=microduck_mdp.ladder_foot_slip_cost,
        weight=-0.35,
        params={
            "sensor_name": "ladder_foot_contact",
            "rung_count": RUNG_COUNT,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
        },
    )
    cfg.rewards["ladder_body_impact"] = RewardTermCfg(
        func=microduck_mdp.ladder_body_impact_cost,
        weight=-0.8,
        params={"sensor_name": "ladder_body_contact", "threshold": 1.5},
    )
    cfg.rewards["ladder_penetration"] = RewardTermCfg(
        func=microduck_mdp.ladder_penetration_cost,
        weight=-25.0,
        params={"sensor_names": ("ladder_foot_contact", "ladder_body_contact")},
    )
    cfg.rewards["ladder_top_stability"] = RewardTermCfg(
        func=microduck_mdp.ladder_top_stability_reward,
        weight=4.0,
        params={
            "rung_count": RUNG_COUNT,
            "stage_schedule": STAGE_SCHEDULE,
            "sensor_name": "ladder_foot_contact",
            "table_x": 1.04,
            "table_top_z": TABLE_TOP_Z,
            "asset_cfg": feet_common["asset_cfg"],
        },
    )

    cfg.terminations["ladder_body_collision"] = TerminationTermCfg(
        func=microduck_mdp.ladder_body_collision_termination,
        time_out=False,
        params={
            "sensor_name": "ladder_trunk_contact",
            # The one-rung bootstrap doubles this threshold only during the
            # approach-to-first-rung discovery window.  MuJoCo collision,
            # impact cost, and penetration cost stay active throughout; after
            # the first foothold the normal high-impact failure gate applies.
            "force_threshold": 8.0,
            "top_clearance_x": LADDER_END_X + 0.10,
            "rung_count": RUNG_COUNT,
            "stage_schedule": STAGE_SCHEDULE,
        },
    )
    cfg.terminations["ladder_foothold_deadline"] = TerminationTermCfg(
        func=microduck_mdp.ladder_foothold_deadline,
        time_out=False,
        params=common | {
            "first_target_deadline": 4.0,
            # The bridge starts from an already-contacting support foot; give
            # the alternating swing enough time to clear the timber and settle.
            "next_target_deadline": 4.0,
        },
    )
    cfg.terminations["ladder_footstep_success"] = TerminationTermCfg(
        func=microduck_mdp.ladder_footstep_success,
        time_out=False,
        params={
            "rung_count": RUNG_COUNT,
            "stage_schedule": STAGE_SCHEDULE,
            "sensor_name": "ladder_foot_contact",
            "table_x": 1.04,
            "table_top_z": TABLE_TOP_Z,
            "finish_x": 0.92,
            "max_tilt_deg": 35.0,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
        },
    )

    cfg.curriculum = {
        "ladder_stage": CurriculumTermCfg(
            func=microduck_mdp.ladder_stage_curriculum,
            params={
                "rung_count": RUNG_COUNT,
                "stage_schedule": STAGE_SCHEDULE,
            },
        )
    }
    return cfg


MicroduckVideoLadderFootstepRlCfg = deepcopy(MicroduckRlCfg)
MicroduckVideoLadderFootstepRlCfg.experiment_name = "ladder_climb_footstep"
MicroduckVideoLadderFootstepRlCfg.run_name = "ladder-footstep-v3"
MicroduckVideoLadderFootstepRlCfg.save_interval = 100
MicroduckVideoLadderFootstepRlCfg.max_iterations = 6_000
# The inherited mirror callback is hard-coded for the old 61D velocity layout.
# New target/phase/contact terms are intentionally not mirrored until a
# task-specific permutation is defined.
MicroduckVideoLadderFootstepRlCfg.algorithm.symmetry_cfg = None
