"""Video-aligned MicroDuck ladder and table-top climbing task.

The reference clip shows a steep, narrow wooden ladder with many close rungs,
a floor foot block, and a high table at the top.  This version keeps the
geometry primitive and cheap enough for vectorized MuJoCo training, but models
those structural landmarks explicitly instead of replacing the ladder with a
low ramp.
"""

from __future__ import annotations

import math
from copy import deepcopy

import mujoco

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


LADDER_START_X = 0.18
LADDER_START_Z = 0.035
LADDER_END_X = 0.72
LADDER_TOP_Z = 0.67
LADDER_RAIL_Y = 0.17
LADDER_RUNG_HALF_WIDTH = 0.20
TABLE_X = 1.04
TABLE_HALF_X = 0.48
TABLE_HALF_Y = 0.48
TABLE_TOP_Z = 0.70


def _quat_z_to_vector(vector: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return a MuJoCo quaternion rotating local +Z onto ``vector``."""
    length = math.sqrt(sum(component * component for component in vector))
    if length < 1e-8:
        raise ValueError("orientation vector must be non-zero")
    target = tuple(component / length for component in vector)
    dot = target[2]
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    cross = (-target[1], target[0], 0.0)
    scale = math.sqrt(2.0 * (1.0 + dot))
    return (scale * 0.5, cross[0] / scale, cross[1] / scale, cross[2] / scale)


def _add_box(
    body: mujoco.MjsBody,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    **geom_kwargs,
) -> None:
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        quat=quat,
        size=size,
        rgba=rgba,
        **geom_kwargs,
    )


def _add_video_ladder_copy(
    spec: mujoco.MjSpec,
    env_id: int,
    origin: tuple[float, float, float],
    *,
    with_training_support: bool = False,
) -> None:
    """Add the steep ladder, floor foot, table top, and four table legs.

    ``with_training_support`` is deliberately restricted to the curriculum
    task.  It is a physical, collidable incline used to teach the first
    transition from flat walking to an inclined ladder; the final video task
    has no hidden ramp and is evaluated on the rung-only geometry.
    """
    root = spec.worldbody.add_body(name=f"ladder_env_{env_id}", pos=origin)
    prefix = f"video_ladder_{env_id}_"
    geom_kwargs = {
        "contype": 1,
        "conaffinity": 1,
        "friction": (1.15, 0.06, 0.01),
        "group": 0,
    }

    rail_vector = (LADDER_END_X - LADDER_START_X, 0.0, LADDER_TOP_Z - LADDER_START_Z)
    rail_length = math.sqrt(sum(component * component for component in rail_vector))
    rail_quat = _quat_z_to_vector(rail_vector)
    rail_midpoint = (
        (LADDER_START_X + LADDER_END_X) * 0.5,
        0.0,
        (LADDER_START_Z + LADDER_TOP_Z) * 0.5,
    )

    if with_training_support:
        # Curriculum-only support: a thin board under the rungs gives PPO a
        # continuous contact basin before it has learned discrete rung
        # stepping.  It is visible in the curriculum preview and disabled in
        # the final reference-aligned task.
        support_normal = (
            -(LADDER_TOP_Z - LADDER_START_Z) / rail_length,
            0.0,
            (LADDER_END_X - LADDER_START_X) / rail_length,
        )
        support_center = tuple(
            rail_midpoint[index] - 0.045 * support_normal[index]
            for index in range(3)
        )
        _add_box(
            root,
            f"{prefix}training_support",
            support_center,
            (0.035, 0.235, rail_length * 0.5),
            (0.37, 0.20, 0.07, 1.0),
            quat=rail_quat,
            **geom_kwargs,
        )

    # Rectangular timber rails match the squared-off beams visible in the
    # reference clip better than the V1 capsule approximation.
    for side, y in (("left", -LADDER_RAIL_Y), ("right", LADDER_RAIL_Y)):
        _add_box(
            root,
            f"{prefix}{side}_rail",
            (rail_midpoint[0], y, rail_midpoint[2]),
            (0.024, 0.024, rail_length * 0.5),
            (0.57, 0.29, 0.08, 1.0),
            quat=rail_quat,
            **geom_kwargs,
        )

    # The reference has many closely spaced crossbars.  Fourteen rungs preserve
    # that visual rhythm while keeping each step small enough for MicroDuck.
    # ``size`` is already expressed as (x-half-width, y-half-width,
    # z-half-width), so the long y axis must stay unrotated.  Rotating local
    # z toward y here would turn every rung into a vertical fence post.
    rung_quat = (1.0, 0.0, 0.0, 0.0)
    for rung_id, t in enumerate((0.04, 0.11, 0.18, 0.25, 0.32, 0.39, 0.46, 0.53, 0.60, 0.67, 0.74, 0.81, 0.88, 0.95)):
        x = LADDER_START_X + t * (LADDER_END_X - LADDER_START_X)
        z = LADDER_START_Z + t * (LADDER_TOP_Z - LADDER_START_Z)
        _add_box(
            root,
            f"{prefix}rung_{rung_id}",
            (x, 0.0, z),
            (0.017, LADDER_RUNG_HALF_WIDTH, 0.014),
            (0.73, 0.42, 0.12, 1.0),
            quat=rung_quat,
            **geom_kwargs,
        )

    # The lower timber foot is a visible landmark in the source clip and
    # prevents the ladder from looking like a floating ramp.
    _add_box(
        root,
        f"{prefix}floor_foot",
        # Keep the front edge beyond the robot's reset envelope.  The duck
        # should walk up to the first rung, not spawn interpenetrating the
        # bottom timber and get toppled before the policy can act.
        (LADDER_START_X + 0.07, 0.0, 0.025),
        (0.08, 0.28, 0.025),
        (0.58, 0.30, 0.08, 1.0),
        **geom_kwargs,
    )

    # Table top and legs reproduce the elevated landing target.  They remain
    # static collision bodies, so the final flip onto the desk is physical.
    _add_box(
        root,
        f"{prefix}table_top",
        (TABLE_X, 0.0, TABLE_TOP_Z - 0.035),
        (TABLE_HALF_X, TABLE_HALF_Y, 0.035),
        (0.78, 0.59, 0.35, 1.0),
        **geom_kwargs,
    )
    for leg_id, x in enumerate((TABLE_X - TABLE_HALF_X + 0.08, TABLE_X + TABLE_HALF_X - 0.08)):
        for side, y in (("left", -0.34), ("right", 0.34)):
            _add_box(
                root,
                f"{prefix}table_leg_{leg_id}_{side}",
                (x, y, (TABLE_TOP_Z - 0.07) * 0.5),
                (0.035, 0.035, (TABLE_TOP_Z - 0.07) * 0.5),
                (0.34, 0.38, 0.42, 1.0),
                **geom_kwargs,
            )

    # The source video visibly contains an office chair behind the desk.  It
    # is visual context only: keeping it out of the collision mask prevents a
    # decorative chair from changing the ladder task's contact dynamics.
    chair_kwargs = {**geom_kwargs, "contype": 0, "conaffinity": 0}
    # Put the chair just beyond the far table edge.  The earlier placement
    # was inside the tabletop footprint, so the table hid it completely in
    # the reference camera.
    chair_x = TABLE_X + 0.04
    chair_y = 0.30
    _add_box(
        root,
        f"{prefix}chair_seat",
        (chair_x, chair_y, 0.39),
        (0.20, 0.20, 0.035),
        (0.035, 0.045, 0.065, 1.0),
        **chair_kwargs,
    )
    _add_box(
        root,
        f"{prefix}chair_back",
        (chair_x + 0.13, chair_y, 0.76),
        (0.035, 0.20, 0.25),
        (0.035, 0.045, 0.065, 1.0),
        **chair_kwargs,
    )
    for side, y in (("left", chair_y - 0.23), ("right", chair_y + 0.23)):
        _add_box(
            root,
            f"{prefix}chair_arm_{side}",
        (chair_x + 0.04, y, 0.58),
            (0.16, 0.025, 0.025),
            (0.035, 0.045, 0.065, 1.0),
            **chair_kwargs,
        )
    _add_box(
        root,
        f"{prefix}chair_post",
        (chair_x, chair_y, 0.20),
        (0.025, 0.025, 0.19),
        (0.09, 0.10, 0.12, 1.0),
        **chair_kwargs,
    )

    # Two short landing blocks bridge the rail ends to the table edge, as in
    # the source clip's thick wooden top assembly.
    for side, y in (("left", -LADDER_RAIL_Y), ("right", LADDER_RAIL_Y)):
        _add_box(
            root,
            f"{prefix}top_landing_{side}",
            (LADDER_END_X + 0.05, y, LADDER_TOP_Z + 0.015),
            (0.09, 0.035, 0.022),
            (0.62, 0.34, 0.10, 1.0),
            **geom_kwargs,
        )


def _add_video_ladder_scene(spec: mujoco.MjSpec) -> None:
    """Insert one ladder at every origin already generated by mjlab.

    Reading ``env_origin_*`` sites is intentional: ``SceneCfg`` is deep-copied
    after registration and command-line overrides can change the vectorized
    environment count/spacing.  A closure over the original config would then
    silently create the wrong number of ladders.
    """
    origins = [
        tuple(float(value) for value in site.pos)
        for site in spec.sites
        if site.name.startswith("env_origin_")
    ]
    for env_id, origin in enumerate(origins):
        _add_video_ladder_copy(spec, env_id, origin)


def _add_video_ladder_support_scene(spec: mujoco.MjSpec) -> None:
    """Curriculum scene with the temporary continuous support board."""
    origins = [
        tuple(float(value) for value in site.pos)
        for site in spec.sites
        if site.name.startswith("env_origin_")
    ]
    for env_id, origin in enumerate(origins):
        _add_video_ladder_copy(
            spec, env_id, origin, with_training_support=True
        )


def _set_fixed_forward_command(cfg) -> None:
    command = cfg.commands["twist"]
    command.resampling_time_range = (8.0, 8.0)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.ranges.lin_vel_x = (0.12, 0.18)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(command, "heading_command"):
        command.heading_command = False
        command.ranges.heading = None
    command.debug_vis = False

    # The reference behavior is a dedicated climbing skill, not a multitask
    # head/body-pose benchmark.  Keep the command slots for checkpoint shape
    # compatibility, but remove their random targets while bootstrapping the
    # ladder policy.
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6


def make_microduck_video_ladder_env_cfg(play: bool = False):
    """Create a video-aligned MicroDuck ladder/table climbing environment."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # A flat ground plane supplies the common contact surface.  The ladder and
    # platform are added by spec_fn at every vectorized environment origin.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.env_spacing = 4.5
    cfg.scene.spec_fn = _add_video_ladder_scene
    cfg.scene.extent = 3.4

    cfg.episode_length_s = 18.0
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 2.4
    # mjlab's camera elevation sign is opposite the usual plotting convention:
    # negative values place the camera above the ground.  A steeper negative
    # angle exposes the desk surface and the chair, matching the reference
    # clip instead of showing the underside of the table.
    cfg.viewer.elevation = -25.0
    cfg.viewer.azimuth = 78.0
    cfg.viewer.max_extra_envs = 0

    # Start just before the first rung.  A small lateral/yaw perturbation is
    # retained so the policy does not only memorize one exact reset pose.
    cfg.events["reset_base"].params["pose_range"] = {
        # Start just before the physical ladder footprint.  This prevents
        # the reset pose from intersecting the first rail/rung while keeping
        # the approach phase part of the task.
        "x": (-0.16, -0.14),
        "y": (-0.04, 0.04),
        "z": (0.12, 0.13),
        "yaw": (-0.06, 0.06),
    }
    _set_fixed_forward_command(cfg)

    # Keep the first benchmark physically legible.  Robustness randomization
    # can be re-enabled after a policy has learned the nominal climb.
    cfg.events.pop("push_robot", None)
    for event_name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_mass_inertia",
        "randomize_joint_friction",
        "randomize_armature",
        "randomize_motor_gains",
        "randomize_base_orientation",
        "encoder_bias",
    ):
        cfg.events.pop(event_name, None)
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)

    # Add a compact forward-looking raycast.  Ladder geoms use group 0, so the
    # scan observes the actual rails, rungs and platform that MuJoCo collides
    # against.  The normal MicroDuck velocity config removes its base terrain
    # scan, so this task owns an explicit one rather than trying to reuse it.
    ladder_scan = RayCastSensorCfg(
        name="ladder_scan",
        frame=ObjRef(type="body", name="trunk_base", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.4, 0.8), resolution=0.1),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=bool(play),
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (ladder_scan,)
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "ladder_scan"},
        noise=Unoise(n_min=-0.02, n_max=0.02),
        scale=1.0,
    )
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": "ladder_scan"},
        scale=1.0,
    )

    # The velocity recipe already provides upright and foot-contact shaping.
    # Add potential-based route progress and height progress: holding a pose
    # pays zero, while advancing toward the platform pays for actual motion.
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].weight = 0.0
    cfg.rewards["upright"].weight = 3.0
    cfg.rewards["air_time"].weight = 0.5
    cfg.rewards["foot_clearance"].weight = -0.5
    cfg.rewards["foot_slip"].weight = -0.2
    cfg.rewards["ladder_path_progress"] = RewardTermCfg(
        func=microduck_mdp.ladder_path_progress,
        weight=10.0,
        params={
            "start_x": 0.0,
            "finish_x": TABLE_X - 0.08,
            "platform_z": TABLE_TOP_Z,
        },
    )
    cfg.rewards["ladder_height_progress"] = RewardTermCfg(
        func=microduck_mdp.ladder_height_progress,
        weight=8.0,
        params={"start_z": 0.115, "finish_z": TABLE_TOP_Z},
    )
    cfg.rewards["ladder_foot_alignment"] = RewardTermCfg(
        func=microduck_mdp.ladder_foot_alignment,
        weight=1.5,
        params={
            "start_x": LADDER_START_X,
            "start_z": LADDER_START_Z,
            "finish_x": LADDER_END_X,
            "finish_z": LADDER_TOP_Z,
            "rail_half_width": LADDER_RAIL_Y,
            "rung_count": 14,
        },
    )
    cfg.rewards["head_pose_tracking"].weight = 0.0

    cfg.terminations["ladder_success"] = TerminationTermCfg(
        func=microduck_mdp.ladder_success,
        time_out=False,
        params={
            "finish_x": TABLE_X - 0.12,
            "platform_z": TABLE_TOP_Z - 0.01,
            "max_tilt_deg": 35.0,
        },
    )

    # The base template's command/DR curricula refer to events removed above
    # and terrain levels that do not exist on a fixed ladder.  V2 keeps the
    # geometry and nominal contact physics explicit before robustness stages.
    cfg.curriculum = {}

    return cfg


MicroduckVideoLadderRlCfg = deepcopy(MicroduckRlCfg)
MicroduckVideoLadderRlCfg.experiment_name = "ladder_climb_video"
MicroduckVideoLadderRlCfg.run_name = "ladder-video-v2"
MicroduckVideoLadderRlCfg.save_interval = 100
MicroduckVideoLadderRlCfg.max_iterations = 4_000


def make_microduck_video_ladder_support_env_cfg(play: bool = False):
    """Create the physical support-board curriculum for the final ladder."""
    cfg = make_microduck_video_ladder_env_cfg(play=play)
    cfg.scene.spec_fn = _add_video_ladder_support_scene
    cfg.commands["twist"].ranges.lin_vel_x = (0.08, 0.12)
    cfg.rewards["ladder_foot_alignment"].weight = 2.0
    cfg.viewer.distance = 2.2
    return cfg


MicroduckVideoLadderSupportRlCfg = deepcopy(MicroduckVideoLadderRlCfg)
MicroduckVideoLadderSupportRlCfg.experiment_name = "ladder_climb_support"
MicroduckVideoLadderSupportRlCfg.run_name = "ladder-support-stage"
