"""Regression tests for the V3 physical foothold-planning ladder task."""

import mujoco
import torch

from mjlab.tasks.registry import list_tasks
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks.microduck_video_ladder_env_cfg import (
    _add_video_ladder_copy,
)
from mjlab_microduck.tasks.microduck_video_ladder_footstep_env_cfg import (
    RUNG_COUNT,
    STAGE_SCHEDULE,
    MicroduckVideoLadderFootstepRlCfg,
    make_microduck_video_ladder_footstep_env_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def _compiled_ladder_model() -> mujoco.MjModel:
    robot_cfg = MICRODUCK_WALK_ROBOT_CFG
    spec = robot_cfg.spec_fn()
    for collision_cfg in robot_cfg.collisions:
        collision_cfg.edit_spec(spec)
    _add_video_ladder_copy(spec, 0, (0.0, 0.0, 0.0), rung_count=RUNG_COUNT)
    return spec.compile()


def _can_collide(model: mujoco.MjModel, first: int, second: int) -> bool:
    return bool(
        (model.geom_contype[first] & model.geom_conaffinity[second])
        and (model.geom_contype[second] & model.geom_conaffinity[first])
    )


def test_v3_task_is_registered_with_staged_footholds():
    assert "Mjlab-Video-Ladder-Footstep-MicroDuck" in list_tasks()
    assert STAGE_SCHEDULE[-1][1] == RUNG_COUNT
    assert [count for _, count in STAGE_SCHEDULE] == [0, 1, 3, 7, 14, 14, 14]
    assert STAGE_SCHEDULE[-1][0] > STAGE_SCHEDULE[-2][0]
    assert MicroduckVideoLadderFootstepRlCfg.algorithm.symmetry_cfg is None

    cfg = make_microduck_video_ladder_footstep_env_cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.0)
    assert cfg.rewards["track_linear_velocity"].weight == 0.0
    assert cfg.rewards["track_angular_velocity"].weight == 0.0
    assert "ladder_path_progress" not in cfg.rewards
    assert "ladder_height_progress" not in cfg.rewards
    assert "ladder_phase_contact_velocity" in cfg.rewards
    collision_params = cfg.terminations["ladder_body_collision"].params
    assert collision_params["rung_count"] == RUNG_COUNT
    assert collision_params["stage_schedule"] == STAGE_SCHEDULE
    assert cfg.events["ladder_domain_randomization"].params["start_step"] == STAGE_SCHEDULE[-1][0]
    assert "ladder_foot_contact" in {
        sensor.name for sensor in cfg.scene.sensors
    }
    body_sensor = next(
        sensor for sensor in cfg.scene.sensors if sensor.name == "ladder_body_contact"
    )
    assert body_sensor.primary.mode == "geom"
    expected_observations = {
        "ladder_step_targets",
        "ladder_phase",
        "ladder_swing_foot",
        "ladder_foot_positions",
        "ladder_foot_contacts",
        "ladder_stage",
    }
    assert expected_observations <= set(cfg.observations["actor"].terms)
    assert expected_observations <= set(cfg.observations["critic"].terms)


def test_curriculum_switches_only_at_declared_step_boundaries():
    class DummyEnv:
        num_envs = 2
        device = "cpu"

    env = DummyEnv()
    for step, expected in (
        (0, 0),
        (STAGE_SCHEDULE[1][0] - 1, 0),
        (STAGE_SCHEDULE[1][0], 1),
        (STAGE_SCHEDULE[2][0], 3),
        (STAGE_SCHEDULE[3][0], 7),
        (STAGE_SCHEDULE[4][0], 14),
    ):
        env.common_step_counter = step
        active = microduck_mdp._ladder_active_rungs(
            env, RUNG_COUNT, STAGE_SCHEDULE
        )
        assert torch.equal(active, torch.full((2,), expected, dtype=torch.long))


def test_ladder_has_fourteen_rungs_and_no_continuous_support_board():
    model = _compiled_ladder_model()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    rung_names = [name for name in names if name and name.startswith("video_ladder_0_rung_")]
    assert len(rung_names) == RUNG_COUNT
    assert not any(name.endswith("training_support") for name in names if name)


def test_ladder_collision_masks_hit_feet_and_body_but_not_chair():
    model = _compiled_ladder_model()
    ladder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "video_ladder_0_left_rail"
    )
    foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"
    )
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "trunk_body_collision",
            "left_leg_body_collision",
            "right_leg_body_collision",
        )
    ]
    assert all(index >= 0 for index in body_ids)
    assert all(model.geom_contype[index] == 2 for index in body_ids)
    assert all(model.geom_conaffinity[index] == 2 for index in body_ids)
    body_id = body_ids[0]
    assert model.geom_contype[ladder_id] == 3
    assert model.geom_conaffinity[ladder_id] == 3
    assert _can_collide(model, ladder_id, foot_id)
    assert _can_collide(model, ladder_id, body_id)

    for name in (
        "video_ladder_0_chair_seat",
        "video_ladder_0_chair_back",
        "video_ladder_0_chair_post",
    ):
        chair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[chair_id] == 0
        assert model.geom_conaffinity[chair_id] == 0


def test_initial_robot_pose_has_no_ladder_interpenetration():
    model = _compiled_ladder_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        pair = {first, second}
        assert not any(
            name and name.startswith("video_ladder_0_") for name in pair
        ), f"initial ladder contact: {first} vs {second}"


def test_body_mask_produces_real_mujoco_contact_when_overlapped():
    model = _compiled_ladder_model()
    data = mujoco.MjData(model)
    root_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_qpos = model.jnt_qposadr[root_joint]
    data.qpos[:] = model.qpos0
    # Put the trunk through a middle rung.  This is an intentional synthetic
    # overlap used to prove the contact masks reach MuJoCo's solver.
    data.qpos[root_qpos : root_qpos + 3] = (0.45, 0.0, 0.35)
    mujoco.mj_forward(model, data)

    body_contact = False
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first, second = contact.geom1, contact.geom2
        first_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first)
        second_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second)
        first_is_ladder = bool(first_name and first_name.startswith("video_ladder_0_"))
        second_is_ladder = bool(second_name and second_name.startswith("video_ladder_0_"))
        other = second if first_is_ladder else first if second_is_ladder else -1
        if other >= 0 and model.geom_contype[other] == 2:
            body_contact = True
            break
    assert body_contact, "body-mask overlap did not reach MuJoCo contact data"
