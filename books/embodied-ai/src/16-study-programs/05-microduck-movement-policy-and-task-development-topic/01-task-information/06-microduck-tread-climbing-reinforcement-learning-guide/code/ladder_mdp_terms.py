"""Minimal ladder-specific MDP terms used by the MicroDuck V1 task.

Copy these functions into the project's task MDP module, or import them from a
small local module and reference them in RewardTermCfg/TerminationTermCfg.
"""

import math

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _ladder_progress_potential(
    env: ManagerBasedRlEnv,
    start_x: float,
    finish_x: float,
    platform_z: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    origin = env.scene.terrain.env_origins
    x = torch.nan_to_num(asset.data.root_link_pos_w[:, 0] - origin[:, 0], nan=0.0)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - origin[:, 2], nan=0.0)
    x_progress = torch.clamp((x - start_x) / max(finish_x - start_x, 1e-6), 0.0, 1.0)
    z_progress = torch.clamp((z - 0.115) / max(platform_z - 0.115, 1e-6), 0.0, 1.0)
    return 0.75 * x_progress + 0.25 * z_progress


def ladder_path_progress(
    env: ManagerBasedRlEnv,
    start_x: float = 0.0,
    finish_x: float = 1.7,
    platform_z: float = 0.24,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential difference for forward-and-up progress."""
    potential = _ladder_progress_potential(
        env, start_x, finish_x, platform_z, asset_cfg
    )
    previous = getattr(env, "_ladder_progress_potential_prev", None)
    if previous is None or previous.shape != potential.shape:
        previous = potential.clone()
    fresh = env.episode_length_buf <= 1
    previous[fresh] = potential[fresh]
    delta = potential - previous
    env._ladder_progress_potential_prev = potential.clone()
    return delta


def ladder_height_progress(
    env: ManagerBasedRlEnv,
    start_z: float = 0.115,
    finish_z: float = 0.24,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential difference for gaining height, capped at the platform."""
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    potential = torch.clamp((z - start_z) / max(finish_z - start_z, 1e-6), 0.0, 1.0)
    previous = getattr(env, "_ladder_height_potential_prev", None)
    if previous is None or previous.shape != potential.shape:
        previous = potential.clone()
    fresh = env.episode_length_buf <= 1
    previous[fresh] = potential[fresh]
    delta = potential - previous
    env._ladder_height_potential_prev = potential.clone()
    return delta


def ladder_success(
    env: ManagerBasedRlEnv,
    finish_x: float = 1.70,
    platform_z: float = 0.23,
    max_tilt_deg: float = 35.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Succeed only after reaching the top while remaining upright."""
    asset: Entity = env.scene[asset_cfg.name]
    origin = env.scene.terrain.env_origins
    x = torch.nan_to_num(asset.data.root_link_pos_w[:, 0] - origin[:, 0], nan=-1.0)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - origin[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return (x > finish_x) & (z > platform_z) & (
        cos_tilt > math.cos(math.radians(max_tilt_deg))
    )
