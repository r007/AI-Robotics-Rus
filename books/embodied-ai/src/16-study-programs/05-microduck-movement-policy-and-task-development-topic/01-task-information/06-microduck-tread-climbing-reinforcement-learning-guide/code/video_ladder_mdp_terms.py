"""Additional MDP shaping used by the video-aligned ladder task.

This term is geometric shaping only.  The MuJoCo contact solver still decides
whether a foot actually supports the robot; do not replace the contact sensor
or success condition with this score.
"""

import math

import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ladder_foot_alignment(
    env: ManagerBasedRlEnv,
    start_x: float,
    start_z: float,
    finish_x: float,
    finish_z: float,
    rail_half_width: float,
    rung_count: int,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Shape foot placement toward the nearest real ladder rung.

    The score is smooth in ladder-plane distance, lateral distance and rung
    distance.  It is intentionally not a contact label: collision, support,
    friction and slip remain physical MuJoCo quantities.
    """
    asset: Entity = env.scene[asset_cfg.name]
    origin = env.scene.terrain.env_origins
    feet = torch.nan_to_num(asset.data.site_pos_w[:, asset_cfg.site_ids], nan=0.0)
    rel = feet - origin[:, None, :]

    dx = float(finish_x - start_x)
    dz = float(finish_z - start_z)
    length = max(math.sqrt(dx * dx + dz * dz), 1e-6)
    tangent = rel.new_tensor((dx / length, 0.0, dz / length))
    normal = rel.new_tensor((-dz / length, 0.0, dx / length))
    along = (rel * tangent).sum(dim=-1)
    signed_normal = (rel * normal).sum(dim=-1)
    t = torch.clamp(along / length, 0.0, 1.0)
    slots = max(rung_count - 1, 1)
    nearest = torch.round(t * slots) / slots
    rung_along = nearest * length
    rung_z = start_z + nearest * (finish_z - start_z)
    rung_error = torch.sqrt((along - rung_along) ** 2 + (rel[..., 2] - rung_z) ** 2)
    score = torch.exp(
        -((signed_normal / 0.045) ** 2)
        -((rel[..., 1] / max(rail_half_width + 0.04, 1e-4)) ** 2)
        -((rung_error / 0.065) ** 2)
    )

    root_x = torch.nan_to_num(asset.data.root_link_pos_w[:, 0] - origin[:, 0], nan=0.0)
    gate = torch.clamp((root_x - (start_x - 0.08)) / 0.25, 0.0, 1.0)
    gate = gate * (root_x < finish_x + 0.10).float()
    return score.mean(dim=-1) * gate
