"""
Модуль карты — туман войны и обнаружение фронтиров
===================================
Содержит:
  - преобразование координат плана сверху
  - обновление тумана войны (раскрытие секторального поля зрения)
  - обнаружение фронтиров (Frontier Exploration)
"""

import numpy as np
import cv2

from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector


# ── Преобразование координат ──────────────────────────────────────────────────

def convert_meters_to_pixel(meters: float, map_resolution: int, sim) -> int:
    """Метры → пиксели (при разрешении плана сверху)."""
    return int(meters / maps.calculate_meters_per_pixel(map_resolution, sim=sim))


def map_coors_to_pixel(position, top_down_map: np.ndarray, sim) -> np.ndarray:
    """Мировые координаты → пиксельные координаты плана сверху [row, col]."""
    a_x, a_y = maps.to_grid(
        position[2], position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    return np.array([a_x, a_y])


def pixel_to_world(pixel: np.ndarray, agent_y: float,
                   top_down_map: np.ndarray, sim) -> np.ndarray:
    """Пиксельные координаты плана сверху → навигируемые мировые координаты (snap к navmesh)."""
    x, y = pixel
    realworld_x, realworld_y = maps.from_grid(
        x, y,
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    return sim.pathfinder.snap_point([realworld_y, agent_y, realworld_x])


# ── Угол ориентации ────────────────────────────────────────────────────

def get_polar_angle(agent_state) -> float:
    """Возвращает текущий угол курса agent (радианы, полярная система плана сверху)."""
    ref_rotation = agent_state.rotation
    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return float(np.array(phi) + np.pi)


def wrap_heading(heading: float) -> float:
    """Нормирует угол курса в [-π, π]."""
    return (heading + np.pi) % (2 * np.pi) - np.pi


# ── Туман войны ──────────────────────────────────────────────────────

def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
) -> np.ndarray:
    """
    По текущему положению и ориентации обновляет маску тумана войны (секторальное поле зрения).

    Parameters
    ----------
    top_down_map          : бинарная карта проходимых областей (>0 — проходимо)
    current_fog_of_war_mask: маска уже исследованных областей
    current_point         : пиксельные координаты agent на плане сверху [row, col]
    current_angle         : угол курса (радианы)
    fov                   : горизонтальный угол обзора (градусы)
    max_line_len          : радиус видимости (пиксели)

    Returns
    -------
    Обновлённая маска исследования
    """
    curr_pt_cv2 = current_point[::-1].astype(int)
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))

    cone_mask = cv2.ellipse(
        np.zeros_like(top_down_map, dtype=np.uint8),
        tuple(curr_pt_cv2),
        (int(max_line_len), int(max_line_len)),
        0,
        angle_cv2 - fov / 2,
        angle_cv2 + fov / 2,
        1,
        -1,
    )

    visible = cv2.bitwise_and(cone_mask, (top_down_map > 0).astype(np.uint8))
    return cv2.bitwise_or(current_fog_of_war_mask.astype(np.uint8), visible.astype(np.uint8))


# ── Обнаружение фронтиров ────────────────────────────────────────────────

def detect_frontiers(full_map: np.ndarray, explored_mask: np.ndarray,
                     min_area: int = 8):
    """
    Обнаруживает фронтиры на границе исследованных и неисследованных областей.

    Parameters
    ----------
    full_map      : бинарная карта проходимых областей всей сцены
    explored_mask : маска исследованных областей
    min_area      : минимальная площадь связной области (пиксели), отсев шума

    Returns
    -------
    (pts, frontier_mask)
        pts           : list of np.ndarray [row, col], пиксельные координаты фронтиров
        frontier_mask : маска областей фронтиров
    """
    navigable = (full_map > 0).astype(np.uint8)
    explored = (explored_mask > 0).astype(np.uint8)
    unexplored_nav = (navigable & (1 - explored)).astype(np.uint8)

    border = cv2.morphologyEx(explored, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    frontier_mask = (border & unexplored_nav).astype(np.uint8)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask, 8)
    pts = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            c = centroids[i]
            pts.append(np.array([int(c[1]), int(c[0])]))  # row, col
    return pts, frontier_mask
