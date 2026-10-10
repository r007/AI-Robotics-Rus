"""
Модуль планирования — планирование пути и выполнение навигации
==================================
Предоставляет:
  - планирование пути на основе habitat_sim GreedyGeodesicFollower
  - навигацию к целевой точке (пошаговое выполнение действий)
  - растеризацию пути в пиксели (для визуализации)
"""

import numpy as np
import habitat_sim

from mapping.map_utils import map_coors_to_pixel


class PathPlanner:
    """
    Инкапсулирует планирование пути и выполнение навигации в habitat_sim.
    Хранится в EmbodiedNavigator.
    """

    def __init__(self, sim, agent, config):
        self.sim = sim
        self.agent = agent
        self.pf = sim.pathfinder
        self.cfg = config

    def plan_path_pixels(self, target_world: np.ndarray,
                         top_down_map: np.ndarray) -> list:
        """
        Планирует путь от текущей позиции до цели; возвращает список пиксельных координат (для визуализации).

        Parameters
        ----------
        target_world  : мировые координаты цели
        top_down_map  : карта плана сверху (для преобразования координат)

        Returns
        -------
        list of np.ndarray [row, col]; пустой список означает, что планирование не удалось
        """
        state = self.agent.get_state()
        island = self.pf.get_island(state.position)
        snapped = self.pf.snap_point(point=target_world, island_index=island)

        if not self.pf.is_navigable(snapped):
            return []

        path = habitat_sim.ShortestPath()
        path.requested_start = state.position
        path.requested_end = snapped

        if self.pf.find_path(path):
            return [
                map_coors_to_pixel(p, top_down_map, self.sim).astype(int)
                for p in path.points
            ]
        return []

    def navigate_to(self, target_position: np.ndarray,
                    top_down_map: np.ndarray,
                    on_step_callback=None) -> None:
        """
        Навигация к целевой точке с callback на каждом шаге (для записи кадров видео, обновления тумана и т.п.).

        Parameters
        ----------
        target_position   : мировые координаты цели
        top_down_map      : карта плана сверху (для преобразования координат)
        on_step_callback  : callable(obs, agent_state) или None
                            вызывается после каждого шага действия; если callback возвращает True, запрашивается досрочное прерывание
        """
        state = self.agent.get_state()
        island = self.pf.get_island(state.position)
        snapped = self.pf.snap_point(point=target_position, island_index=island)

        if not self.pf.is_navigable(snapped):
            print(f"[NAV] Цель недостижима: {target_position}")
            return

        follower = habitat_sim.GreedyGeodesicFollower(
            self.pf, self.agent,
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        try:
            action_list = follower.find_path(snapped)
        except Exception as e:
            print(f"[NAV] Ошибка планирования пути: {e}")
            return

        for action in action_list:
            if action is None:
                continue
            obs = self.sim.step(action)
            s = self.agent.get_state()

            if on_step_callback is not None:
                stop = on_step_callback(obs, s)
                if stop:
                    break
