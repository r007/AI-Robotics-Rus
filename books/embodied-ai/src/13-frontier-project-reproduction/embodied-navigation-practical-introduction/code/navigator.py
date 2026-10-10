"""
Главный навигатор — EmbodiedNavigator
==============================
Объединяет модули восприятия, построения карты, планирования, решений VLM и вывода и запускает полный цикл навигации.
"""

import numpy as np
import habitat_sim
from habitat.utils.visualizations import maps

from config import Config
from utils import setup_simulator, get_camera_intrinsics, depth_to_3d_world
from perception import YOLODetector
from mapping import (
    convert_meters_to_pixel,
    map_coors_to_pixel,
    pixel_to_world,
    get_polar_angle,
    reveal_fog_of_war,
    detect_frontiers,
)
from planning import PathPlanner
from output import VisualOutput


class EmbodiedNavigator:
    """Навигатор воплощённого ИИ: координирует все подмодули для выполнения задачи поиска цели."""

    def __init__(self, config: Config, vlm_engine):
        self.cfg = config
        self.vlm = vlm_engine

        # Симулятор и базовые компоненты
        self.sim = setup_simulator(config)
        self.agent = self.sim.get_agent(0)
        self.pf = self.sim.pathfinder
        self.K = get_camera_intrinsics(config)

        # Восприятие
        self.detector = YOLODetector(config)

        # Состояние карты (действует после initialize())
        self.tdm = None
        self.fog = None
        self.vis_px = 0

        # Планирование и визуализация (действует после initialize())
        self.planner = None
        self.vis = None

        # Состояние выполнения
        self.total_steps = 0
        self.decision_count = 0
        self.visited_targets = set()
        self.visited_frontier_set = set()
        self.nav_history = []
        self.video_frames = []
        self.current_frontiers = []

    # ── Инициализация ────────────────────────────────────────────────

    def initialize(self):
        """Случайно размещает agent, инициализирует карту и подмодули."""
        pos = self.pf.get_random_navigable_point()
        st = habitat_sim.AgentState()
        st.position = pos
        self.agent.set_state(st)

        self.tdm = maps.get_topdown_map_from_sim(
            self.sim, map_resolution=self.cfg.MAP_RESOLUTION, draw_border=False
        )
        self.tdm = (self.tdm > 0).astype(np.uint8)
        self.fog = np.zeros_like(self.tdm, dtype=np.uint8)
        self.vis_px = convert_meters_to_pixel(
            self.cfg.VISIBLE_RADIUS, self.cfg.MAP_RESOLUTION, self.sim
        )

        self.planner = PathPlanner(self.sim, self.agent, self.cfg)
        self.vis = VisualOutput(self.cfg, self.tdm, self.fog)

        print(f"[INIT] Старт {pos}, цели {self.cfg.TARGET_OBJECTS}")

    # ── Внутренние утилиты ──────────────────────────────────────────────

    def _update_fog(self, state):
        px = map_coors_to_pixel(state.position, self.tdm, self.sim)
        ang = get_polar_angle(state)
        self.fog = reveal_fog_of_war(
            self.tdm, self.fog, px, ang,
            fov=self.cfg.HFOV // 2, max_line_len=self.vis_px,
        )
        self.vis.update_fog(self.fog)

    def _record_frame(self, rgb, state, phase=""):
        frame = rgb.copy()
        self.vis.hud(frame, state, self.total_steps, self.decision_count,
                     self.visited_targets, self.cfg.TARGET_OBJECTS, phase=phase)
        out = self.vis.compose_frame(frame, state, self.sim,
                                     frontiers=self.current_frontiers)
        self.video_frames.append(out)

    def _record_detection_frame(self, rgb, dets, state):
        ann = self.detector.draw_detections(rgb, dets)
        self._record_frame(ann, state, phase="[SCAN]")

    # ── Восприятие: круговой обзор ────────────────────────────────────────

    def spin_and_observe(self):
        """Поворачивается на месте на полный круг и собирает RGB, глубину, состояние и результаты YOLO."""
        colors, depths, states, all_dets = [], [], [], []

        for _ in range(self.cfg.SPIN_STEPS):
            obs = self.sim.step("turn_left")
            c = obs["color_sensor"][:, :, :3]
            d = obs["depth_sensor"]
            s = self.agent.get_state()

            colors.append(c)
            depths.append(d)
            states.append(s)

            dets = self.detector.detect(c)
            for det in dets:
                wp = depth_to_3d_world(d, det["bbox"], s, self.K)
                det["world_pos"] = wp
                det["distance"] = (
                    float(np.linalg.norm(np.array(s.position) - wp))
                    if wp is not None else None
                )
                all_dets.append(det)

            self._record_detection_frame(c, dets, s)
            self._update_fog(s)
            self.total_steps += 1

        return colors, depths, states, all_dets

    @staticmethod
    def merge_detections(dets: list) -> list:
        """Объединяет результаты детекции без дублей (для одного класса и позиции оставляет с наибольшей уверенностью)."""
        merged = {}
        for d in dets:
            if d["world_pos"] is None:
                continue
            key = (d["class"], tuple(np.round(d["world_pos"], 1)))
            if key not in merged or d["confidence"] > merged[key]["confidence"]:
                merged[key] = d
        return list(merged.values())

    # ── Карта: получение фронтиров ──────────────────────────────────────

    def get_frontiers(self) -> list:
        """Находит текущие исследуемые фронтиры, отфильтровывает посещённые и возвращает 8 ближайших."""
        state = self.agent.get_state()
        area_thr = max(
            8,
            convert_meters_to_pixel(
                self.cfg.AREA_THRESHOLD_M2, self.cfg.MAP_RESOLUTION, self.sim
            ) // 10,
        )
        fpxs, frontier_mask = detect_frontiers(self.tdm, self.fog, min_area=area_thr)
        self.vis.last_frontier_mask = frontier_mask

        infos = []
        for fp in fpxs:
            wp = pixel_to_world(fp, state.position[1], self.tdm, self.sim)
            if wp is None or np.any(np.isnan(wp)):
                continue
            if self.pf.is_navigable(wp):
                d = float(np.linalg.norm(np.array(state.position) - wp))
                key = tuple(np.round(wp, 1))
                if key not in self.visited_frontier_set:
                    infos.append({"position": wp, "distance": d, "pixel": fp})
        infos.sort(key=lambda x: x["distance"])
        return infos[:8]

    # ── Планирование: переход к цели ──────────────────────────────────────

    def navigate_to(self, target_position: np.ndarray):
        """Планирует и выполняет движение к целевой точке, записывая кадры видео по пути."""
        # Сначала вычисляем пиксели пути для визуализации карты
        self.vis.path_pixels = self.planner.plan_path_pixels(
            target_position, self.tdm
        )

        def on_step(obs, state):
            rgb = obs["color_sensor"][:, :, :3]
            self._record_frame(rgb, state, phase="[MOVE]")
            self._update_fog(state)
            self.total_steps += 1
            return self.total_steps >= self.cfg.MAX_STEPS  # True = досрочно прервать

        self.planner.navigate_to(target_position, self.tdm, on_step_callback=on_step)
        self.vis.path_pixels = []

    # ── Главный цикл ────────────────────────────────────────────────

    def run(self):
        self.initialize()
        print(f"\n{'=' * 60}")
        print(f"  Навигация начата  цели: {self.cfg.TARGET_OBJECTS}")
        print(f"{'=' * 60}\n")

        while self.total_steps < self.cfg.MAX_STEPS:
            print(f"\n[Шаг {self.total_steps}] Круговой обзор на месте...")
            colors, depths, states, raw_dets = self.spin_and_observe()
            all_dets = self.merge_detections(raw_dets)
            tgt_dets = [d for d in all_dets if d["is_target"]]

            print(f"  Обнаружено объектов: {len(all_dets)}, из них целевых: {len(tgt_dets)}:")
            for d in tgt_dets:
                dist_text = f"{d['distance']:.2f}" if d.get("distance") is not None else "None"
                print(f"{d['class']}  conf={d['confidence']:.2f}"
                      f"dist={dist_text}m  pos={d['world_pos']}")

            agent_state = self.agent.get_state()
            frontiers = self.get_frontiers()
            self.current_frontiers = frontiers
            print(f"  Фронтиров: {len(frontiers)}")

            # Формируем сводку входных данных VLM
            idx = np.linspace(0, len(colors) - 1, min(4, len(colors)), dtype=int)
            key_imgs = [colors[i] for i in idx]

            det_summary = [{
                "class": d["class"],
                "confidence": d["confidence"],
                "is_target": d["is_target"],
                "world_pos": d["world_pos"].tolist() if d.get("world_pos") is not None else None,
                "distance": d.get("distance"),
            } for d in all_dets]

            frontier_summary = [{
                "position": (f["position"].tolist()
                             if isinstance(f["position"], np.ndarray)
                             else list(f["position"])),
                "distance": f["distance"],
            } for f in frontiers]

            hist_text = f"Решений принято: {self.decision_count}, шагов всего: {self.total_steps}\n"
            for h in self.nav_history[-5:]:
                hist_text += f"  - {h}\n"

            # Решение VLM
            print(f"[Шаг {self.total_steps}] Решение VLM...")
            decision = self.vlm.make_decision(
                key_images=key_imgs,
                detection_summary=det_summary,
                frontier_info=frontier_summary,
                visited_targets=list(self.visited_targets),
                nav_history=hist_text,
                agent_pos=np.array(agent_state.position),
            )
            self.decision_count += 1

            print(f"  Действие: {decision['action']}")
            print(f"  Рассуждение: {decision['reasoning'][:300]}")

            # Выполняем решение
            if decision["action"] == "task_complete":
                print(f"\n Задача выполнена! Найдено: {list(self.visited_targets)}")
                self.nav_history.append(f"Решение {self.decision_count}: задача выполнена")
                break

            elif decision["action"] == "go_to_object":
                i = decision["target_index"]
                if i < len(all_dets) and all_dets[i].get("world_pos") is not None:
                    det = all_dets[i]
                    print(f" Направляемся к {det['class']} @ {det['world_pos']}")
                    self.navigate_to(det["world_pos"])

                    final_dist = float(np.linalg.norm(
                        np.array(self.agent.get_state().position) - det["world_pos"]
                    ))
                    if final_dist < self.cfg.TARGET_REACH_DIST:
                        self.visited_targets.add(det["class"])
                        print(f" Достигли {det['class']}  расстояние {final_dist:.2f} м")
                        self.nav_history.append(
                            f"Решение {self.decision_count}: → {det['class']} ({final_dist:.1f} м)"
                        )
                    else:
                        print(f" Приближаемся к {det['class']}  расстояние {final_dist:.2f} м")
                        self.nav_history.append(
                            f"Решение {self.decision_count}: → {det['class']} ({final_dist:.1f} м)"
                        )
                else:
                    print(" Индекс недействителен, случайное исследование")
                    rp = self.pf.get_random_navigable_point()
                    self.navigate_to(rp)
                    self.nav_history.append(f"Решение {self.decision_count}: индекс недействителен, случайно")

            elif decision["action"] == "explore_frontier":
                i = decision["target_index"]
                if frontiers and i < len(frontiers):
                    fp = frontiers[i]
                    self.visited_frontier_set.add(tuple(np.round(fp["position"], 1)))
                    print(f" Исследуем фронтир[{i}] dist={fp['distance']:.1f} м")
                    self.navigate_to(fp["position"])
                    self.nav_history.append(
                        f"Решение {self.decision_count}: фронтир[{i}] {fp['distance']:.1f} м"
                    )
                else:
                    print(" Фронтиров нет, случайно")
                    rp = self.pf.get_random_navigable_point()
                    self.navigate_to(rp)
                    self.nav_history.append(f"Решение {self.decision_count}: случайно")

            # Проверяем, все ли цели найдены
            if self.visited_targets >= set(self.cfg.TARGET_OBJECTS):
                print(f"\n Все цели найдены: {list(self.visited_targets)}")
                break

        # Шаги исчерпаны
        if self.total_steps >= self.cfg.MAX_STEPS:
            print(f"\n Шаги исчерпаны ({self.cfg.MAX_STEPS})")
            print(f"  Найдено: {list(self.visited_targets)}")
            print(f"  Не найдено: {[t for t in self.cfg.TARGET_OBJECTS if t not in self.visited_targets]}")

        self._save()
        self.sim.close()

    # ── Сохранение результатов ──────────────────────────────────────────

    def _save(self):
        self.vis.save_video(self.video_frames, self.cfg.OUTPUT_VIDEO, self.cfg.VIDEO_FPS)
        self.vlm.save_log(self.cfg.OUTPUT_LOG)

        print(f"\n{'=' * 60}")
        print(" Полный ход рассуждений большой модели")
        print(f"{'=' * 60}")
        for e in self.vlm.reasoning_log:
            print(f"\n--- Решение #{e['decision_id']} ---")
            print(f"  Позиция: {e['agent_position']}")
            print(f"  Обнаружено: {e['detections_count']} объектов,  фронтиров: {e['frontiers_count']}")
            print(f"  Действие: {e['decision']['action']}")
            print(f"  Уверенность: {e['decision'].get('confidence', '-')}")
            print(f"  Рассуждение: {e['decision']['reasoning']}")
