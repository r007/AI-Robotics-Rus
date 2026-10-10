"""
Базовый класс решений VLM
============
Инкапсулирует построение system prompt, построение пользовательского текста, разбор ответа, стратегию резервных правил и ведение журнала.
Подклассу достаточно реализовать метод _call_model().
"""

import json
from datetime import datetime
import numpy as np


class BaseVLMDecisionEngine:
    """
    Базовый класс движка решений большой модели.

    Подкласс должен реализовать:
        _call_model(system_prompt, user_text, key_images) -> str
            Вызывает базовую модель и возвращает исходный текстовый ответ.
    """

    def __init__(self, config):
        self.cfg = config
        self.targets = config.TARGET_OBJECTS
        self.reasoning_log = []
        self.count = 0

    # ── Публичный интерфейс ──────────────────────────────────────────────

    def make_decision(self, key_images, detection_summary, frontier_info,
                      visited_targets, nav_history, agent_pos) -> dict:
        """
        Выполняет одно решение VLM.

        Returns
        -------
        dict: {action, target_index, reasoning, confidence}
        """
        self.count += 1
        system_prompt = self._build_system_prompt()
        user_text = self._build_user_text(
            agent_pos, visited_targets, detection_summary,
            frontier_info, nav_history
        )

        try:
            raw = self._call_model(system_prompt, user_text, key_images).strip()
            print(f"\n{'=' * 60}")
            print(f"[Решение VLM #{self.count}] Исходный ответ:")
            print(raw)
            print(f"{'=' * 60}\n")
            decision = self._parse(raw)
        except Exception as e:
            print(f"[VLM Error] {e}")
            decision = self._fallback(detection_summary, frontier_info, visited_targets)

        self.reasoning_log.append({
            "decision_id": self.count,
            "agent_position": agent_pos.tolist() if isinstance(agent_pos, np.ndarray) else list(agent_pos),
            "detections_count": len(detection_summary) if detection_summary else 0,
            "frontiers_count": len(frontier_info) if frontier_info else 0,
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
        })
        return decision

    def save_log(self, path: str):
        """Сохраняет журнал вывода в JSON-файл."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.reasoning_log, f, ensure_ascii=False, indent=2)
        print(f"[LOG] Журнал вывода -> {path}")

    # ── Переопределяется в подклассах ────────────────────────────────────

    def _call_model(self, system_prompt: str, user_text: str, key_images) -> str:
        raise NotImplementedError("Подкласс обязан реализовать _call_model()")

    # ── Внутренние утилиты ──────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return f"""Ты — «мозг» принятия решений навигационного робота с воплощённым ИИ. Твоя задача — найти в помещении все следующие целевые объекты: {self.targets}\n\nНа основе текущего изображения, результатов YOLO (с 3D-координатами и расстоянием до объектов) и доступных фронтиров исследования прими решение о навигации.\n\nПравила решения:\n1. Если обнаружен целевой объект на разумном расстоянии (< 8 м), в первую очередь иди к нему\n2. Если обнаружено несколько целевых объектов, выбери ближайший непосещённый\n3. Если целевых объектов не обнаружено, выбери наилучший фронтир для исследования\n4. При выборе фронтира предпочитай направление с умеренным расстоянием и большей неисследованной областью\n5. Если все цели уже посещены, верни "task_complete"\n\nСтрого отвечай в следующем JSON-формате, ничего другого не выводи:\n{{\n    "thinking": "Твой подробный ход рассуждений: 1) что увидел 2) анализ вариантов 3) обоснование решения",\n    "action": "go_to_object" или "explore_frontier" или "task_complete",\n    "target_index": индекс выбранной цели в соответствующем списке (с 0),\n    "confidence": уверенность от 0 до 1\n}}"""

    def _build_user_text(self, agent_pos, visited_targets,
                         detection_summary, frontier_info, nav_history) -> str:
        text = f"""## Решение № {self.count}\n\n### Текущее состояние\n- Позиция agent: [{agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f}]\n- Посещённые цели: {visited_targets if visited_targets else 'нет'}\n- Цели для поиска: {[t for t in self.targets if t not in visited_targets]}\n\n### Результаты YOLO (сводка за поворот на 360°)\n"""""
        if detection_summary:
            for i, d in enumerate(detection_summary):
                p = (f"[{d['world_pos'][0]:.2f}, {d['world_pos'][1]:.2f}, {d['world_pos'][2]:.2f}]"
                     if d.get("world_pos") else "неизвестно")
                dist = f"{d['distance']:.2f}m" if d.get("distance") is not None else "неизвестно"
                tag = "цель" if d["is_target"] else "не цель"
                text += f"  [{i}] {d['class']} (уверенность:{d['confidence']:.2f}) 3D-позиция:{p} расстояние:{dist} {tag}\n"
        else:
            text += "  За полный поворот объекты не обнаружены\n"

        text += "\n### Доступные фронтиры\n"
        if frontier_info:
            for i, f in enumerate(frontier_info):
                text += (f"  [{i}] позиция:[{f['position'][0]:.2f}, {f['position'][1]:.2f}, "
                         f"{f['position'][2]:.2f}] расстояние:{f['distance']:.2f} м\n")
        else:
            text += "  Нет доступных фронтиров\n"

        text += f"\n### История навигации\n{nav_history}\n"
        return text

    def _parse(self, text: str) -> dict:
        """Разбирает JSON из ответа VLM; при ошибке использует резервный разбор по ключевым словам."""
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                d = json.loads(text[l:r + 1])
                return {
                    "action": d.get("action", "explore_frontier"),
                    "target_index": d.get("target_index", 0),
                    "reasoning": d.get("thinking", ""),
                    "confidence": d.get("confidence", 0.5),
                }
            except Exception:
                pass

        if "go_to_object" in text:
            return {"action": "go_to_object", "target_index": 0,
                    "reasoning": text, "confidence": 0.5}
        if "task_complete" in text:
            return {"action": "task_complete", "target_index": -1,
                    "reasoning": text, "confidence": 0.5}
        return {"action": "explore_frontier", "target_index": 0,
                "reasoning": text, "confidence": 0.3}

    def _fallback(self, dets, fronts, visited) -> dict:
        """Стратегия резервных правил при сбое вызова VLM."""
        if dets:
            tgts = [d for d in dets if d["is_target"] and d["class"] not in visited
                    and d.get("distance") is not None and d["distance"] < 8]
            if tgts:
                tgts.sort(key=lambda x: x["distance"])
                idx = dets.index(tgts[0])
                return {"action": "go_to_object", "target_index": idx,
                        "reasoning": f"[Резерв] Обнаружено {tgts[0]['class']}", "confidence": 0.6}
        if fronts:
            return {"action": "explore_frontier", "target_index": 0,
                    "reasoning": "[Резерв] Цели нет, исследуем фронтиры", "confidence": 0.3}
        return {"action": "explore_frontier", "target_index": 0,
                "reasoning": "[Резерв] Случайное исследование", "confidence": 0.1}
