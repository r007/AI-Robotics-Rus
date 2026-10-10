"""FastAPI entrypoint for the OpenClaw web frontend."""

from __future__ import annotations

import json
from io import BytesIO
import os
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from grasp_process import (
    TABLE_SURFACE_Z,
    _move_joint_waypoint,
    estimate_body_image_bbox,
    estimate_direct_grasp_target_world,
    estimate_place_target_world,
    execute_grasp,
    get_apple_rack_slot_world,
    get_available_table_sponge_bodies,
    get_sponge_rack_slot_world,
    run_grasp_inference,
)
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from integrations import notify_robot_backend
from vlm_process import segment_image
from workflow_hooks import get_inventory_store, record_task_success_effects


FRONTEND_DIR = ROOT_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"


def _assert_frontend_present() -> None:
    if not INDEX_FILE.exists():
        raise RuntimeError(
            f"Frontend assets not found at {INDEX_FILE}. "
            "Build or copy the frontend folder first."
        )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _extract_objects(text: str) -> list[str]:
    aliases = {
        # Ключевые слова и синонимы команд на китайском языке (苹果 — яблоко, 放 — положить и т.п.) — функциональные строки, сопоставляются с вводом; не переводим.
        "apple": ["apple", "apple_2", "pingguo", "苹果"],  # 苹果 — яблоко
        "apple_rack": [
            "apple rack",
            "fruit rack",
            "fruit basket",
            "apple basket",
            "有苹果的架子",  # 有苹果的架子 — полка с яблоками
            "放苹果的架子",  # 放苹果的架子 — полка, куда кладут яблоко
            "苹果架",  # 苹果架 — полка для яблок
            "苹果篮",  # 苹果篮 — корзина с яблоками
            "果篮",  # 果篮 — фруктовая корзина
            "果架",  # 果架 — фруктовая полка
        ],
        "banana": ["banana", "bananas", "xiangjiao", "香蕉"],  # 香蕉 — банан
        "chocolate": ["chocolate", "choco", "chocolate bar", "巧克力", "巧克力棒", "巧克力块"],  # 巧克力 — шоколад, 巧克力棒 — шоколадный батончик, 巧克力块 — кусочек шоколада
        "sponge": ["sponge", "海绵", "海棉", "海绵块", "海绵垫", "海绵片"],  # 海绵 — губка, 海棉 — губка, 海绵块 — губка-брусок, 海绵垫 — губковая подкладка, 海绵片 — губковый лист
        "duck": ["duck", "yellow duck", "toy duck", "duckie", "鸭子"],  # 鸭子 — утка
        "hammer": ["hammer", "锤子"],  # 锤子 — молоток
        "knife": ["knife", "刀", "小刀"],  # 刀 — нож, 小刀 — ножик
        "plate": ["plate", "dish", "saucer", "盘子", "盘里", "盘中", "盘上"],  # 盘子 — тарелка, 盘里 — на тарелке, 盘中 — в тарелке, 盘上 — на тарелке
        "sponge_rack": ["sponge rack", "sponge shelf", "海绵架", "海绵货架", "海绵收纳架", "海绵架子", "海绵位"],  # 海绵架 — стеллаж для губок, 海绵货架 — стеллаж для губок, 海绵收纳架 — органайзер для губок, 海绵架子 — полка для губок, 海绵位 — место для губки
        "shelf": ["shelf", "rack", "架子", "搁架"],  # 架子 — полка, 搁架 — полка
        "glass_cup": ["glass cup", "glass", "cup", "玻璃杯", "杯子"],  # 玻璃杯 — стакан, 杯子 — чашка
        "flower_vase": ["flower vase", "vase", "花瓶"],  # 花瓶 — ваза
    }
    hits: list[tuple[int, str]] = []
    for canonical, variants in aliases.items():
        for variant in variants:
            idx = text.find(variant)
            if idx >= 0:
                hits.append((idx, canonical))
                break
    hits.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, canonical in hits:
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def _infer_relation(text: str) -> str:
    for relation, variants in {
        "in": [
            "in",
            "into",
            "inside",
            "里面",  # 里面 — внутри
            "里",  # 里 — внутрь
            "放进",  # 放进 — положить внутрь
            "放入",  # 放入 — положить в
            "放到里",  # 放到里 — положить внутрь
            "放到里面",  # 放到里面 — положить внутрь
            "放回",  # 放回 — вернуть
            "放回去",  # 放回去 — вернуть обратно
            "归位",  # 归位 — вернуть на место
            "收纳",  # 收纳 — убрать
            "整理放回",  # 整理放回 — убрать на место
        ],
        "on_top_of": ["on top of", "on", "upon", "放到上", "放到盘上", "放到桌面上"],  # 放到上 — положить сверху, 放到盘上 — положить на тарелку, 放到桌面上 — положить на стол
        "next_to": ["next to", "beside", "near", "旁边", "附近"],  # 旁边 — рядом, 附近 — рядом
        "left_of": ["left of", "to the left of", "左边"],  # 左边 — слева
        "right_of": ["right of", "to the right of", "右边"],  # 右边 — справа
        "in_front_of": ["in front of", "front of", "前面"],  # 前面 — спереди
        "behind": ["behind", "at the back of", "后面"],  # 后面 — сзади
    }.items():
        if any(variant in text for variant in variants):
            return relation
    return "place"


def _parse_task(command: str) -> dict[str, Any]:
    text = _normalize_text(command)
    if not text:
        return {"type": "idle", "source": None, "destination": None, "relation": None, "all_items": False}
    if text in {"exit", "quit"}:
        return {"type": "interrupted", "source": None, "destination": None, "relation": None, "all_items": False}
    if any(keyword in text for keyword in ["teleop", "manual", "keyboard"]):
        return {"type": "teleop", "source": None, "destination": None, "relation": None, "all_items": False}
    if any(keyword in text for keyword in ["dance", "wave"]):
        return {"type": "dance", "source": None, "destination": None, "relation": None, "all_items": False}
    if any(keyword in text for keyword in ["drop", "throw", "扔", "抛", "摔", "丢"]):  # 扔 — бросить, 抛 — бросить, 摔 — уронить, 丢 — выбросить
        objects = _extract_objects(text)
        return {
            "type": "drop",
            "source": objects[0] if objects else "object",
            "destination": None,
            "relation": "drop",
            "all_items": False,
        }

    objects = _extract_objects(text)
    relation = _infer_relation(text)
    place_keywords = [
        "place",
        "put",
        "move",
        "relocate",
        "放回",  # 放回 — вернуть
        "放回去",  # 放回去 — вернуть обратно
        "放入",  # 放入 — положить в
        "放进",  # 放进 — положить внутрь
        "放置",  # 放置 — разместить
        "放到",  # 放到 — положить на
        "放在",  # 放在 — положить на
        "归位",  # 归位 — вернуть на место
        "整理放回",  # 整理放回 — убрать на место
        "收纳",  # 收纳 — убрать
    ]
    all_items = any(keyword in text for keyword in ["全部", "所有", "全都", "都", "一起", "批量"])  # 全部 — все, 所有 — все, 全都 — все, 都 — все, 一起 — вместе, 批量 — пакетом
    if len(objects) >= 2 or any(keyword in text for keyword in place_keywords):
        source = objects[0] if objects else "object"
        destination = objects[1] if len(objects) > 1 else "target"
        if source == "apple" and destination in {"target", "shelf", "apple"}:
            if any(
                keyword in text
                for keyword in ["有苹果的架子", "放苹果的架子", "苹果架", "苹果篮", "果篮", "果架"]  # 有苹果的架子 — полка с яблоками, 放苹果的架子 — полка, куда кладут яблоко, 苹果架 — полка для яблок, 苹果篮 — корзина с яблоками, 果篮 — фруктовая корзина, 果架 — фруктовая полка
            ):
                destination = "apple_rack"
        if source == "apple" and destination == "apple_rack":
            relation = "in"
        if source == "sponge" and destination in {"target", "shelf"}:
            destination = "sponge_rack"
        if source == "sponge" and destination == "sponge":
            destination = "sponge_rack"
        if source == "sponge" and destination == "sponge_rack":
            relation = "in"
        return {
            "type": "pick_place",
            "source": source,
            "destination": destination,
            "relation": relation,
            "all_items": all_items,
        }
    return {
        "type": "grasp",
        "source": objects[0] if objects else "object",
        "destination": None,
        "relation": None,
        "all_items": False,
    }


def _build_trace(task: dict[str, Any], blocked_reason: str | None = None) -> list[dict[str, Any]]:
    task_type = task["type"]
    library = {
        "grasp": [
            "Ввод получен",
            "Разбор команды",
            "Планирование задачи",
            "Съёмка сцены",
            "Оценка цели захвата",
            "Решение IK",
            "Выполнение действия",
            "Результат",
        ],
        "pick_place": [
            "Ввод получен",
            "Разбор команды",
            "Планирование задачи",
            "Сегментация цели",
            "Оценка цели захвата",
            "Оценка цели размещения",
            "Решение IK",
            "Выполнение действия",
            "Результат",
        ],
        "teleop": ["Ввод получен", "Разбор команды", "Планирование задачи", "Переключение телеуправления", "Результат"],
        "dance": ["Ввод получен", "Разбор команды", "Планирование задачи", "Воспроизведение траектории", "Результат"],
        "drop": ["Ввод получен", "Разбор команды", "Планирование задачи", "Планирование броска", "Результат"],
        "interrupted": ["Ввод получен", "Планирование задачи", "Завершение сессии"],
        "blocked": ["Ввод получен", "Проверка политики", "Результат"],
    }
    steps = library["blocked"] if blocked_reason else library.get(task_type, library["grasp"])
    trace: list[dict[str, Any]] = []
    for index, name in enumerate(steps):
        status = "done"
        if index == 1:
            status = "running"
        if index >= 2:
            status = "pending"
        if blocked_reason:
            status = "failed" if index == 1 else "pending"
        trace.append(
            {
                "name": name,
                "status": status,
                "detail": _trace_detail(name, task, blocked_reason),
            }
        )
    return trace


def _trace_detail(step: str, task: dict[str, Any], blocked_reason: str | None) -> str:
    parts: list[str] = []
    if task.get("source"):
        parts.append(f"source={task['source']}")
    if task.get("destination"):
        parts.append(f"destination={task['destination']}")
    if task.get("relation"):
        parts.append(f"relation={task['relation']}")
    if blocked_reason and step == "Проверка политики":
        parts.append(f"blocked by {blocked_reason}")
    if step == "Результат":
        parts.append("Ожидание сводки выполнения")
    return " | ".join(parts) if parts else "queued"


def _fake_preview(task: dict[str, Any]) -> dict[str, Any]:
    if task["type"] == "pick_place":
        return {
            "boxes": [
                {"label": task["source"], "x": 16, "y": 56, "w": 18, "h": 14},
                {"label": task["destination"], "x": 63, "y": 24, "w": 20, "h": 16},
            ]
        }
    if task["type"] == "grasp":
        return {"boxes": [{"label": task["source"], "x": 54, "y": 40, "w": 18, "h": 16}]}
    if task["type"] == "teleop":
        return {"boxes": [{"label": "teleop target", "x": 42, "y": 36, "w": 18, "h": 18}]}
    return {"boxes": []}


def _extract_block_reason(command: str) -> str:
    lower = _normalize_text(command)
    for token in ["self-harm", "suicide", "weapon", "bomb"]:
        if token in lower:
            return token
    return ""


def _encode_frame_png(frame: np.ndarray) -> bytes:
    # Encode directly from RGB so browser previews match the simulator output.
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def _task_label(task_type: str) -> str:
    return {
        "grasp": "Задача захвата",
        "pick_place": "Задача размещения",
        "drop": "Задача броска",
        "teleop": "Задача телеуправления",
        "dance": "Демонстрация танца",
        "interrupted": "Сессия",
    }.get(task_type, task_type)


def _stage_label(stage: str) -> str:
    return {
        "home": "Возврат в исходное положение",
        "hub": "Перенос и подъём",
        "hover_pick": "Позиционирование над целью",
        "pregrasp": "Предзахват",
        "grasp": "Захват",
        "grasp_close": "Смыкание захвата",
        "lift": "Подъём цели",
        "hover_place": "Позиционирование над местом",
        "preplace": "Предразмещение",
        "place": "Размещение",
        "release": "Освобождение цели",
        "retreat": "Отход назад",
        "hub_return": "Возврат в промежуточное положение",
        "reset": "Возврат в исходное положение",
        "start": "Начало выполнения",
        "capture": "Съёмка изображения",
        "segment": "Семантическая сегментация",
        "infer": "Оценка захвата",
        "prepare": "Подготовка задачи",
        "dance": "Воспроизведение траектории",
        "teleop": "Подготовка телеуправления",
        "drop": "Планирование броска",
    }.get(stage, stage)


class MuJoCoCommandRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._env: UR5GraspEnv | None = None

    def _temp_image_path(self, session: "SessionRecord", stem: str) -> Path:
        temp_dir = ROOT_DIR / "temp" / "images"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"{session.session_id}_{stem}.png"

    def close(self) -> None:
        with self._lock:
            if self._env is not None:
                try:
                    self._env.close()
                finally:
                    self._env = None

    def _ensure_env(self) -> UR5GraspEnv:
        if self._env is None:
            self._env = UR5GraspEnv()
            self._env.reset()
        return self._env

    def _capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        env = self._ensure_env()
        imgs = env.render()
        color = np.array(imgs["img"], copy=True)
        depth = np.array(imgs["depth"], copy=True)
        return color, depth

    def _refresh_trace(
        self,
        session: "SessionRecord",
        *,
        stage: str,
        final: bool = False,
        failed: bool = False,
    ) -> None:
        task_type = session.task.get("type", "grasp")
        stage_index_map = {
            "grasp": {
                "capture": 3,
                "segment": 4,
                "infer": 4,
                "prepare": 3,
                "home": 6,
                "hub": 6,
                "hover_pick": 6,
                "pregrasp": 6,
                "grasp": 6,
                "grasp_close": 6,
                "lift": 6,
                "hover_place": 6,
                "preplace": 6,
                "place": 6,
                "release": 6,
                "retreat": 6,
                "hub_return": 6,
                "reset": 6,
                "result": 7,
                "error": 6,
            },
            "pick_place": {
                "capture": 3,
                "segment": 3,
                "infer": 4,
                "prepare": 3,
                "home": 7,
                "hub": 7,
                "hover_pick": 7,
                "pregrasp": 7,
                "grasp": 7,
                "grasp_close": 7,
                "lift": 7,
                "hover_place": 7,
                "preplace": 7,
                "place": 7,
                "release": 7,
                "retreat": 7,
                "hub_return": 7,
                "reset": 7,
                "result": 8,
                "error": 7,
            },
            "drop": {
                "capture": 3,
                "segment": 3,
                "infer": 3,
                "prepare": 3,
                "home": 3,
                "hub": 3,
                "hover_pick": 3,
                "pregrasp": 3,
                "grasp": 3,
                "grasp_close": 3,
                "lift": 3,
                "hover_place": 3,
                "preplace": 3,
                "place": 3,
                "release": 3,
                "retreat": 3,
                "hub_return": 3,
                "reset": 3,
                "drop": 3,
                "result": 4,
                "error": 3,
            },
            "teleop": {
                "capture": 3,
                "prepare": 3,
                "teleop": 3,
                "home": 3,
                "result": 4,
                "error": 3,
            },
            "dance": {
                "capture": 3,
                "prepare": 3,
                "dance": 3,
                "home": 3,
                "result": 4,
                "error": 3,
            },
        }
        indices = stage_index_map.get(task_type, stage_index_map["grasp"])
        progress_index = indices.get(stage, indices.get("infer", 0))
        if final:
            progress_index = len(session.trace)
        refreshed = []
        for index, item in enumerate(session.trace):
            if final:
                status = "failed" if failed and index == progress_index else "done"
            elif index < progress_index:
                status = "done"
            elif index == progress_index:
                status = "failed" if failed else "running"
            else:
                status = "pending"
            refreshed.append({**item, "status": status})
        session.trace = refreshed

    def _touch_preview(
        self,
        session: "SessionRecord",
        *,
        stage: str,
        note: str = "",
        boxes: list[dict[str, Any]] | None = None,
    ) -> None:
        with session.condition:
            self._refresh_trace(session, stage=stage)
            session.current_step = _stage_label(stage)
            env = self._ensure_env()
            frame = env.render()["img"]
            session.frame_jpeg = _encode_frame_png(frame)
            session.preview = {
                "image_url": f"/api/session/{session.session_id}/frame?rev={session.revision}",
                "stage": stage,
                "stage_label": _stage_label(stage),
                "note": note or _stage_label(stage),
                "boxes": boxes or [],
            }
            session.updated_at = _now_iso()
            session.revision += 1
            session.condition.notify_all()

    def _set_running(self, session: "SessionRecord", stage: str, result: str | None = None) -> None:
        session.current_step = _stage_label(stage)
        if result is not None:
            session.result = result
        session.updated_at = _now_iso()
        session.revision += 1
        session.condition.notify_all()

    def _run_dance(self, session: "SessionRecord") -> dict[str, Any]:
        env = self._ensure_env()
        from grasp_process import _move_joint_waypoint

        robot = env.robot
        action = np.zeros(7, dtype=np.float64)
        q_start = np.array(robot.get_joint(), dtype=np.float64)
        waypoints = [
            np.array([0.35, -0.70, 1.45, -1.05, -1.25, 0.05], dtype=np.float64),
            np.array([-0.35, -0.70, 1.45, -1.05, -1.25, -0.05], dtype=np.float64),
            np.array([0.55, -1.05, 1.25, -0.75, -1.45, 0.75], dtype=np.float64),
            np.array([-0.55, -1.05, 1.25, -0.75, -1.45, -0.75], dtype=np.float64),
            q_start,
        ]
        for index, q_target in enumerate(waypoints):
            _move_joint_waypoint(
                env,
                robot,
                action,
                q_target,
                0.75,
                frame_callback=lambda _stage, _: self._touch_preview(session, stage="dance", note="Демонстрация танца"),
                stage_name=f"dance_{index}",
            )
        return {"status": "ok", "task": "dance"}

    def _run_teleop(self, session: "SessionRecord") -> dict[str, Any]:
        env = self._ensure_env()
        from grasp_process import _move_joint_waypoint

        robot = env.robot
        action = np.zeros(7, dtype=np.float64)
        q_home = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=np.float64)
        self._touch_preview(session, stage="teleop", note="Подготовка телеуправления")
        _move_joint_waypoint(
            env,
            robot,
            action,
            q_home,
            0.9,
            frame_callback=lambda _stage, _: self._touch_preview(session, stage="teleop", note="Подготовка телеуправления"),
            stage_name="teleop",
        )
        return {"status": "ok", "task": "teleop"}

    def _run_single_grasp_like(
        self,
        session: "SessionRecord",
        task: dict[str, Any],
        command: str,
        *,
        source_name_override: str | None = None,
        destination_name_override: str | None = None,
        place_target_override: np.ndarray | None = None,
        note_prefix: str = "",
    ) -> dict[str, Any]:
        env = self._ensure_env()
        color_img, depth_img = self._capture_rgbd()
        capture_note = "Съёмка кадра симуляционной камеры"
        if note_prefix:
            capture_note = f"{note_prefix}{capture_note}"
        self._touch_preview(session, stage="capture", note=capture_note)

        source_name = source_name_override or task.get("source")
        destination_name = destination_name_override or task.get("destination")
        relation = task.get("relation") or "place"

        chocolate_like = isinstance(source_name, str) and source_name in {
            "chocolate",
            "chocolate_bar",
            "snickers",
        }
        sponge_like = isinstance(source_name, str) and source_name == "sponge"
        source_bbox = None
        source_command = str(command)
        source_label = str(source_name) if isinstance(source_name, str) and source_name else None
        if chocolate_like:
            source_command = f"{command}. Пожалуйста, в первую очередь выберите шоколадный батончик с надписью SNICKERS на упаковке на изображении."
            source_label = "SNICKERS chocolate bar"
        elif sponge_like:
            source_command = f"{command}. Пожалуйста, в первую очередь выберите губку, отмеченную красным кружком на изображении."
            source_label = "sponge"
            source_bbox = estimate_body_image_bbox(env, source_name, color_img.shape)
        elif isinstance(source_name, str) and source_name:
            source_bbox = estimate_body_image_bbox(env, source_name, color_img.shape)

        source_mask = segment_image(
            color_img,
            output_mask=str(self._temp_image_path(session, "mask_source")),
            command_text=source_command,
            bbox_override=source_bbox,
            label_override=source_label,
        )
        source_segment_note = "Сегментация исходной цели завершена"
        if note_prefix:
            source_segment_note = f"{note_prefix}{source_segment_note}"
        self._touch_preview(session, stage="segment", note=source_segment_note)

        grasp_target_world = None
        if isinstance(source_name, str) and source_name:
            try:
                grasp_target_world, _ = estimate_direct_grasp_target_world(
                    env,
                    depth_img,
                    source_mask,
                    source_name=source_name,
                )
            except Exception:
                grasp_target_world = None

        place_target_world = None if place_target_override is None else np.array(place_target_override, dtype=np.float64)
        place_mode = None
        if task["type"] == "pick_place":
            if destination_name == "plate" and relation == "in":
                relation = "on_top_of"
            if destination_name == "plate":
                place_mode = "drop_above_plate"
            if place_target_world is None:
                destination_command = str(destination_name or "")
                destination_label = (
                    str(destination_name) if isinstance(destination_name, str) and destination_name else None
                )
                destination_bbox = (
                    estimate_body_image_bbox(env, destination_name, color_img.shape)
                    if isinstance(destination_name, str) and destination_name
                    else None
                )
                if isinstance(destination_name, str) and destination_name == "sponge_rack":
                    destination_command = f"{command}. Пожалуйста, в первую очередь выберите место размещения губки на стойке, отмеченное зелёным кругом."
                    destination_label = "sponge rack"
                destination_mask = segment_image(
                    color_img,
                    output_mask=str(self._temp_image_path(session, "mask_destination")),
                    command_text=destination_command,
                    bbox_override=destination_bbox,
                    label_override=destination_label,
                )
                destination_segment_note = "Сегментация цели размещения завершена"
                if note_prefix:
                    destination_segment_note = f"{note_prefix}{destination_segment_note}"
                self._touch_preview(session, stage="segment", note=destination_segment_note)
                place_target_world = estimate_place_target_world(
                    env,
                    depth_img,
                    destination_mask,
                    source_mask=source_mask,
                    relation=relation,
                    source_name=source_name if isinstance(source_name, str) else None,
                    destination_name=destination_name if isinstance(destination_name, str) else None,
                )
        elif task["type"] == "drop":
            if grasp_target_world is not None:
                place_target_world = np.array(grasp_target_world, dtype=np.float64).copy()
                place_target_world[2] = max(TABLE_SURFACE_Z + 0.03, float(place_target_world[2]))
            place_mode = "drop_above"

        infer_note = "Оценка захвата и размещения завершена"
        if note_prefix:
            infer_note = f"{note_prefix}{infer_note}"
        self._touch_preview(session, stage="infer", note=infer_note)
        gg = run_grasp_inference(
            color_img,
            depth_img,
            source_mask,
            camera_fovy_deg=getattr(env, "camera_fovy_deg", 45.0),
        )
        infer_grasp_note = "Вывод кандидатов захвата завершён"
        if note_prefix:
            infer_grasp_note = f"{note_prefix}{infer_grasp_note}"
        self._touch_preview(session, stage="infer", note=infer_grasp_note)

        execute_grasp(
            env,
            gg,
            place_target_world=place_target_world,
            grasp_target_world=grasp_target_world,
            source_name=source_name if isinstance(source_name, str) else None,
            place_mode=place_mode,
            frame_callback=lambda stage, _: self._touch_preview(
                session,
                stage=stage or "motion",
                note=f"{note_prefix}выполнение в реальном MuJoCo" if note_prefix else "Выполнение в реальном MuJoCo",
            ),
        )
        return {
            "status": "ok",
            "task": task["type"],
            "source": source_name,
            "destination": destination_name,
            "relation": relation,
            "grasp_count": len(gg),
        }

    def _run_grasp_like(self, session: "SessionRecord", task: dict[str, Any], command: str) -> dict[str, Any]:
        env = self._ensure_env()
        if (
            task.get("type") == "pick_place"
            and task.get("source") == "apple"
            and task.get("destination") == "apple_rack"
        ):
            slot_world = get_apple_rack_slot_world(env, slot_index=0)
            if slot_world is None:
                raise RuntimeError("Не удалось определить положение корзины с яблоками.")
            return self._run_single_grasp_like(
                session,
                task,
                command,
                source_name_override="apple",
                destination_name_override="apple_rack",
                place_target_override=slot_world,
            )
        if (
            task.get("type") == "pick_place"
            and task.get("source") == "sponge"
            and task.get("destination") == "sponge_rack"
            and task.get("all_items")
        ):
            source_names = get_available_table_sponge_bodies(env)
            if not source_names:
                raise RuntimeError("На столе нет губок для уборки.")
            results = []
            for index, source_name in enumerate(source_names):
                slot_world = get_sponge_rack_slot_world(env, slot_index=index)
                if slot_world is None:
                    raise RuntimeError("Не удалось определить ячейку стойки для губок.")
                results.append(
                    self._run_single_grasp_like(
                        session,
                        task,
                        command,
                        source_name_override=source_name,
                        destination_name_override="sponge_rack",
                        place_target_override=slot_world,
                        note_prefix=f"Губка {index + 1}:",
                    )
                )
                with session.condition:
                    session.logs.append(
                        f"[{_now_iso()}] INFO: sponge batch item {index + 1}/{len(source_names)} -> {source_name}"
                    )
                    session.condition.notify_all()
            return {
                "status": "ok",
                "task": task["type"],
                "source": "sponge",
                "destination": "sponge_rack",
                "relation": task.get("relation"),
                "grasp_count": sum(int(item.get("grasp_count", 0)) for item in results),
                "batch_count": len(results),
            }
        return self._run_single_grasp_like(session, task, command)

    def run(self, session: "SessionRecord") -> None:
        task = session.task or {"type": "grasp", "source": "object", "destination": None, "relation": None}
        with self._lock:
            try:
                if task["type"] == "dance":
                    result = self._run_dance(session)
                elif task["type"] == "teleop":
                    result = self._run_teleop(session)
                else:
                    result = self._run_grasp_like(session, task, session.command)

                inventory_event = record_task_success_effects(
                    task=task,
                    command=session.command,
                    session_id=session.session_id,
                )

                with session.condition:
                    session.status = "success"
                    session.current_step = "Результат"
                    session.result = {
                        "grasp": "Задача захвата выполнена на реальном бэкенде.",
                        "pick_place": "Задача размещения выполнена на реальном бэкенде.",
                        "drop": "Симуляция броска выполнена на реальном бэкенде.",
                        "teleop": "Демонстрация телеуправления готова.",
                        "dance": "Демонстрация танца завершена.",
                    }.get(task["type"], "Задача выполнена на реальном бэкенде.")
                    session.inventory_event = inventory_event or {}
                    session.inventory = get_inventory_store().snapshot()
                    session.logs.append(f"[{_now_iso()}] OK: выполнение в реальном MuJoCo завершено")
                    if inventory_event:
                        session.logs.append(
                            f"[{_now_iso()}] INFO: inventory {inventory_event['sku']} -> {inventory_event['remaining']}"
                        )
                        if inventory_event.get("alert_sent"):
                            session.logs.append(
                                f"[{_now_iso()}] WARN: low stock alert emitted for {inventory_event['sku']}"
                            )
                    self._refresh_trace(session, stage="result", final=True)
                    self._touch_preview(session, stage="result", note=session.result)
                    session.condition.notify_all()
            except Exception as exc:
                with session.condition:
                    session.status = "failure"
                    session.current_step = "Результат"
                    session.result = f"Ошибка выполнения: {exc}"
                    session.inventory = get_inventory_store().snapshot()
                    session.logs.append(f"[{_now_iso()}] ERROR: {exc}")
                    self._refresh_trace(session, stage="error", final=True, failed=True)
                    self._touch_preview(session, stage="error", note=str(exc))
                    session.condition.notify_all()


BACKEND = MuJoCoCommandRunner()


def _step_delay(task_type: str, index: int) -> float:
    plan = {
        "grasp": [0.28, 0.34, 0.30, 0.34, 0.38, 0.34, 0.40, 0.32],
        "pick_place": [0.28, 0.34, 0.30, 0.34, 0.36, 0.36, 0.34, 0.40, 0.34],
        "teleop": [0.24, 0.28, 0.28, 0.36, 0.32],
        "dance": [0.24, 0.28, 0.30, 0.42, 0.34],
        "interrupted": [0.20, 0.20, 0.20],
    }
    steps = plan.get(task_type, plan["grasp"])
    return steps[min(index, len(steps) - 1)]


def _progress_trace(trace: list[dict[str, Any]], active_index: int | None, terminal: bool = False) -> list[dict[str, Any]]:
    next_trace: list[dict[str, Any]] = []
    for index, item in enumerate(trace):
        if terminal:
            status = "failed" if item["status"] == "failed" else "done"
        elif active_index is None:
            status = item["status"]
        elif index < active_index:
            status = "done"
        elif index == active_index:
            status = "running"
        else:
            status = "pending"
        next_trace.append({**item, "status": status})
    return next_trace


def _touch_session(session: "SessionRecord") -> None:
    session.updated_at = _now_iso()
    session.revision += 1
    session.condition.notify_all()


def _run_session(session_id: str) -> None:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        return

    with session.condition:
        if session.status != "running":
            return
        session.logs.append(f"[{_now_iso()}] INFO: переход к выполнению в реальном MuJoCo")
        session.revision += 1
        session.condition.notify_all()

    BACKEND.run(session)


@dataclass
class SessionRecord:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    revision: int = 0
    command: str = ""
    task: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    status: str = "idle"
    current_step: str = "Waiting"
    result: str = "No result yet."
    preview: dict[str, Any] = field(default_factory=dict)
    inventory: dict[str, Any] = field(default_factory=dict)
    inventory_event: dict[str, Any] = field(default_factory=dict)
    frame_jpeg: bytes = b""
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
            "command": self.command,
            "parsed": self.task,
            "trace": self.trace,
            "logs": self.logs,
            "status": self.status,
            "current_step": self.current_step,
            "result": self.result,
            "preview": self.preview,
            "inventory": self.inventory,
            "inventory_event": self.inventory_event,
            "preview_url": self.preview.get("image_url", ""),
        }


sessions: dict[str, SessionRecord] = {}
sessions_lock = threading.Lock()
app = FastAPI(title="OpenClaw Web Frontend")
INVENTORY = get_inventory_store()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/inventory")
def get_inventory() -> dict[str, Any]:
    return INVENTORY.snapshot()


def _inventory_order_html(order: dict[str, Any], snapshot: dict[str, Any]) -> Response:
    item = next((entry for entry in snapshot.get("items", []) if entry.get("key") == order.get("sku")), {})
    title = f"{order.get('label') or order.get('sku') or 'Товар'} — заказ оформлен"
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: #0f1720;
      color: #eff6ff;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }}
    .card {{
      width: min(640px, calc(100vw - 32px));
      background: #17212c;
      border: 1px solid rgba(124, 158, 190, 0.25);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }}
    h1 {{ margin: 0 0 12px; font-size: 1.6rem; }}
    p, li {{ color: #c9d7e6; line-height: 1.6; }}
    .meta {{ display: grid; gap: 8px; margin: 18px 0; }}
    a.button {{
      display: inline-block;
      margin-top: 14px;
      padding: 10px 16px;
      border-radius: 999px;
      background: linear-gradient(135deg, #dff3ff, #b7d7f5);
      color: #0f1720;
      text-decoration: none;
      font-weight: 700;
    }}
    code {{ background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 8px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <p>Заказ записан. Можно закрыть эту страницу или вернуться в Feishu, чтобы продолжить просмотр уведомлений.</p>
    <div class="meta">
      <div>SKU: <code>{order.get("sku")}</code></div>
      <div>Количество: <code>{order.get("quantity")}</code></div>
      <div>Статус: <code>{order.get("source")}</code></div>
      <div>Остаток на складе: <code>{item.get("count", "-")}</code></div>
    </div>
    <p>Снимок склада обновлён.</p>
    <a class="button" href="/">Вернуться в консоль</a>
  </div>
</body>
</html>"""
    return Response(content=html, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


@app.get("/api/inventory/order")
def place_inventory_order_get(
    sku: str,
    quantity: int = 1,
    source: str = "feishu",
    token: str | None = None,
) -> Response:
    order = INVENTORY.record_order(
        sku=sku,
        quantity=quantity,
        source=source,
        token=token,
    )
    snapshot = INVENTORY.snapshot()
    notify_robot_backend(
        {
            "kind": "inventory_order",
            "order": order,
            "inventory": snapshot,
        }
    )
    return _inventory_order_html(order, snapshot)


@app.post("/api/inventory/order")
def place_inventory_order_post(payload: dict[str, Any]) -> dict[str, Any]:
    sku = str(payload.get("sku") or "").strip()
    if not sku:
        raise HTTPException(status_code=400, detail="sku is required")
    quantity = int(payload.get("quantity") or 1)
    source = str(payload.get("source") or "manual").strip() or "manual"
    token = str(payload.get("token") or payload.get("order_token") or "").strip() or None
    order = INVENTORY.record_order(
        sku=sku,
        quantity=quantity,
        source=source,
        token=token,
    )
    snapshot = INVENTORY.snapshot()
    notify_robot_backend(
        {
            "kind": "inventory_order",
            "order": order,
            "inventory": snapshot,
        }
    )
    return {
        "order": order,
        "inventory": snapshot,
    }


@app.post("/api/inventory/replenish")
def replenish_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    sku = str(payload.get("sku") or "").strip()
    if not sku:
        raise HTTPException(status_code=400, detail="sku is required")
    quantity = int(payload.get("quantity") or 0)
    event = INVENTORY.replenish(sku=sku, quantity=quantity)
    snapshot = INVENTORY.snapshot()
    notify_robot_backend(
        {
            "kind": "inventory_replenish",
            "event": event,
            "inventory": snapshot,
        }
    )
    return {
        "event": event,
        "inventory": snapshot,
    }


@app.post("/api/command")
def submit_command(payload: dict[str, Any]) -> dict[str, Any]:
    command = _normalize_text(str(payload.get("command", "")))
    if not command:
        raise HTTPException(status_code=400, detail="command is required")

    session_id = str(payload.get("session_id") or payload.get("sessionId") or "").strip()
    with sessions_lock:
        session = sessions.get(session_id) if session_id else None
        if session is None:
            session = SessionRecord()
            sessions[session.session_id] = session

    task = _parse_task(command)
    blocked_reason = _extract_block_reason(command)
    trace = _build_trace(task, blocked_reason=blocked_reason or None)

    with session.condition:
        session.command = command
        session.task = task
        session.trace = trace
        session.preview = _fake_preview(task)
        session.inventory = get_inventory_store().snapshot()
        session.inventory_event = {}
        session.logs = [
            f"[{_now_iso()}] INFO: command accepted: {command}",
            f"[{_now_iso()}] INFO: parsed task: {task['type']}",
        ]
        session.updated_at = _now_iso()
        session.revision += 1

        if blocked_reason:
            session.status = "failure"
            session.current_step = "Проверка политики"
            session.result = f"Заблокировано ключевым словом политики: {blocked_reason}"
            session.logs.append(f"[{_now_iso()}] WARN: заблокировано ключевое слово политики: {blocked_reason}")
            session.trace = _progress_trace(session.trace, 1, terminal=True)
            session.revision += 1
            session.condition.notify_all()
        elif task["type"] == "interrupted":
            session.status = "interrupted"
            session.current_step = "Завершение сессии"
            session.result = "Сессия закрыта."
            session.logs.append(f"[{_now_iso()}] OK: сессия прервана пользователем")
            session.trace = _progress_trace(session.trace, None, terminal=True)
            session.revision += 1
            session.condition.notify_all()
        else:
            session.status = "running"
            session.current_step = trace[0]["name"] if trace else "Running"
            session.result = "Команда выполняется..."
            session.trace = _progress_trace(session.trace, 0)
            session.revision += 1
            session.condition.notify_all()
            threading.Thread(target=_run_session, args=(session.session_id,), daemon=True).start()

    with sessions_lock:
        sessions[session.session_id] = session
    return session.payload()


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session.payload()


@app.get("/api/session/{session_id}/events")
def get_session_events(session_id: str) -> StreamingResponse:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    def stream():
        last_revision = -1
        yield f"data: {json.dumps(session.payload(), ensure_ascii=False)}\n\n"
        last_revision = session.revision
        while True:
            with session.condition:
                session.condition.wait(timeout=15)
                current_revision = session.revision
                current_status = session.status
                payload = session.payload()
            if current_revision != last_revision:
                last_revision = current_revision
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if current_status != "running":
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/session/{session_id}/frame")
def get_session_frame(session_id: str) -> Response:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if not session.frame_jpeg:
        raise HTTPException(status_code=404, detail="frame not ready")
    return Response(
        content=session.frame_jpeg,
        media_type="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/")
def root() -> FileResponse:
    return FileResponse(INDEX_FILE)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def _open_browser_later(url: str) -> None:
    time.sleep(1.0)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    _assert_frontend_present()
    os.environ.setdefault("OPENCLAW_RENDER_BACKEND", "glfw")
    host = os.getenv("OPENCLAW_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("OPENCLAW_WEB_PORT", "8000"))
    auto_open = os.getenv("OPENCLAW_WEB_AUTO_OPEN", "1").strip().lower() not in {"0", "false", "no"}
    url = f"http://{host}:{port}/"
    if auto_open:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
