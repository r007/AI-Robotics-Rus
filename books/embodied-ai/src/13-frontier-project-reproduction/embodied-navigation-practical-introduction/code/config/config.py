"""
Модуль конфигурации
========
Единое управление всеми параметрами для обоих режимов: local и api.
"""

import os


class Config:
    # ── Сцена ──────────────────────────────────────────────────
    SCENE_PATH = "./data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    TARGET_OBJECTS = ["chair", "couch", "tv"]

    # ── YOLO ──────────────────────────────────────────────────
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONF_THRESHOLD = 0.35

    # ── Локальный VLM (режим local) ─────────────────────────────────
    LOCAL_MODEL_NAME = "/home/robot/navigation/Qwen3-VL-4B-Instruct"
    MODEL_DTYPE = "auto"
    MODEL_DEVICE_MAP = "auto"
    MODEL_MAX_NEW_TOKENS = 1024
    MODEL_TEMPERATURE = 0.3
    MODEL_TOP_P = 0.9
    MODEL_TOP_K = 50

    # ── API VLM (режим api) ────────────────────────────────────
    API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your-api-key")
    API_MODEL_NAME = "qwen3-vl-8b-instruct"

    # ── Навигация ──────────────────────────────────────────────────
    MAX_STEPS = 500
    SPIN_STEPS = 12
    VISIBLE_RADIUS = 3.0
    MAP_RESOLUTION = 512
    AREA_THRESHOLD_M2 = 9
    TARGET_REACH_DIST = 1.5

    # ── Камера ──────────────────────────────────────────────────
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    HFOV = 90
    SENSOR_HEIGHT = 1.5

    # ── Вывод ──────────────────────────────────────────────────
    OUTPUT_VIDEO = "navigation_output.avi"
    OUTPUT_LOG = "vlm_reasoning_log.json"
    VIDEO_FPS = 4
