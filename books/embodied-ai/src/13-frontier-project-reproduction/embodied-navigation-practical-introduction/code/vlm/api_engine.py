"""
Движок API VLM — совместимый с OpenAI интерфейс (DashScope / Qwen3-VL)
=========================================================
Вызывает удалённую большую модель для вывода через HTTP API.
"""

import cv2
import base64
import numpy as np
from openai import OpenAI

from .base_engine import BaseVLMDecisionEngine


def _image_to_base64(image_rgb: np.ndarray) -> str:
    """Массив RGB → строка JPEG base64 (длинная сторона масштабирована до 512 пикселей)."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if max(h, w) > 512:
        s = 512 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


class APIVLMDecisionEngine(BaseVLMDecisionEngine):
    """
    Движок решений API VLM.
    Наследует BaseVLMDecisionEngine, реализует _call_model() для вызова удалённого API.
    """

    def __init__(self, config):
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
        )
        self._model = config.API_MODEL_NAME

    def _call_model(self, system_prompt: str, user_text: str, key_images) -> str:
        # Строим мультимодальные parts
        parts = [{"type": "text", "text": user_text}]
        for i, img in enumerate(key_images[-4:]):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img)}"},
            })
            parts.append({"type": "text", "text": f"[Ключевой кадр обзора {i + 1}]"})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": parts},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
