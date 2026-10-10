"""
Пакет VLM
=======
Предоставляет единую фабричную функцию create_vlm_engine(), через которую главная функция создаёт движок по режиму.
"""

from .base_engine import BaseVLMDecisionEngine
from .local_engine import LocalVLMDecisionEngine
from .api_engine import APIVLMDecisionEngine


def create_vlm_engine(config, mode: str = "local") -> BaseVLMDecisionEngine:
    """
    Фабричная функция: возвращает подходящий движок решений VLM в зависимости от mode.

    Parameters
    ----------
    config : Config
    mode   : "local" или "api"

    Returns
    -------
    Экземпляр подкласса BaseVLMDecisionEngine
    """
    mode = mode.lower()
    if mode == "local":
        return LocalVLMDecisionEngine(config)
    elif mode == "api":
        return APIVLMDecisionEngine(config)
    else:
        raise ValueError(f"Неизвестный режим VLM: {mode!r}, выберите 'local' или 'api'")


__all__ = [
    "BaseVLMDecisionEngine",
    "LocalVLMDecisionEngine",
    "APIVLMDecisionEngine",
    "create_vlm_engine",
]
