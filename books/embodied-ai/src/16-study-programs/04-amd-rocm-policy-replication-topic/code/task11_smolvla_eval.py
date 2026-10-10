"""Точка входа для скрипта SmolVLA с четырьмя seed (Task 11).

Исходный PR повторно использовал логику, выполняя фиксированные номера ячеек Notebook, но после добавления новых ячеек такой подход легко ломался.
Теперь все вызывают run_closed_loop.py, поэтому пути, строгое определение успеха,
кодирование видео и поля результатов поддерживаются в одном экземпляре.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("POLICY_TYPE", "smolvla")
os.environ.setdefault("EVAL_SEEDS", "1000,1001,1002,1003")
os.environ.setdefault(
    "MODEL_RUN_DIR",
    str(
        Path(os.environ.get("MODEL_ROOT", TOPIC_ROOT / "checkpoints"))
        / "smolvla_weighted_000500"
    ),
)

runpy.run_path(
    str(Path(__file__).with_name("run_closed_loop.py")),
    run_name="__main__",
)
