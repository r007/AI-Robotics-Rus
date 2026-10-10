#!/usr/bin/env python3
"""Add a visible, notebook-native run control cell to the three model notebooks."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = (
    "14_smolvla_end_to_end.ipynb",
    "15_pi0_end_to_end.ipynb",
    "16_act_end_to_end.ipynb",
)

RUN_CONTROL_MARKER = "# RUN_CONTROL_CELL"
RUN_CONTROL_MARKDOWN = (
    "## Панель управления: сначала выберите здесь запуск\n\n"
    "Эта ячейка — общий переключатель для трёх моделей. Длительное обучение включено по умолчанию, Smoke, оценка и protected-дообучение выключены; так Notebook по умолчанию показывает реальный процесс длительного обучения, но не запускает одновременно несколько задач обучения или оценки.\n\n"
    "Рекомендуемый порядок:\n\n"
    "1. В первый раз установите `RUN_SMOKE` в `True` и убедитесь, что окружение, данные, модель и один шаг обратного распространения работают.\n"
    "2. Если всё в порядке, установите `RUN_LONG_TRAIN` в `True` и запустите реальную ячейку обучения ниже: Notebook покажет живой `tqdm`, loss, время и checkpoint.\n"
    "3. После обучения установите `RUN_EVAL` в `True` и запустите ячейку строгой оценки: результат сведётся к успешности `x/y`, а при наличии условий рендеринга видео будет сохранено.\n"
    "4. Чтобы воспроизвести официальные защищённые результаты, установите `RUN_PROTECTED_TRAIN` в `True` и запустите соответствующую ячейку protected recipe. Это отдельная, явно обозначенная линия обучения, отличная от обычного учебного длительного обучения.\n\n"
    "Циклы обучения и оценки выполняются внутри Python-ядра Notebook, а не через `cat` статического лога и не передаются внешнему shell-скрипту."
)
RUN_CONTROL_CODE = (
    f"{RUN_CONTROL_MARKER}\n"
    "# Измените только четыре булевых значения ниже, затем выполняйте ячейки по порядку.\n"
    "RUN_SMOKE = False\n"
    "RUN_LONG_TRAIN = True\n"
    "RUN_EVAL = False\n"
    "RUN_PROTECTED_TRAIN = False\n\n"
    "print({\n"
    "    'RUN_SMOKE': RUN_SMOKE,\n"
    "    'RUN_LONG_TRAIN': RUN_LONG_TRAIN,\n"
    "    'RUN_EVAL': RUN_EVAL,\n"
    "    'RUN_PROTECTED_TRAIN': RUN_PROTECTED_TRAIN,\n"
    "})\n"
    "print('Длительное обучение обычно занимает от десятков минут до нескольких часов; оценка идёт по эпизодам в замкнутом цикле и занимает ещё дольше.')\n"
)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source + "\n"}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


for filename in NOTEBOOKS:
    path = NOTEBOOK_DIR / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    already_patched = any(RUN_CONTROL_MARKER in "".join(cell.get("source", [])) for cell in notebook["cells"])
    if not already_patched:
        notebook["cells"][3:3] = [
            markdown_cell(RUN_CONTROL_MARKDOWN),
            code_cell(RUN_CONTROL_CODE),
        ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if RUN_CONTROL_MARKER in source:
            source = source.replace("RUN_LONG_TRAIN = False", "RUN_LONG_TRAIN = True")
        source = source.replace(
            'RUN_LONG_TRAIN = env_flag("RUN_LONG_TRAIN")',
            'RUN_LONG_TRAIN = env_flag("RUN_LONG_TRAIN", True)',
        )
        if "默认全部关闭" in source:  # маркер «默认全部关闭» (всё выключено по умолчанию): функциональная строка, не переводить
            source = source.replace(
                "По умолчанию всё выключено, чтобы после открытия Notebook случайно не запустилось обучение на несколько часов.",
                "Длительное обучение включено по умолчанию, Smoke, оценка и protected-дообучение выключены; одновременно не запускается несколько задач.",
            )
        source = source.replace(
            'protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)',
            'protected_train_enabled = globals().get("RUN_PROTECTED_TRAIN", env_flag("RUN_PROTECTED_TRAIN", False))',
        )
        cell["source"] = source
        if "train_lerobot_config_in_notebook(" in source and not source.lstrip().startswith("def train_lerobot_config_in_notebook"):
            # Удаляем устаревший вывод «未启动» (не запущено). Следующее реальное выполнение должно создать новый вывод
            # the progress bar and metrics in this cell rather than showing a
            # misleading snapshot from the previous default.
            cell["execution_count"] = None
            cell["outputs"] = []
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("patched:", path, "(existing control updated)" if already_patched else "")
