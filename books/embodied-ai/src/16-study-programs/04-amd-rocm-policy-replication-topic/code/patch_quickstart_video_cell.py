#!/usr/bin/env python3
"""Idempotently add the zero-training video preview to Notebook 11."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = TOPIC_ROOT / "notebooks" / "11_mujoco_closed_loop_deploy.ipynb"
MARKER = "## Предварительный просмотр успешного выполнения без обучения: сначала поведение"
VIDEO_SOURCE = """
SUCCESS_PREVIEW = ASSET_DIR / "pnp_four_view_strict_success.mp4"
if not SUCCESS_PREVIEW.exists():
    raise FileNotFoundError(f"Не найдено видео успешного превью: {SUCCESS_PREVIEW}")

try:
    from IPython.display import Video
except ImportError:
    print("Сейчас это чистый Python-исполнитель; чтобы встроить воспроизведение видео, запустите эту ячейку в Jupyter.")
else:
    display(Video(filename=str(SUCCESS_PREVIEW), embed=True, html_attributes="controls muted"))

print("Строгое превью успеха:", SUCCESS_PREVIEW)
"""


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if MARKER in "".join(cell.get("source", [])):
            if index + 1 >= len(cells) or cells[index + 1].get("cell_type") != "code":
                raise RuntimeError("После пояснения к превью не найдена ячейка с кодом видео.")
            expected_source = VIDEO_SOURCE.strip() + "\n"
            current_source = "".join(cells[index + 1].get("source", []))
            if current_source == expected_source:
                print("Notebook 11 already contains the current zero-training preview.")
                return
            cells[index + 1]["source"] = expected_source
            cells[index + 1]["execution_count"] = None
            cells[index + 1]["outputs"] = []
            NOTEBOOK.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            print("Updated the existing zero-training preview cell.")
            return

    insert_at = 3
    cells[insert_at:insert_at] = [
        markdown_cell(
            """
## Предварительный просмотр успешного выполнения без обучения: сначала поведение

Эта ячейка не загружает модель, не запускает MuJoCo и не расходует бюджет обучения. Она показывает встроенное в репозиторий строгое успешное воспроизведение с четырьмя ракурсами, чтобы сначала проверить правильный порядок: подход, захват, подъём, перенос, освобождение и устойчивое размещение. Видео получено на фиксированном окружении с `pi0 + visual/history learned head`; это не raw pi0 и не говорит о генерализации по случайным положениям.
"""
        ),
        code_cell(VIDEO_SOURCE),
    ]
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Patched {NOTEBOOK}")


if __name__ == "__main__":
    main()
