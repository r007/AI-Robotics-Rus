#!/usr/bin/env python3
"""Append idempotent post-long-training eval cells to model notebooks.

These cells are deliberately placed at the end of each notebook so a finished
long-training transcript is not overwritten. Execute them with:

  RUN_POST_LONG_EVAL=1 EVAL_USE_LONG_TRAIN=1 \
  python code/execute_tutorial_notebooks.py \
    --prepare-marker "from pathlib import Path" \
    --prepare-marker "def md_table(headers, rows)" \
    --prepare-marker "<model config marker>" \
    --record-marker "POST_LONG_EVAL_CELL" notebooks/...
"""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
MARKER = "POST_LONG_EVAL_CELL"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def append_cells(notebook_name: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell
        for cell in notebook.get("cells", [])
        if MARKER not in ("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", "")))
    ]
    notebook["cells"].extend(cells)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {path}")


def smolvla_cells() -> list[dict]:
    return [
        md(
            """
            ## Оценка после длительного обучения: строгая замкнутая оценка checkpoint этого запуска

            Эта ячейка выполняется только при завершении длительного обучения. Она принудительно использует последний checkpoint из `LONG_OUTPUT` этого запуска и записывает результаты в `post_long_eval/`, чтобы они не смешивались с результатами «исторических защищённых весов» выше.
            """
        ),
        code(
            r'''
# POST_LONG_EVAL_CELL
post_eval_enabled = env_flag("RUN_POST_LONG_EVAL", False)
os.environ.setdefault("EVAL_USE_LONG_TRAIN", "1")
post_eval_dir = OUTPUT_ROOT / "post_long_eval"
post_eval_dir.mkdir(parents=True, exist_ok=True)
eval_episodes = os.environ.get("SMOLVLA_POST_EVAL_EPISODES", os.environ.get("SMOLVLA_EVAL_EPISODES", "10"))
checkpoints = list_checkpoints(LONG_OUTPUT)
if post_eval_enabled and not checkpoints:
    raise RuntimeError("Checkpoint длительного обучения SmolVLA этого запуска не найден; нельзя откатиться к старым весам и выдать их за оценку этого запуска.")
eval_policy = checkpoints[-1] if checkpoints else resolve_eval_policy(PRETRAINED_POLICY, LONG_OUTPUT, "SMOLVLA_EVAL_POLICY_PATH")
result_path = post_eval_dir / f"smolvla_post_long_seed0_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "smolvla",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=os.environ.get("SMOLVLA_EVAL_SEED_START", "0"),
    render=env_flag("RENDER_EVAL"),
    enabled=post_eval_enabled,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
rows = summarize_jsonl(result_path)
display(Markdown("**Точка проверки видео**: ниже показаны подготовленные в учебнике видео успеха и провала; пути JSONL реальной оценки — в таблице выше."))
show_video("smolvla_weighted500_red_success_seed0.mp4", "Эталонное видео успеха SmolVLA с красной чашкой")
show_video("smolvla_weighted500_blue_success_seed0.mp4", "Эталонное видео успеха SmolVLA с синей чашкой")
show_video("smolvla_weighted500_red_failure_seed8.mp4", "Эталонное видео провала SmolVLA: наблюдайте release и upright")
''',
        ),
    ]


def pi0_cells() -> list[dict]:
    return [
        md(
            """
            ## Оценка после длительного обучения: строгая замкнутая оценка checkpoint Pi0 этого запуска

            Эта ячейка выполняется только после завершения длительного обучения Pi0. Она принудительно использует последний checkpoint из `LONG_OUTPUT` этого запуска, оценивает по протоколу `observation.state` размерности 8 из датасета Pi0 и записывает результаты в `post_long_eval/`.
            """
        ),
        code(
            r'''
# POST_LONG_EVAL_CELL
post_eval_enabled = env_flag("RUN_POST_LONG_EVAL", False)
os.environ.setdefault("EVAL_USE_LONG_TRAIN", "1")
post_eval_dir = OUTPUT_ROOT / "post_long_eval"
post_eval_dir.mkdir(parents=True, exist_ok=True)
eval_episodes = os.environ.get("PI0_POST_EVAL_EPISODES", os.environ.get("PI0_EVAL_EPISODES", "14"))
eval_seed_start = os.environ.get("PI0_EVAL_SEED_START", "3000")
checkpoints = list_checkpoints(LONG_OUTPUT)
if post_eval_enabled and not checkpoints:
    raise RuntimeError("Checkpoint длительного обучения Pi0 этого запуска не найден; нельзя откатиться к старым весам и выдать их за оценку этого запуска.")
eval_policy = checkpoints[-1] if checkpoints else resolve_eval_policy(PI0_POLICY_PATH, LONG_OUTPUT, "PI0_EVAL_POLICY_PATH")
result_path = post_eval_dir / f"pi0_post_long_seed{eval_seed_start}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "pi0",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=eval_seed_start,
    render=env_flag("RENDER_EVAL"),
    enabled=post_eval_enabled,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
rows = summarize_jsonl(result_path)
display(Markdown("**Точка проверки видео**: результаты Pi0 raw нужно читать отдельно от scaffold и finisher; ниже — диагностические эталонные видео учебника."))
show_video("pi0_ep2_raw_vs_finisher_side_by_side.mp4", "Видео сравнения Pi0 raw и диагностического finisher")
show_image("pi0_raw_vs_finisher_diagnostic.png", "Диагностический график Pi0: raw, scaffold и строгая оценка")
''',
        ),
    ]


def act_cells() -> list[dict]:
    return [
        md(
            """
            ## Оценка после длительного обучения: строгая замкнутая оценка checkpoint ACT этого запуска

            Эта ячейка выполняется только после завершения длительного обучения ACT. Она использует последний checkpoint из `LONG_OUTPUT` этого запуска и записывает результаты в `post_long_eval/`. ACT сейчас — диагностический контрпример, результаты нужно определять по строгому physical success и проверке видео.
            """
        ),
        code(
            r'''
# POST_LONG_EVAL_CELL
post_eval_enabled = env_flag("RUN_POST_LONG_EVAL", False)
os.environ.setdefault("EVAL_USE_LONG_TRAIN", "1")
post_eval_dir = OUTPUT_ROOT / "post_long_eval"
post_eval_dir.mkdir(parents=True, exist_ok=True)
eval_episodes = os.environ.get("ACT_POST_EVAL_EPISODES", os.environ.get("ACT_EVAL_EPISODES", "6"))
eval_seed_start = os.environ.get("ACT_EVAL_SEED_START", "1030")
checkpoints = list_checkpoints(LONG_OUTPUT)
if post_eval_enabled and not checkpoints:
    raise RuntimeError("Checkpoint длительного обучения ACT этого запуска не найден; нельзя откатиться к старым весам и выдать их за оценку этого запуска.")
eval_policy = checkpoints[-1] if checkpoints else resolve_eval_policy(ACT_POLICY_PATH, LONG_OUTPUT, "ACT_EVAL_POLICY_PATH")
result_path = post_eval_dir / f"act_post_long_seed{eval_seed_start}_{int(eval_episodes)}ep.jsonl"
run_eval_policy_in_notebook(
    "act",
    eval_policy,
    result_path,
    episodes=eval_episodes,
    seed_start=eval_seed_start,
    render=env_flag("RENDER_EVAL"),
    enabled=post_eval_enabled,
    repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
)
rows = summarize_jsonl(result_path)
display(Markdown("**Точка проверки видео**: для ACT смотрите в первую очередь этап провала, а не только loss."))
show_image("act_success_sequence.jpg", "Ключевые кадры успеха ACT")
show_image("act_failure_sequence.jpg", "Ключевые кадры провала ACT")
''',
        ),
    ]


def main() -> None:
    append_cells("14_smolvla_end_to_end.ipynb", smolvla_cells())
    append_cells("15_pi0_end_to_end.ipynb", pi0_cells())
    append_cells("16_act_end_to_end.ipynb", act_cells())


if __name__ == "__main__":
    main()
