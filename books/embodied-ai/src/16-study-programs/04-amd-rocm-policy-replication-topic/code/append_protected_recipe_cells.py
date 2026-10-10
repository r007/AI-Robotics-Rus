#!/usr/bin/env python3
"""Add protected-checkpoint recipe cells to the end-to-end notebooks.

The notebooks have two purposes:

1. let learners run a short, cheap training/eval loop in class;
2. document the exact recipe that produced the protected public results.

Those two modes must be explicit.  Otherwise a learner sees a fresh Notebook
checkpoint and a protected score table without understanding the gap.
"""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
MARKER = "PROTECTED_RECIPE_CELL"


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


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def insert_after_first(notebook_name: str, needle: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    old_cells = [
        cell
        for cell in notebook.get("cells", [])
        if MARKER not in cell_source(cell) and "PROTECTED_TRAIN_CELL" not in cell_source(cell)
    ]

    # Remove accidental duplicated markdown-only post-long sections that were
    # created while iterating on the append script. Keep the code cell.
    compacted: list[dict] = []
    previous_post_markdown = False
    for cell in old_cells:
        src = cell_source(cell)
        is_post_markdown = (
            cell.get("cell_type") == "markdown"
            and src.strip().startswith("## 长训后追加评估")  # заголовок upstream-ноутбука, функциональный маркер, не переводить
        )
        is_protected_train_markdown = (
            cell.get("cell_type") == "markdown"
            and src.strip().startswith("### 可执行 protected")
        )
        is_protected_recipe_markdown = (
            cell.get("cell_type") == "markdown"
            and (
                src.strip().startswith("## Checkpoint 1.5：保护权重")  # функциональные маркеры upstream-ноутбуков, не переводить
                or src.strip().startswith("## Checkpoint 1.5：Pi0 保护权重")
                or src.strip().startswith("## Checkpoint 1.5：ACT 保护结果")
            )
        )
        if is_protected_train_markdown or is_protected_recipe_markdown:
            continue
        if is_post_markdown and previous_post_markdown:
            continue
        compacted.append(cell)
        previous_post_markdown = is_post_markdown

    insert_at = None
    for idx, cell in enumerate(compacted):
        if needle in cell_source(cell):
            insert_at = idx + 1
            break
    if insert_at is None:
        insert_at = min(6, len(compacted))

    compacted[insert_at:insert_at] = cells
    notebook["cells"] = compacted
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {path}")


def smolvla_cells() -> list[dict]:
    return [
        md(
            """
            ## Checkpoint 1.5: рецепт обучения защищённых весов, не смешивается с лёгким обучением на занятии

            Предыдущая ячейка показывает уже защищённый результат `57/60`. Здесь объясняется, как он был обучен. Длительное обучение на занятии по умолчанию лишь показывает полный процесс; чтобы воспроизвести защищённые веса, переключитесь на protected recipe ниже и выполните полное обучение и строгую оценку на 60 episode.
            """
        ),
        md(
            """
            ### Согласование критериев результата: малая панель этого запуска и официальная защищённая оценка

            Ячейка оценки ниже в Notebook запускает недорогую малую панель, чтобы проверить, что checkpoint этого запуска работает в замкнутом цикле и создаёт видео; официальная защищённая оценка использует более крупную фиксированную панель, и именно её критерий принимается для публикации весов и отчёта учебника.

            | Критерий | Успешность | Охват оценки | Пояснение |
            | --- | --- | --- | --- |
            | Малая панель Notebook этого запуска | `3/4` | post-long eval seed0-3 | для показа на занятии и приёмки видео, не заменяет официальный результат |
            | Официальная защищённая оценка | `57/60` | red30 + blue30 strict physical success | красная `27/30`, синяя `30/30`, принимается как веса по умолчанию для публикации |
            """
        ),
        code(
            r'''
# PROTECTED_RECIPE_CELL
rows = [
    ("Учебный режим по умолчанию", "demo_data_language", "обычный EpisodeAwareSampler", "SMOLVLA_STEPS=5000", "малая панель этого запуска 3/4", "для прохождения на занятии и показа видео, не гарантирует 57/60"),
    ("Защищённый рецепт parent", "demo_data_language", "базовое длительное обучение SmolVLA", "5000 steps", "родительские веса для взвешенного дообучения", "сначала получить пригодный parent checkpoint"),
    ("Защищённый рецепт weighted-blue", "demo_data_language", "взвешивание frame/episode для синей чашки, без копирования исходного parquet", "дообучение 1000 steps, выбор step500", "red30 + blue30 strict", "текущее воспроизведение 57/60; красная 27/30, синяя 30/30"),
]
md_table(["Режим", "Данные", "Стратегия выборки и данных", "Шаги обучения", "Панель оценки", "Пояснение"], rows)

protected_env = {
    "TEACHING_RECIPE": "protected",
    "TRAIN_DATA_ROOT": str(DATA_ROOT / "demo_data_language"),
    "SMOLVLA_STEPS": "5000 + weighted-blue continuation",
    "SMOLVLA_EVAL_EPISODES": "60",
    "SMOLVLA_POLICY_PATH": str(MODEL_ROOT / "smolvla_weighted_000500" / "pretrained_model"),
}
print(json.dumps(protected_env, ensure_ascii=False, indent=2))
print("Внимание: логика выборки weighted-blue должна совпадать с Weighted sampler из README_04 и README_06; обычное длительное обучение на занятии нельзя выдавать за результат protected.")
'''
        ),
        md(
            """
            ### Исполняемое protected-обучение: parent 5000 и дообучение weighted-blue

            После установки `RUN_PROTECTED_TRAIN=1` эта ячейка реально обучает две стадии внутри Notebook: сначала parent, затем продолжает взвешенное дообучение blue с parent checkpoint в качестве инициализации. По умолчанию выключено, чтобы при открытии на занятии не запускалось длительное обучение.
            """
        ),
        code(
            r'''
# PROTECTED_TRAIN_CELL
protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)
if not protected_train_enabled:
    print("Не запущено. После установки RUN_PROTECTED_TRAIN=1 эта ячейка нативно обучит protected recipe SmolVLA.")
else:
    DATASET_REPO_ID = globals().get("DATASET_REPO_ID", os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language"))
    TRAIN_DATA_ROOT = globals().get("TRAIN_DATA_ROOT", Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language")))
    CONFIG_DIR = OUTPUT_ROOT / "configs"
    RUN_ROOT = OUTPUT_ROOT / "runs" / "smolvla_protected_recipe"
    PARENT_OUTPUT = RUN_ROOT / "parent_5000"
    WEIGHTED_OUTPUT = RUN_ROOT / "weighted_blue2_step1000"
    parent_config = make_lerobot_train_config(
        "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, PARENT_OUTPUT,
        steps=int(os.environ.get("SMOLVLA_PARENT_STEPS", "5000")),
        batch_size=int(os.environ.get("SMOLVLA_BATCH_SIZE", "4")),
        chunk_size=50,
        n_action_steps=50,
    )
    weighted_config = make_lerobot_train_config(
        "smolvla", DATASET_REPO_ID, TRAIN_DATA_ROOT, WEIGHTED_OUTPUT,
        steps=int(os.environ.get("SMOLVLA_WEIGHTED_STEPS", "1000")),
        batch_size=int(os.environ.get("SMOLVLA_BATCH_SIZE", "4")),
        chunk_size=50,
        n_action_steps=50,
    )
    parent_path = write_json_yaml(CONFIG_DIR / "smolvla_protected_parent_5000.yaml", parent_config)
    weighted_path = write_json_yaml(CONFIG_DIR / "smolvla_protected_weighted_blue2.yaml", weighted_config)
    train_lerobot_config_in_notebook(parent_path, enabled=True, progress_name="SmolVLA protected parent")
    parent_ckpt = list_checkpoints(PARENT_OUTPUT)[-1]
    old_override = os.environ.get("SMOLVLA_PRETRAINED_PATH_OVERRIDE")
    old_mode = os.environ.get("NOTEBOOK_FRAME_WEIGHT_MODE")
    old_blue = os.environ.get("NOTEBOOK_BLUE_WEIGHT")
    os.environ["SMOLVLA_PRETRAINED_PATH_OVERRIDE"] = str(parent_ckpt)
    os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = "blue"
    os.environ["NOTEBOOK_BLUE_WEIGHT"] = os.environ.get("SMOLVLA_BLUE_WEIGHT", "2.0")
    try:
        train_lerobot_config_in_notebook(weighted_path, enabled=True, progress_name="SmolVLA protected weighted-blue")
    finally:
        if old_override is None:
            os.environ.pop("SMOLVLA_PRETRAINED_PATH_OVERRIDE", None)
        else:
            os.environ["SMOLVLA_PRETRAINED_PATH_OVERRIDE"] = old_override
        if old_mode is None:
            os.environ.pop("NOTEBOOK_FRAME_WEIGHT_MODE", None)
        else:
            os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = old_mode
        if old_blue is None:
            os.environ.pop("NOTEBOOK_BLUE_WEIGHT", None)
        else:
            os.environ["NOTEBOOK_BLUE_WEIGHT"] = old_blue
    print("protected candidate checkpoints:")
    list_checkpoints(WEIGHTED_OUTPUT)
'''
        ),
    ]


def pi0_cells() -> list[dict]:
    return [
        md(
            """
            ## Checkpoint 1.5: родословная обучения защищённых весов Pi0

            Результат Pi0 `12/14` не появляется сам собой из обычной ячейки `PI0_STEPS=8500`; он получен из данных clean40 success-only, дополнительного веса для синей чашки и выбора по порогу S7500→S8500. Лёгкое обучение Notebook по умолчанию нужно для понимания процесса, а protected recipe — для воспроизведения критерия из отчёта.
            """
        ),
        md(
            """
            ### Согласование критериев результата: малая панель этого запуска, защищённая оценка и hard-панель

            Малая панель Pi0 на занятии и порог защищённых весов — не один и тот же результат. Малая панель показывает лишь то, что обычный checkpoint этого длительного обучения не восстановился; защищённый результат получен по родословной S8500 clean40 success-only + blue2x.

            | Критерий | Успешность | Охват оценки | Пояснение |
            | --- | --- | --- | --- |
            | Малая панель Notebook этого запуска | `0/4` | post-long eval seed3000-3003 | обычный checkpoint этого запуска по-прежнему проваливается, нельзя записывать как успех |
            | Официальная защищённая оценка | `12/14` | full14 strict physical success | текущий кандидат Pi0 для публикации, красная `6/7`, синяя `6/7` |
            | Панель unseen | `9/10` | unseen seeds 3010-3019 | лучше раннего результата unseen `7/10` |
            | Панель hard | `6/8` | hard8 | историческая hard `8/8` ещё не восстановлена, можно писать только как частичное восстановление |
            """
        ),
        code(
            r'''
# PROTECTED_RECIPE_CELL
rows = [
    ("clean40-only S7500", "pi0_clean40_successonly", "фильтрация успешных траекторий; неуспешные не смешиваются", "7500 steps", "canary 2/4", "родительские веса, начинается восстановление для красной чашки"),
    ("blue2x S8500", "clean40 + взвешивание и догрузка синей чашки", "вес синей чашки повышен, но сохранён защитный порог для красной", "+1000 steps", "canary 3/4 -> full14", "защищённый результат 12/14, unseen 9/10, hard8 6/8"),
    ("blue2x S9500", "то же", "дальнейшее увеличение шагов", "+2000 steps", "откат к 2/4", "показывает, что длительное обучение нельзя наращивать вслепую; точка выбора — S8500"),
]
md_table(["Этап", "Данные", "Стратегия выборки и отбора", "Шаги обучения", "Порог", "Вывод"], rows)

protected_env = {
    "TEACHING_RECIPE": "protected",
    "PI0_TRAIN_DATA_ROOT": str(DATA_ROOT / "pi0_clean40_successonly"),
    "PI0_STEPS": "8500",
    "PI0_BATCH_SIZE": "2",
    "PI0_EVAL_EPISODES": "14",
    "PI0_EVAL_SEED_START": "3000",
    "PI0_POLICY_PATH": str(MODEL_ROOT / "pi0_clean40_successonly_blue2x_s8500" / "pretrained_model"),
}
print(json.dumps(protected_env, ensure_ascii=False, indent=2))
print("Внимание: эту линию можно записать только как частичное восстановление Pi0. Историческая hard8 8/8 не восстановлена; нельзя выдавать scaffold или малую панель за успех raw Pi0.")
'''
        ),
        md(
            """
            ### Исполняемое protected-обучение: clean40 S7500 и blue2x S8500

            После установки `RUN_PROTECTED_TRAIN=1` эта ячейка сначала обучает S7500 на данных clean40 success-only, затем дообучает blue2x до S8500, инициализируясь из S7500. Так путь обучения в Notebook совпадает с причинно-следственной цепочкой защищённого результата `12/14`.

            Если времени на занятии или в облаке не хватает, можно задать `PI0_PARENT_CHECKPOINT_PATH=/path/to/s7500/pretrained_model`: Notebook пропустит длительное обучение S7500 и сразу нативно выполнит дообучение blue2x от этих родительских весов. Это не другой метод, а повторное использование уже обученного промежуточного защищённого узла.
            """
        ),
        code(
            r'''
# PROTECTED_TRAIN_CELL
protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)
if not protected_train_enabled:
    print("Не запущено. После установки RUN_PROTECTED_TRAIN=1 эта ячейка нативно обучит protected recipe Pi0.")
else:
    DATASET_REPO_ID = globals().get("DATASET_REPO_ID", "datawhale_eai_pnp_pi0_clean_oracle_y060z000_3100_3139_g8_rebuild_v1")
    TRAIN_DATA_ROOT = globals().get("TRAIN_DATA_ROOT", Path(os.environ.get("PI0_TRAIN_DATA_ROOT", DATA_ROOT / "pi0_clean40_successonly")))
    CONFIG_DIR = OUTPUT_ROOT / "configs"
    RUN_ROOT = OUTPUT_ROOT / "runs" / "pi0_protected_recipe"
    CLEAN_OUTPUT = RUN_ROOT / "clean40_s7500"
    BLUE_OUTPUT = RUN_ROOT / "blue2x_s8500"
    clean_config = make_lerobot_train_config(
        "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, CLEAN_OUTPUT,
        steps=int(os.environ.get("PI0_CLEAN_STEPS", "7500")),
        batch_size=int(os.environ.get("PI0_BATCH_SIZE", "2")),
        chunk_size=50,
        n_action_steps=50,
    )
    blue_config = make_lerobot_train_config(
        "pi0", DATASET_REPO_ID, TRAIN_DATA_ROOT, BLUE_OUTPUT,
        steps=int(os.environ.get("PI0_BLUE_STEPS", "1000")),
        batch_size=int(os.environ.get("PI0_BATCH_SIZE", "2")),
        chunk_size=50,
        n_action_steps=50,
    )
    clean_path = write_json_yaml(CONFIG_DIR / "pi0_protected_clean40_s7500.yaml", clean_config)
    blue_path = write_json_yaml(CONFIG_DIR / "pi0_protected_blue2x_s8500.yaml", blue_config)
    parent_checkpoint = os.environ.get("PI0_PARENT_CHECKPOINT_PATH") or os.environ.get("PI0_S7500_CHECKPOINT_PATH")
    if parent_checkpoint:
        clean_ckpt = Path(parent_checkpoint)
        if not clean_ckpt.exists():
            raise FileNotFoundError(f"PI0_PARENT_CHECKPOINT_PATH does not exist: {clean_ckpt}")
        print("Используются переданные веса S7500, длительное обучение clean40 пропущено:", public_path(clean_ckpt))
    else:
        train_lerobot_config_in_notebook(clean_path, enabled=True, progress_name="Pi0 protected clean40")
        clean_ckpt = list_checkpoints(CLEAN_OUTPUT)[-1]
    old_override = os.environ.get("PI0_PRETRAINED_PATH_OVERRIDE")
    old_mode = os.environ.get("NOTEBOOK_FRAME_WEIGHT_MODE")
    old_blue = os.environ.get("NOTEBOOK_BLUE_WEIGHT")
    os.environ["PI0_PRETRAINED_PATH_OVERRIDE"] = str(clean_ckpt)
    os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = "blue"
    os.environ["NOTEBOOK_BLUE_WEIGHT"] = os.environ.get("PI0_BLUE_WEIGHT", "2.0")
    try:
        train_lerobot_config_in_notebook(blue_path, enabled=True, progress_name="Pi0 protected blue2x")
    finally:
        if old_override is None:
            os.environ.pop("PI0_PRETRAINED_PATH_OVERRIDE", None)
        else:
            os.environ["PI0_PRETRAINED_PATH_OVERRIDE"] = old_override
        if old_mode is None:
            os.environ.pop("NOTEBOOK_FRAME_WEIGHT_MODE", None)
        else:
            os.environ["NOTEBOOK_FRAME_WEIGHT_MODE"] = old_mode
        if old_blue is None:
            os.environ.pop("NOTEBOOK_BLUE_WEIGHT", None)
        else:
            os.environ["NOTEBOOK_BLUE_WEIGHT"] = old_blue
    print("protected candidate checkpoints:")
    list_checkpoints(BLUE_OUTPUT)
'''
        ),
    ]


def act_cells() -> list[dict]:
    return [
        md(
            """
            ## Checkpoint 1.5: результаты защиты ACT и исторические результаты — не одна родословная обучения

            Текущий восстановленный защищённый кандидат ACT — `15/30`, запись исторического учебника — `17/30`. Notebook должен разделять разные родословные обучения и протоколы оценки: текущий защищённый кандидат repair15 уже можно проверить, но историческое 17/30 пока не восстановлено полностью.
            """
        ),
        md(
            """
            ### Согласование критериев результата: малая панель этого запуска, текущее воспроизведение и историческая запись

            Главная задача этого Notebook по ACT — диагностика, а не продвижение успешных весов. Малая панель этого запуска, защищённый результат текущего воспроизведения и исторический лучший результат нужно записывать отдельно, чтобы старые результаты эксперимента не выдавались за воспроизводимый вывод этого запуска.

            | Критерий | Успешность | Охват оценки | Пояснение |
            | --- | --- | --- | --- |
            | Малая панель Notebook этого запуска | `0/4` | post-long eval seed1030-1033 | обычное обучение этого запуска по-прежнему не даёт успеха в замкнутом цикле |
            | Текущая защита после воспроизведения | `15/30` | stable61 -> dagger low-lr step1500 strict30 | лучший проверяемый защищённый кандидат |
            | Старый baseline fallback | `7/30` | stable61 step2500 strict30 | сохранён для сравнения |
            | Запись исторического учебника | `17/30` | старый DAgger best025 v1 strict30 | ещё не воспроизведено, используется только как опытная цель и контрпример |
            """
        ),
        code(
            r'''
# PROTECTED_RECIPE_CELL
rows = [
    ("Учебный режим по умолчанию", "demo_data_language", "обычное обучение ACT", "ACT_STEPS=5000", "малая панель этого запуска 0/4", "для понимания диагностики провалов ACT в замкнутом цикле"),
    ("Текущий protected repair15", "act_base72_plus_dagger3x3_rebuild_v1", "дообучение с низким learning rate от stable61; chunk20/n10; timestamp offset 49-80:2.0", "2500 continuation steps", "strict30", "текущая защита 15/30; три панели 3/10 + 4/10 + 8/10"),
    ("Старый baseline fallback", "данные и конфигурация stable61", "выбор промежуточного checkpoint, step2500 лучше step5000", "около 5000 steps", "strict30", "7/30"),
    ("Лучший результат исторического учебника", "старый DAgger best025 v1", "опыт on-policy и recovery полнее", "историческая серия экспериментов", "strict30", "17/30; ещё не восстановлено полностью, нельзя выдавать за результат текущих весов"),
]
md_table(["Режим", "Данные", "Стратегия обучения и выбора", "Шаги обучения", "Панель оценки", "Вывод"], rows)

act_recipe = os.environ.get("ACT_RECIPE", "stable61").strip().lower()
is_repair = act_recipe in {"repair15", "stable61_to_dagger", "nomemleak"}
protected_env = {
    "TEACHING_RECIPE": "protected-repair15" if is_repair else "protected-fallback",
    "ACT_RECIPE": act_recipe,
    "ACT_TRAIN_DATA_ROOT": str(DATA_ROOT / ("act_base72_plus_dagger3x3_rebuild_v1" if is_repair else "demo_data_language")),
    "ACT_STEPS": "2500 continuation" if is_repair else "5000",
    "ACT_EVAL_EPISODES": "30",
    "ACT_EVAL_SEED_START": "1030",
    "ACT_EVAL_N_ACTION_STEPS": "10" if is_repair else "50",
    "ACT_EVAL_CLAMP_TIMESTAMP": "1" if is_repair else "0",
    "ACT_EVAL_EPISODE_TIMESTAMP_OFFSETS": "49-80:2.0" if is_repair else "",
    "ACT_POLICY_PATH": str(MODEL_ROOT / ("act_stable61_to_dagger_nomemleak_step1500_strict15of30" if is_repair else "act_stable61_step2500") / "pretrained_model"),
}
print(json.dumps(protected_env, ensure_ascii=False, indent=2))
print("Внимание: в учебнике ACT — отрицательный пример и линия диагностики. Главное, чему нужно научиться: строгий success, проверка видео, корзины провалов по этапам и выбор защищённого checkpoint.")
'''
        ),
        md(
            """
            ### Исполняемое protected-current обучение: fallback stable61 или repair15

            После установки `RUN_PROTECTED_TRAIN=1` эта ячейка нативно обучает protected recipe ACT. По умолчанию используется stable61/fallback; при `ACT_RECIPE=repair15` готовится конфигурация continuation `chunk20/n10` для текущего защищённого кандидата, а при наличии пути загружается инициализация stable61. Историческое `17/30` ещё не воспроизведено полностью, поэтому Notebook не выдаёт его за лучший результат.
            """
        ),
        code(
            r'''
# PROTECTED_TRAIN_CELL
protected_train_enabled = env_flag("RUN_PROTECTED_TRAIN", False)
if not protected_train_enabled:
    print("Не запущено. После установки RUN_PROTECTED_TRAIN=1 эта ячейка нативно обучит protected recipe ACT; ACT_RECIPE=repair15 подготовит текущий защищённый протокол 15/30.")
else:
    DATASET_REPO_ID = globals().get("DATASET_REPO_ID", os.environ.get("ACT_DATASET_REPO_ID", "datawhale_eai_pnp_language"))
    TRAIN_DATA_ROOT = globals().get("TRAIN_DATA_ROOT", Path(os.environ.get("ACT_TRAIN_DATA_ROOT", DATA_ROOT / "demo_data_language")))
    CONFIG_DIR = OUTPUT_ROOT / "configs"
    RUN_ROOT = OUTPUT_ROOT / "runs" / "act_protected_current_recipe"
    if is_repair:
        REPAIR_OUTPUT = RUN_ROOT / "repair15_continuation"
        os.environ.setdefault("ACT_PRETRAINED_PATH_OVERRIDE", str(MODEL_ROOT / "act_stable61_step2500" / "pretrained_model"))
        repair_config = make_lerobot_train_config(
            "act", DATASET_REPO_ID,
            Path(os.environ.get("ACT_REPAIR_TRAIN_DATA_ROOT", TRAIN_DATA_ROOT)),
            REPAIR_OUTPUT,
            steps=int(os.environ.get("ACT_REPAIR_STEPS", "2500")),
            batch_size=int(os.environ.get("ACT_BATCH_SIZE", "16")),
            chunk_size=20,
            n_action_steps=10,
        )
        repair_path = write_json_yaml(CONFIG_DIR / "act_protected_repair15.yaml", repair_config)
        train_lerobot_config_in_notebook(repair_path, enabled=True, progress_name="ACT protected repair15")
        list_checkpoints(REPAIR_OUTPUT)
    else:
        STABLE_OUTPUT = RUN_ROOT / "stable61_full"
        stable_config = make_lerobot_train_config(
            "act", DATASET_REPO_ID,
            Path(os.environ.get("ACT_STABLE61_TRAIN_DATA_ROOT", TRAIN_DATA_ROOT)),
            STABLE_OUTPUT,
            steps=int(os.environ.get("ACT_STEPS", "5000")),
            batch_size=int(os.environ.get("ACT_BATCH_SIZE", "8")),
            chunk_size=50,
            n_action_steps=50,
        )
        stable_path = write_json_yaml(CONFIG_DIR / "act_protected_current_stable61.yaml", stable_config)
        train_lerobot_config_in_notebook(stable_path, enabled=True, progress_name="ACT protected-current stable61")
        print("protected-current candidate checkpoints:")
        list_checkpoints(STABLE_OUTPUT)
'''
        ),
    ]


def main() -> None:
    insert_after_first("14_smolvla_end_to_end.ipynb", "Checkpoint 2：显示严格成功与失败视频", smolvla_cells())  # имена upstream-ноутбуков и заголовки-якоря, функциональные, не переводить
    insert_after_first("15_pi0_end_to_end.ipynb", "Checkpoint 2：gated 权限和缓存检查", pi0_cells())  # имена upstream-ноутбуков и заголовки-якоря, функциональные, не переводить
    insert_after_first("16_act_end_to_end.ipynb", "Checkpoint 2：生成配置并真实启动训练", act_cells())  # имена upstream-ноутбуков и заголовки-якоря, функциональные, не переводить


if __name__ == "__main__":
    main()
