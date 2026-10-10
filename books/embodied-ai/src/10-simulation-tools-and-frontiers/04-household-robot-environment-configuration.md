**Бенчмарк оценки низкоуровневых операций в задачах уборки дома**

**[Статья](https://arxiv.org/abs/2412.13211)** | **[Сайт](https://arth-shukla.github.io/mshab/)** | **[Модели](https://huggingface.co/arth-shukla/mshab_checkpoints)** | **[Набор данных](https://arth-shukla.github.io/mshab/#dataset-section)** | **[Дополнительные материалы](https://sites.google.com/view/maniskill-hab)**

---

## Установка и настройка

1. **Установка окружения**

   ```bash
   conda create -n mshab python=3.9
   conda activate mshab
   ```

2. **Установка ManiSkill3 и кода проекта**

   ```bash
   git clone https://github.com/haosulab/ManiSkill.git -b mshab --single-branch
   pip install -e ManiSkill
   pip install -e .  # Необязательно: установить зависимости для обучения и разработки: pip install -e .[train,dev]
   ```

3. **Загрузка нужных наборов данных**

   ```bash
   for dataset in ycb ReplicaCAD ReplicaCADRearrange; do python -m mani_skill.utils.download_asset "$dataset"; done
   ```

4. **Импорт окружений**

   ```python
   import mshab.envs
   ```

---

## [Необязательно] Контрольные точки, наборы данных и генерация данных

- Модели и наборы данных можно получить с HuggingFace (около 490 ГБ):

  ```bash
  huggingface-cli login

  huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints

  export MS_ASSET_DIR="~/.maniskill"
  export MSHAB_DATASET_DIR="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange-dataset"

  huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-TidyHouse --local-dir "$MSHAB_DATASET_DIR/tidy_house"
  huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-PrepareGroceries --local-dir "$MSHAB_DATASET_DIR/prepare_groceries"
  huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-SetTable --local-dir "$MSHAB_DATASET_DIR/set_table"
  ```

- Генерация данных скриптом (критерии фильтрации траекторий можно настроить):

  ```bash
  bash scripts/gen_dataset.sh
  ```

---

## Быстрый старт

MS-HAB предоставляет:

- **Среду оценки** `SequentialTask-v0`: для оценки полной задачи (поддерживает цепочки подзадач);

- **Учебную среду** `[Name]SubtaskTrain-v0`: для обучения отдельного навыка, например `PickSubtaskTrain-v0`.

Пример кода:

```python
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import mshab.envs
from mshab.envs.planner import plan_data_from_file

task = "tidy_house"
subtask = "pick"
split = "train"

plan_data = plan_data_from_file(REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json")
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

env = gym.make(
    f"{subtask.capitalize()}SubtaskTrain-v0",
    num_envs=252,
    obs_mode="rgbd",
    sim_backend="gpu",
    robot_uids="fetch",
    control_mode="pd_joint_delta_pos",
    reward_mode="normalized_dense",
    render_mode="rgb_array",
    max_episode_steps=200,
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    spawn_data_fp=spawn_data_fp,
)

venv = ManiSkillVectorEnv(env, max_episode_steps=1000, ignore_terminations=True)
obs, info = venv.reset()
```

---

## Сцены и настройка задач

- **Сцены**: ReplicaCAD предоставляет 84 квартиры (63 для обучения, 21 для валидации).

- **План задачи**: определяет порядок задач и цели, например «Взять → Положить».

- **Сгенерированные данные**: начальные состояния робота и объектов заранее создаются скриптом (это нужно для стабильности при GPU-симуляции).

### Простая настройка

- Можно загрузить только часть планов задач для отладки.

- Критерии успеха и неудачи можно менять через `task_cfgs` (например, диапазон цели укладки).

### Расширенная настройка

- Цепочку задач можно удлинять, но это должно быть логично, например не ставить «Взять → Взять» подряд.

- С помощью изменения конфигурации и `SceneBuilder` можно менять объекты, сцену и т. д.

---

## Обучение

Поддерживаются несколько алгоритмов (SAC, PPO, клонирование поведения, диффузионная политика):

```bash
bash scripts/train_[algo].sh
```

Пример:

```bash
python -m mshab.train_sac \
  mshab_checkpoints/rl/tidy_house/pick/all/config.yml \
  algo.gamma=0.8
```

**Продолжение обучения**

```bash
python -m mshab.train_ppo \
  mshab_checkpoints/rl/tidy_house/pick/all/config.yml \
  model_ckpt=mshab_checkpoints/rl/tidy_house/pick/all/policy.pt \
  algo.lr=1e-3
```

---

## Оценка

Перед оценкой нужно скачать или сгенерировать данные. Для оценки полной задачи или подзадач используйте скрипт:

```bash
bash scripts/evaluate_sequential_task.sh
```
