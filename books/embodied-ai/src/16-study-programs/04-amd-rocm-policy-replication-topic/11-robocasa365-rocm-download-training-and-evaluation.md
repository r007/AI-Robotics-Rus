# RoboCasa365: загрузка, обучение и оценка на ROCm

RoboCasa365 — это фреймворк обучения и бенчмарк для роботов в бытовой обстановке: 365 задач, кухонные ресурсы, демонстрационные данные, а также точки входа для стратегий GR00T и Pi0.5. В этой главе воспроизводятся загрузка ресурсов, безголовый рендеринг, вывод официального checkpoint, официальная оценка 16 задач и видео с четырёх видов, которые мы получили на AMD Ryzen AI MAX+ 395.

## 1. Результаты главы

- RoboCasa, RoboSuite и среда MuJoCo;
- около 10 ГБ кухонных ресурсов и необязательные сгенерированные ИИ объекты;
- checkpoint GR00T N1.5 и многозадачный checkpoint Pi0.5;
- одинаковый протокол 16 задач × 50 эпизодов;
- видео с четырёх камер: центральной, левой, правой и «глаз-рука»;
- точки входа для продолжения обучения GR00T или Pi0.5.

### Версия и контракт задач

После установки зафиксируйте версии исходного кода:

```bash
git -C "$SRC_ROOT/robosuite" rev-parse HEAD
git -C "$SRC_ROOT/robocasa" rev-parse HEAD
```

Официальная оценка использует PandaOmron, 12-мерное действие, одни и те же 16 задач и по 50 эпизодов на задачу. Успех задачи определяет проверка условий в среде RoboCasa: состояние объектов, отношения контейнеров, положение суставов открытия и устойчивый кадр должны одновременно соответствовать определению задачи. Скрипт оценки записывает это поле в результаты по каждому эпизоду без изменений.

## 2. Загрузка исходного кода и ресурсов

```bash
# Сначала в текущем терминале задайте переменные каталогов WORK_ROOT, SRC_ROOT и др. из главы README_10.
test -n "$SRC_ROOT" && test -n "$DATA_ROOT" && test -n "$MODEL_ROOT"
cd "$SRC_ROOT"

git clone https://github.com/ARISE-Initiative/robosuite.git
git clone https://github.com/robocasa/robocasa.git

python -m pip install -e "$SRC_ROOT/robosuite"
python -m pip install -e "$SRC_ROOT/robocasa"
python -m pip install imageio imageio-ffmpeg

cd "$SRC_ROOT/robocasa"
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

Ресурсы загружаются долго; если загрузка прервалась, повторите ту же команду. Чтобы посмотреть кухни, объекты и задачи:

```bash
python -m robocasa.demos.demo_kitchen_scenes
python -m robocasa.demos.demo_objects
python -m robocasa.demos.demo_tasks
```

Для обучения `pretrain_human300` нужны человеческие демонстрации из предобучающего разбиения. RoboCasa по реестру наборов данных загрузит все соответствующие задачи:

```bash
cd "$SRC_ROOT/robocasa"
python -m robocasa.scripts.download_datasets \
  --split pretrain \
  --source human
```

Этот набор данных очень большой. Если изучаете только одну-две задачи, загрузите их по именам:

```bash
python -m robocasa.scripts.download_datasets \
  --tasks PickPlaceCounterToCabinet CloseFridge
```

После загрузки проверьте по реестру, что `pretrain_human300` найден и содержит каталоги данных:

```bash
python - <<'PY'
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

rows = DATASET_SOUP_REGISTRY["pretrain_human300"]
print("datasets:", len(rows))
print("first:", rows[0])
assert len(rows) > 0
PY
```

## 3. Загрузка официальных стратегий

```bash
hf download robocasa/robocasa365_checkpoints \
  --include 'gr00t_n1-5_multitask_120000/**' \
  --local-dir "$MODEL_ROOT/robocasa365"

hf download robocasa/robocasa365_checkpoints \
  --include 'pi05_pretrain_human300_multitask_75000/**' \
  --local-dir "$MODEL_ROOT/robocasa365"
```

В каталоге модели должно быть:

```text
checkpoints/robocasa365/
├── gr00t_n1-5_multitask_120000/
└── pi05_pretrain_human300_multitask_75000/
```

## 4. Проверка окружения и действий

Публичные вспомогательные скрипты находятся в [Radeon Physical AI Evidence Suite](https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite):

```bash
git clone https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite.git \
  "$SRC_ROOT/radeon-physical-ai-evidence-suite"

export MUJOCO_GL=egl
python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/robocasa_amd_smoke.py" \
  --task PickPlaceCounterToCabinet \
  --episodes 3 \
  --steps 40 \
  --out-dir "$RUN_ROOT/robocasa365/smoke"
```

Мобильный манипулятор PandaOmron имеет 12-мерный интерфейс: правая рука 6, захват 1, плоское шасси 3, подъёмный корпус 1, режим шасси 1. Проверьте каждое измерение действия вспомогательным скриптом:

```bash
python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/robocasa_mobile_mvp.py" \
  --env-name PickPlaceCounterToMicrowave \
  --dataset-base-path "$DATA_ROOT/robocasa365" \
  --output-dir "$RUN_ROOT/robocasa365/mobile_gate" \
  --episodes 1 \
  --steps 40
```

## 5. Обучение GR00T N1.5

```bash
cd "$SRC_ROOT"
git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git
python -m pip install -e "$SRC_ROOT/Isaac-GR00T"

cd "$SRC_ROOT/Isaac-GR00T"
python scripts/gr00t_finetune.py \
  --output-dir "$RUN_ROOT/robocasa365/gr00t_train" \
  --dataset_soup pretrain_human300 \
  --max_steps 120000 \
  --save_steps 5000 \
  --batch_size 32 \
  --report_to tensorboard
```

`pretrain_human300` соответствует комбинации человеческих демонстраций RoboCasa365 для 300 задач. Перед длительным обучением установите `--max_steps` равным 10, проверьте loss, загрузку GPU и запись checkpoint, и только потом верните 120000 шагов. Официально для обучения рекомендуется видеопамять от 80 ГБ; на AMD устройствах меньшего размера лучше проводить вывод, оценку или эксперименты с меньшим размером батча. При уменьшении batch size обязательно записывайте фактическое значение в конфигурацию запуска.

## 6. Обучение Pi0.5

Pi0.5 использует отдельное окружение JAX:

```bash
cd "$SRC_ROOT"
git clone https://github.com/robocasa-benchmark/openpi.git robocasa-openpi
python -m pip install -e "$SRC_ROOT/robocasa-openpi"
python -m pip install -e "$SRC_ROOT/robocasa-openpi/packages/openpi-client"

cd "$SRC_ROOT/robocasa-openpi"
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
  pi05_pretrain_human300 \
  --exp-name=robocasa_pi05_amd
```

Перед запуском AMD JAX обязательно пройдите предварительную проверку матрицы из главы README_10. Модель, статистику нормализации и конфигурацию обучения храните в одном выходном каталоге.

## 7. Официальная оценка

GR00T:

```bash
cd "$SRC_ROOT/Isaac-GR00T"
export MATCH16_SITE="$EVERY_EMBODIED_ROOT/<topic-folder>/code/robocasa_match16"
export PYTHONPATH="$MATCH16_SITE${PYTHONPATH:+:$PYTHONPATH}"

python scripts/run_eval.py \
  --model_path "$MODEL_ROOT/robocasa365/gr00t_n1-5_multitask_120000" \
  --task_set amd_match16 \
  --split pretrain \
  --n_episodes 50 \
  --n_envs 1 \
  --video_dir "$RUN_ROOT/robocasa365/gr00t_eval"

python gr00t/eval/get_eval_stats.py \
  --dir "$RUN_ROOT/robocasa365/gr00t_eval" \
  --task_set amd_match16
```

`<topic-folder>` — это каталог этой темы в исходном репозитории `every-embodied` (раздел тем для совместного обучения, тема AMD ROCm).

`sitecustomize.py` регистрирует `amd_match16` при запуске Python и не меняет исходный код RoboCasa. Переменная `MATCH16_SITE` берёт значение `$EVERY_EMBODIED_ROOT` из главы README_10, поэтому Every Embodied и исходный код симуляции могут находиться на разных дисках.

Pi0.5 разделяется на сервис стратегии и клиент симуляции. Сервис постоянно занимает терминал, поэтому откройте два терминала и в каждом активируйте одно и то же окружение ROCm JAX.

Терминал A запускает сервис стратегии:

```bash
cd "$SRC_ROOT/robocasa-openpi"
python scripts/serve_policy.py \
  --port=8000 policy:checkpoint \
  --policy.config=pi05_pretrain_human300 \
  --policy.dir="$MODEL_ROOT/robocasa365/pi05_pretrain_human300_multitask_75000"
```

Когда порт `8000` начнёт слушать, оставьте терминал A без изменений. В терминале B запускаются клиент симуляции и сводка результатов:

```bash
cd "$SRC_ROOT/robocasa-openpi"
export MATCH16_SITE="$EVERY_EMBODIED_ROOT/<topic-folder>/code/robocasa_match16"
export PYTHONPATH="$MATCH16_SITE${PYTHONPATH:+:$PYTHONPATH}"
python examples/robocasa/main.py \
  --args.port 8000 \
  --args.task_soup amd_match16 \
  --args.split pretrain \
  --args.num_trials 50 \
  --args.log_dir "$RUN_ROOT/robocasa365/pi05_eval"

python examples/robocasa/get_eval_stats.py \
  --dir "$RUN_ROOT/robocasa365/pi05_eval"
```

Наш официальный протокол фиксирует 16 задач по 50 эпизодов на каждую; обе модели используют одни и те же задачи, знаменатель и условия успеха среды:

```text
ArrangeTea, CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug,
DeliverStraw, OpenCabinet, OpenDrawer, PackIdenticalLunches,
PickPlaceDrawerToCounter, PortionHotDogs, PrepareCoffee,
RecycleBottlesByType, RinseSinkBasin, ScrubCuttingBoard,
SeparateFreezerRack, SlideDishwasherRack
```

| Модель | Успешных | Всего эпизодов | Успешность |
| --- | ---: | ---: | ---: |
| GR00T N1.5 | 230 | 800 | 28,75% |
| Pi0.5 | 142 | 800 | 17,75% |

## 8. Длинное видео с четырёх видов

```bash
python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/robocasa365_showcase.py" \
  --help
```

Скрипт показа сохраняет разрешение входа стратегии неизменным и повышает разрешение только камер записи. Рекомендуется записывать центральную, левую, правую камеры и камеру «глаз-рука» по 960×540, а затем склеивать их в 1920×1080 при 20 кадрах в секунду. Полный пример см. в [длинном успешном видео PackIdenticalLunches](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/robocasa-recovery/pack-success.mp4).

## 9. Ключевые моменты переноса на AMD

| Модуль | Допущение вышестоящего проекта | Решение на AMD | Способ проверки |
| --- | --- | --- | --- |
| GR00T N1.5 | PyTorch на CUDA | Установить PyTorch для ROCm, строка устройства остаётся `cuda` | Прямой проход, обратный проход, оптимизатор и сохранение модели |
| Pi0.5 | JAX на CUDA | Изолированная установка ROCm JAX | Матрица 64×64, загрузка модели и сервис стратегии |
| MuJoCo | Локальное графическое окно | `MUJOCO_GL=egl` | `reset → step → render → MP4` |
| Данные | Относительные пути в реестре | Смонтировать на диск данных, оба репозитория стратегий читают один реестр | Число записей в реестре, декодирование видео, размерность действий |
| Интерфейс действий | 12-мерное действие PandaOmron | Сохранить порядок правой руки, захвата, шасси, корпуса и режимных битов | Поэлементная проверка фиксированного действия |
| Оценка | Свой скрипт у каждой стратегии | Зарегистрировать один и тот же набор задач `amd_match16` | Сводка 16×50 с одним знаменателем |

Лучше устанавливать PyTorch, JAX и MuJoCo [симулятор динамики роботов] в отдельных окружениях. После успешной загрузки стратегии сначала проверьте интерфейс действий: нормализация действий, тип робота и частота управления определяют, соответствует ли действие в замкнутом цикле среде.

## 10. Частые проблемы

| Симптом | Что проверить |
| --- | --- |
| При создании окружения не хватает ресурсов | Повторно выполнить `download_kitchen_assets` и проверить `macros_private.py` |
| Ошибка рендеринга на удалённой машине | Задать `MUJOCO_GL=egl` и `PYOPENGL_PLATFORM=egl` |
| Модель загружается, но действия неверны | Сверить тип робота, размерность действия, статистику нормализации и частоту управления |
| Видео успеха не совпадает со статистикой | Видео должно быть из того же каталога эпизода; проверяйте поле успеха среды |
| Pi0.5 не видит устройство AMD | Переустановить ROCm JAX в изолированном окружении и повторить проверку матрицы |

## 11. Официальные материалы

- [Исходный код RoboCasa](https://github.com/robocasa/robocasa)
- [Документация RoboCasa](https://robocasa.ai/docs/introduction/overview.html)
- [RoboCasa GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T)
- [RoboCasa OpenPI](https://github.com/robocasa-benchmark/openpi)
- [Checkpoint RoboCasa365](https://huggingface.co/robocasa/robocasa365_checkpoints)
