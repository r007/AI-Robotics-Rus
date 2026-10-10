# DexJoCo: Pi0.5, ROCm JAX и воспроизведение 11 задач

DexJoCo — это бенчмарк ловких манипуляций на основе MuJoCo, который охватывает 11 задач: одноруч­ные, двуручные, с инструментами и долгие последовательности. В этой главе симулятор и OpenPI/Pi0.5 разнесены в два изолированных окружения и связаны через WebSocket [сетевой протокол]. Мы выполняем нативный вывод AMD JAX [высокопроизводительная среда вычислений AMD], конвертацию данных, обучение и замкнутую оценку с записью видео.

## Результаты главы и задачи

| Задача | Тип |
| --- | --- |
| `click_mouse`, `fold_glasses`, `hammer_nail`, `pick_bucket`, `pinch_tongs`, `water_plant` | Одноручная или инструментальная операция |
| `bimanual_assembly`, `bimanual_hanoi`, `bimanual_microwave_cook`, `bimanual_photograph`, `bimanual_unlock_ipad` | Двуручная координация и долгие операции |

Одноручная стратегия выдаёт 22-мерное действие, двуручная — 44-мерное. Признак `success` для каждой задачи формируется условиями завершения официальной среды: отношения целевых объектов, состояние инструмента или итоговое состояние задачи. Результаты по эпизодам сохраняют имя задачи, random seed, число шагов, поле успеха и видео.

После установки зафиксируйте версию исходного кода:

```bash
git -C "$SRC_ROOT/dexjoco" rev-parse HEAD
```

## 1. Загрузка исходного кода, данных и моделей

```bash
cd "$SRC_ROOT"
git clone https://github.com/brave-eai/dexjoco.git
cd dexjoco

conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

Наборы данных LeRobot [библиотека робототехнического обучения] и модель Pi0.5:

```bash
hf download DexJoCo/DexJoCo-Datasets-LeRobot \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/dexjoco-lerobot"

hf download DexJoCo/DexJoCo-Pi05 \
  --include 'pi05_dexjoco_multi_task/**' \
  --local-dir "$MODEL_ROOT/dexjoco-pi05"
```

Чтобы сэкономить место, загружайте только многозадачную модель. Веса для отдельной задачи и для визуальной рандомизации `rand_full` скачивайте по мере необходимости.

## 2. Проверка среды симулятора и 11 задач

```bash
export MUJOCO_GL=egl
python scripts/amd_rocm_smoke.py \
  --output "$RUN_ROOT/dexjoco/smoke_11_tasks"
```

В выходном каталоге должны быть `smoke_report.json` и по одному MP4 на каждую из 11 задач. Эта проверка подтверждает среду, интерфейс действий и цепочку рендеринга; оценку стратегии выполняем на следующих шагах.

## 3. Нативное окружение ROCm JAX без Docker

Окружение Python 3.12 для OpenPI отделено от окружения Python 3.11 для симулятора:

```bash
export ROCM_JAX010_ENV="$ENV_ROOT/openpi-amd-jax010"
python3.12 -m venv "$ROCM_JAX010_ENV"
export PIP_NO_CACHE_DIR=1

"$ROCM_JAX010_ENV/bin/python" -m pip install --upgrade pip
"$ROCM_JAX010_ENV/bin/python" -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm[libraries,device-gfx1151]==7.14.0'
"$ROCM_JAX010_ENV/bin/python" -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'jax_rocm7_plugin==0.10.0+rocm7.14.0' \
  'jax_rocm7_pjrt==0.10.0+rocm7.14.0'
"$ROCM_JAX010_ENV/bin/python" -m pip install 'jax==0.10.0' 'jaxlib==0.10.0'
```

После этого выполните предварительную проверку матрицы JAX из главы [загрузка бенчмарков и единый каталог](./10-simulation-benchmark-download-and-unified-directory.md). Разные версии ROCm JAX не устанавливайте в одно окружение.

## 4. Сбор и воспроизведение данных

Интерактивный сбор успешных демонстраций:

```bash
conda activate dexjoco
python scripts/record_demos_zarr.py \
  --exp_name=water_plant \
  --successes_needed=20 \
  --randomize=True \
  --out_dir="$DATA_ROOT/dexjoco-raw/water_plant"
```

Воспроизведите данные и проверьте начальные состояния, действия и видео:

```bash
python scripts/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir="$DATA_ROOT/dexjoco-raw/water_plant" \
  --out_dir="$RUN_ROOT/dexjoco/replay/water_plant" \
  --randomize=True \
  --restore_state=True
```

## 5. Конвертация в данные LeRobot

```bash
bash dexjoco-data-converter/install.bash
conda activate dexjoco-dc

dexjoco-dc-single-lerobot \
  --input "$DATA_ROOT/dexjoco-raw/water_plant" \
  --output "$DATA_ROOT/dexjoco-lerobot-local/water_plant" \
  --language-instruction 'Grasp the watering can and apply water to the plant.' \
  --selected-data-yaml '{action: action_rotvec, state: state, cameras: {front: front, wrist: wrist}}' \
  --slice-yaml '{state: [null, 23]}'
```

Одноручное действие стратегии имеет размерность 22, двуручное — 44 (вектор поворота). Обёртка симулятора переводит их во внутренние кватернионные действия. После конвертации проверьте границы эпизодов, число кадров камер, размерность состояния и размерность действий.

## 6. Обучение Pi0.5

```bash
cd "$SRC_ROOT/dexjoco/openpi"
bash install.bash
```

Перед обучением выполните по порядку:

1. В `openpi/config.yaml` укажите `dataset_root` на `$DATA_ROOT/dexjoco-lerobot`, а `pretrained_model_path`, `pretrained_model_action_dim_44_path` и `ckpts_root` на реальные каталоги checkpoint.
2. Для двуручных задач запустите `convert_to_action_dim_44_model.py`.
3. Запустите `compute_norm_stats.py`, чтобы получить статистику нормализации.
4. Запустите `scripts/train.py` для обучения на одной задаче или на нескольких.

```bash
export DEXJOCO_CONFIG=multi_task

JAX_PLATFORMS=rocm \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$ROCM_JAX010_ENV/bin/python" scripts/compute_norm_stats.py "$DEXJOCO_CONFIG"

JAX_PLATFORMS=rocm \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$ROCM_JAX010_ENV/bin/python" scripts/train.py "$DEXJOCO_CONFIG" \
  --exp-name=dexjoco_pi05_amd
```

`multi_task` читает три камеры, данные нескольких задач и использует 44-мерное двуручное действие. Конфигурация для одной задачи называется так же, как сама задача: например, `water_plant`, `hammer_nail`, `fold_glasses` или `bimanual_hanoi`. Полный список конфигураций можно вывести так:

```bash
rg 'name="' src/openpi/training/dexjoco_configs.py
```

В официальном описании одноручная LoRA [низкоранговая адаптация] обычно обучается 30k шагов, двуручное и многозадачное обучение — 60k шагов. Локальное число шагов нужно выбирать по валидационной выборке и замкнутой оценке.

## 7. Формальная оценка 11 задач

Публичный адаптационный скрипт:

```bash
git clone https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite.git \
  "$SRC_ROOT/radeon-physical-ai-evidence-suite"

DEXJOCO_ROOT="$SRC_ROOT/dexjoco" \
CHECKPOINT="$MODEL_ROOT/dexjoco-pi05/pi05_dexjoco_multi_task" \
OPENPI_PYTHON="$ROCM_JAX010_ENV/bin/python" \
DEXJOCO_EVAL="$CONDA_PREFIX/bin/dexjoco-openpi-eval" \
OUT="$RUN_ROOT/dexjoco/pi05_multitask_seed0" \
  bash "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/run_dexjoco_pi05_multitask_eval.sh"
```

Скрипт для каждой задачи запускает соответствующий сервис стратегии, использует фиксированный seed 0 и по одному эпизоду на задачу, а также сохраняет результаты и видео. По тому же протоколу мы получили результат `5/11`.

Чтобы найти воспроизводимые успешные действия для неуспешных задач, можно перебрать фиксированные seed 1–10 и остановиться на первом успехе:

```bash
DEXJOCO_ROOT="$SRC_ROOT/dexjoco" \
CHECKPOINT="$MODEL_ROOT/dexjoco-pi05/pi05_dexjoco_multi_task" \
OPENPI_PYTHON="$ROCM_JAX010_ENV/bin/python" \
BASE_OUT="$RUN_ROOT/dexjoco/pi05_multitask_seed0" \
OUT="$RUN_ROOT/dexjoco/pi05_multitask_recovery" \
  python "$SRC_ROOT/radeon-physical-ai-evidence-suite/scripts/run_dexjoco_pi05_multitask_recovery.py"
```

Этот архив успешных примеров охватывает 10 из 11 задач и нужен для изучения хода выполнения. Официальный результат по seed 0 хранится отдельно.

## 8. Важнейшие интерфейсы при переносе

| Уровень | Решение на AMD |
| --- | --- |
| Симулятор MuJoCo | Python 3.11, NumPy 1.26, EGL [интерфейс рендеринга без окна] |
| OpenPI | Python 3.12, ROCm JAX [AMD JAX-стек] 0.10 |
| Связь процессов | Сервис стратегии по WebSocket [сетевой протокол] |
| Checkpoint | Orbax [система checkpoint JAX] читает и старые, и новые объекты метаданных |
| Входное состояние | Дополняется до 46 измерений по требованию модели, затем срезается конфигурацией |
| Действия | Вектор поворота 22/44 измерения переводится в кватернионные действия симулятора |

Изоляция трёх окружений — ключ к стабильности воспроизведения: симулятор, конвертер данных и OpenPI не используют общий набор зависимостей.

## 9. Пример видео

[Полное видео двуручной Ханойской башни](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/dexjoco/recovery/bimanual-hanoi.mp4) длится 47,6 с и показывает внешний вид плюс левое и правое запястья. Все камеры записаны в одном и том же цикле симуляции, а видео связано с отметкой успеха данного эпизода.

## 10. Официальные материалы

- [Исходный код DexJoCo](https://github.com/brave-eai/dexjoco)
- [Данные DexJoCo в формате LeRobot](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot)
- [Модель DexJoCo Pi0.5](https://huggingface.co/DexJoCo/DexJoCo-Pi05)
- [Руководство по установке AMD ROCm JAX](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)
