# Загрузка бенчмарков и единый каталог

Эта глава готовит хранилище, авторизацию, докачку и каталог запусков, общие для всех бенчмарков. Последующие главы о RoboCasa365, DexJoCo, DISCOVERSE, RoboWits и Unitree G1 используют эти переменные, поэтому данные, checkpoint и кэш не попадают в репозиторий с исходным кодом.

## 1. Рекомендуемый каталог

Сначала перейдите в корень репозитория Every Embodied и задайте каталоги учебника и крупных файлов:

```bash
export EVERY_EMBODIED_ROOT="$(git rev-parse --show-toplevel)"
export WORK_ROOT=${WORK_ROOT:-$HOME/physical-ai}

# Для постоянного PVC в облаке разработчика AMD:
# export WORK_ROOT=/workspace/physical-ai

# Если образ AUP сохраняет /home/jovyan, можно использовать:
# export WORK_ROOT=/home/jovyan/physical-ai
export SRC_ROOT=$WORK_ROOT/src
export DATA_ROOT=$WORK_ROOT/datasets
export MODEL_ROOT=$WORK_ROOT/checkpoints
export RUN_ROOT=$WORK_ROOT/runs
export CACHE_ROOT=$WORK_ROOT/cache
export ENV_ROOT=$WORK_ROOT/envs
export HF_HOME=$CACHE_ROOT/huggingface
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export UV_CACHE_DIR=$CACHE_ROOT/uv

test -f "$EVERY_EMBODIED_ROOT/README.md"
mkdir -p "$SRC_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT" "$ENV_ROOT" \
  "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

В каталоге исходного кода хранятся только код и небольшие конфигурации. Данные, модели, видео и результаты обучения лежат в отдельных каталогах, поэтому кэш можно удалять, не затрагивая результаты обучения.

## 2. Проверка устройства AMD

PyTorch на ROCm по-прежнему использует интерфейс устройства `torch.cuda`; чтобы определить бэкенд, проверяйте и `torch.version.hip`.

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
assert torch.version.hip is not None
PY
```

Для проекта JAX выполняется отдельная проверка матрицы:

```bash
JAX_PLATFORMS=rocm python - <<'PY'
import jax
import jax.numpy as jnp

print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
x = jnp.ones((64, 64), dtype=jnp.float32)
print("sum:", float((x @ x).sum().block_until_ready()))
assert jax.default_backend() == "rocm"
PY
```

## 3. Авторизация и загрузка Hugging Face

Открытые модели загружаются напрямую. Закрытые наборы данных сначала требуют заявки на доступ на сайте, после чего выполняется вход в терминале. Токен [access token] сохраняется только на локальной машине и не записывается в скрипты или Notebook [интерактивный блокнот].

```bash
hf auth login
hf auth whoami
```

Для загрузки используйте `hf download` с параметром `--local-dir`: если загрузка прервалась, повторите ту же команду, и она продолжится. Сначала задайте репозиторий и локальный каталог, затем запустите команды:

```bash
export HF_MODEL_REPO=robocasa/robocasa365_checkpoints
export HF_DATASET_REPO=DexJoCo/DexJoCo-Datasets-LeRobot
export LOCAL_MODEL_DIR="$MODEL_ROOT/robocasa365"
export LOCAL_DATASET_DIR="$DATA_ROOT/dexjoco-lerobot"

hf download "$HF_MODEL_REPO" \
  --repo-type model \
  --local-dir "$LOCAL_MODEL_DIR"

hf download "$HF_DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$LOCAL_DATASET_DIR"
```

Если нужна только одна папка из репозитория, используйте `--include`:

```bash
export HF_SUBDIR='pi05_pretrain_human300_multitask_75000/**'
hf download "$HF_MODEL_REPO" \
  --include "$HF_SUBDIR" \
  --local-dir "$LOCAL_MODEL_DIR"
```

## 4. Безголовый рендеринг

Если на удалённом устройстве нет окна рабочего стола, используйте EGL [интерфейс рендеринга без окна]:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
```

Минимальная проверка означает не то, что программа импортируется, а то, что проходит полный цикл `reset → step → render → MP4`:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 "$RUN_ROOT/robocasa365/smoke/episode_000.mp4"
```

## 5. Два уровня запуска протокола

Каждый бенчмарк разделён на два уровня:

| Уровень | Цель | Минимальный результат |
| --- | --- | --- |
| smoke [дымовой тест] | Проверить окружение, ресурсы, размерность действий и цепочку видео | 1 задача, 1 короткий эпизод, MP4, информация об окружении |
| formal evaluation [формальная оценка] | Оценить возможности модели | Фиксированные задачи, фиксированные seed [random seed], полный знаменатель, результаты по эпизодам |

Перед обучением проходите smoke-тест, но официальные результаты собирайте только из полного каталога оценки.

## 6. Единый каталог запусков

```text
runs/{benchmark}/{model}/{run_name}/
├── run_config.json
├── environment.txt
├── train.log
├── metrics.jsonl
├── checkpoints/
├── eval/
│   └── {task}/
│       ├── eval_info.json
│       └── videos/
└── summary.json
```

В `run_config.json` должны быть как минимум: версия вышестоящего проекта, задача, модель, размерности действий и наблюдений, камеры, частота управления, число шагов обучения и seed оценки. Файл `summary.json` обязательно агрегируется из результатов по эпизодам; процент успеха вписывать вручную нельзя.

## 7. Проверка завершения загрузки

```bash
export PROJECT_DIR="$SRC_ROOT/robocasa"
export DATASET_DIR="$DATA_ROOT/dexjoco-lerobot"
export CHECKPOINT_DIR="$MODEL_ROOT/robocasa365"

test -d "$PROJECT_DIR/.git"
test -n "$(find "$DATASET_DIR" -type f -print -quit)"
test -n "$(find "$CHECKPOINT_DIR" -type f -print -quit)"
df -h "$WORK_ROOT"
```

Убедитесь, что каталоги исходного кода, данных и моделей не пусты, и оставьте место для checkpoint обучения и видео, прежде чем переходить к соответствующей главе.

## 8. Дальнейшие главы

- [RoboCasa365: ресурсы, обучение стратегий и оценка 16 задач](./11-robocasa365-rocm-download-training-and-evaluation.md)
- [DexJoCo: Pi0.5, нативный ROCm JAX и оценка 11 задач](./12-dexjoco-pi05-rocm-jax-migration-training-and-evaluation.md)
- [DISCOVERSE: экспертные данные, обучение стратегий и мультивидовое видео](./13-discoverse-rocm-data-generation-training-and-multi-view-video.md)
- [RoboWits: ограниченные данные, ACT/Pi0/Pi0.5 и оценка с мутациями](./14-robowits-rocm-download-training-and-creative-task-evaluation.md)
- [Unitree G1: воспроизведение безопасного управления с предсказательным CBF](./15-unitree-g1-prediction-cbf-rocm-reproduction.md)
- [Единая оценка, видео и архивация результатов](./16-unify-the-evaluation-videos-and-result-archiving.md)
