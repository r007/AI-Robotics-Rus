# Предобученные веса и опыт без обучения

В этой главе приведены три готовых к загрузке учебных модели. Можно сначала запустить инференс и замкнутую оценку, освоить правильное поведение, входы и выходы и структуру каталога результатов, а уже потом решать, переобучать ли модель.

## Репозитории моделей

| Модель | Репозиторий Hugging Face | Назначение в курсе |
| --- | --- | --- |
| SmolVLA | [Datawhale/every-embodied-smolvla-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp) | Захват красной и синей кружки и баланс выборки |
| Pi0 | [Datawhale/every-embodied-pi0-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-pi0-mujoco-pnp) | Загрузка визуально-языковой модели действий, продолжение обучения и диагностика |
| ACT | [Datawhale/every-embodied-act-mujoco-pnp](https://huggingface.co/Datawhale/every-embodied-act-mujoco-pnp) | Лёгкое обучение, базовая замкнутая оценка и эксперименты DAgger |

Репозитории моделей содержат веса для инференса, конфигурацию модели и конфигурацию обучения. Состояние оптимизатора, кэш и пакетные видео для опыта без обучения не нужны.

## Загрузка

```bash
export MODEL_ROOT=${MODEL_ROOT:-$HOME/physical-ai/checkpoints}
mkdir -p "$MODEL_ROOT"

hf download Datawhale/every-embodied-smolvla-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-smolvla"

hf download Datawhale/every-embodied-pi0-mujoco-pnp \
  weights/model.safetensors weights/config.json weights/train_config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-pi0"

hf download Datawhale/every-embodied-act-mujoco-pnp \
  weights/model.safetensors weights/config.json \
  --repo-type model --local-dir "$MODEL_ROOT/every-embodied-act"
```

После загрузки убедитесь, что файл открывается через `safetensors`:

```bash
export POLICY_PATH="$MODEL_ROOT/every-embodied-smolvla/weights"
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
import os

path = Path(os.environ["POLICY_PATH"]) / "model.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
print("tensor_count:", len(keys))
print("config_exists:", (path.parent / "config.json").is_file())
PY
```

## Воспроизведение четырёх видов

Откройте [11_mujoco_closed_loop_deploy.ipynb](./notebooks/11_mujoco_closed_loop_deploy.ipynb) и выполните ячейку «предпросмотр успеха без обучения». Она воспроизводит:

```text
assets/pnp_four_view_strict_success.mp4
```

По этому видео изучаются фазы действия: подход, захват, подъём, перенос, освобождение и устойчивая укладка. Затем загружается модель, запускаются новые замкнутые эпизоды, а результаты записываются в собственный `$RUN_ROOT`.

## Загрузка и оценка

На примере SmolVLA:

```bash
export POLICY_TYPE=smolvla
export POLICY_PATH="$MODEL_ROOT/every-embodied-smolvla/weights"
export EVAL_SEEDS=1000,1001,1002,1003
export OUTPUT_ROOT="$RUN_ROOT/zero_train/smolvla"
python code/run_closed_loop.py
```

Для Pi0 и ACT измените `POLICY_TYPE` на `pi0` или `act` и укажите `POLICY_PATH` на каталог `weights/` соответствующей модели.

После запуска проверьте:

```text
$OUTPUT_ROOT/
├── results.jsonl
├── summary.json
└── videos/
```

Загрузку модели, версии окружения, random seed и условия успеха сохраняйте в одном каталоге запуска, чтобы потом сравнивать результаты с моделью, переобученной самостоятельно.

## Индекс результатов курса

| Модель | Результат по фиксированному протоколу | Материалы |
| --- | ---: | --- |
| SmolVLA weighted500 | Красная `27/30`, синяя `30/30`, всего `57/60` | Сводка оценки, видео успеха и неудачи, Notebook 14 |
| Pi0 protected clean40 | strict `12/14`; unseen `9/10`; hard `6/8` | Сводка оценки, видео диагностики, Notebook 15 |
| ACT protected repair15 | strict `15/30` | Результаты по трём группам фиксированных random seed, Notebook 16 |

Значения в таблице соответствуют протоколу, сохранённому курсом. Если вы запускаете оценку сами, ориентируйтесь на новые результаты по эпизодам и на текущую конфигурацию окружения.

## Продолжение обучения

Чтобы продолжить обучение, укажите каталог загруженной модели как `pretrained_path` и используйте новый выходной каталог:

```text
base model: $MODEL_ROOT/every-embodied-<model>/weights
run output: $RUN_ROOT/<model>/<run_name>
```

Записывайте в конфигурацию обучения:

- версию набора данных, число эпизодов и распределение задач;
- ключи наблюдений, размерность состояния и действия;
- частоту управления, длину блока действий и способ нормализации;
- размер батча, скорость обучения, число шагов и интервал сохранения;
- версии GPU, ROCm и фреймворка;
- задачи оценки, random seed и условие физического успеха.

## Рекомендации по хранению

Сохраняйте веса модели, конфигурации, метрики обучения и результаты официальной оценки. Кэш загрузки можно пересоздать, поэтому при нехватке места сначала удаляйте кэш и повторные промежуточные экспорты. Крупные файлы не должны попадать в Git-репозиторий учебника.
