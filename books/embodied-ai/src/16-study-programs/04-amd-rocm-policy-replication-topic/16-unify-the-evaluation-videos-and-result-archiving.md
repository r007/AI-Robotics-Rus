# Единая оценка, видео и архивация результатов

В этой главе результаты обучения, оценки и видео разных бенчмарков сводятся к единой процедуре проверки. Цель в том, чтобы по одному каталогу результатов можно было установить: какая модель запущена, на какой задаче, с какими seed [random seed], каков знаменатель успеха и относится ли видео к тому же эпизоду.

## 1. Проверка перед оценкой

```bash
python - <<'PY'
import json
import os
import torch

info = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "mujoco_gl": os.environ.get("MUJOCO_GL"),
}
print(json.dumps(info, indent=2))
PY
```

Одновременно записывайте хэш коммита upstream [код с коммитом], каталог модели, статистику нормализации, конфигурацию задачи и список камер.

## 2. Формат результата одного эпизода

```json
{
  "benchmark": "DexJoCo",
  "model": "pi05_dexjoco_multi_task",
  "task": "bimanual_hanoi",
  "seed": 0,
  "episode": 0,
  "success": true,
  "steps": 476,
  "elapsed_s": 47.6,
  "video": "videos/bimanual_hanoi_seed0.mp4"
}
```

Разные проекты могут добавлять поля этапов, но `task`, `seed`, `success`, `steps` и `video` должны быть одинаковыми везде.

Адаптер переводит поля каждого upstream-проекта в этот формат:

| Единое поле | Типичный источник в upstream |
| --- | --- |
| `task` | Имя среды, имя конфигурации задачи или идентификатор задачи |
| `seed` | Random seed инициализации среды |
| `success` | Поле успеха среды или официальная проверка задачи |
| `steps` | Фактическое число шагов симуляции |
| `video` | Относительный путь к видео того же эпизода |

Адаптер не переписывает условие успеха upstream, а только приводит имена полей, типы данных и расположение файлов.

## 3. Агрегация процента успеха

```python
from collections import defaultdict
import json
from pathlib import Path

rows = [json.loads(path.read_text()) for path in Path("eval").glob("**/episode.json")]
by_task = defaultdict(list)
for row in rows:
    by_task[row["task"]].append(bool(row["success"]))

summary = {
    task: {
        "successes": sum(values),
        "episodes": len(values),
        "success_rate": sum(values) / len(values),
    }
    for task, values in sorted(by_task.items())
}
Path("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
```

Не выводите результат по тексту в имени файла; лучше читать поле успеха среды или официальный `eval_info.json`.

## 4. Синхронная съёмка с нескольких камер

```python
camera_names = ["center", "left", "right", "wrist"]
frames = {name: [] for name in camera_names}

observation = env.reset()
for _ in range(max_steps):
    action = policy.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    for name in camera_names:
        frames[name].append(env.render(camera_name=name))
    if terminated or truncated:
        break
```

Все камеры должны записываться на одном и том же шаге симуляции [time step]. Видео одной камеры используется для диагностики, а сводное многокамерное — для просмотра.

## 5. Кодирование видео

```bash
ffmpeg -y -i input.mp4 \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p -movflags +faststart output.mp4
```

Проверка полного декодирования:

```bash
ffmpeg -v error -i output.mp4 -f null -
ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of json output.mp4
```

## 6. Проверка выходов обучения

Каталог обучения должен содержать как минимум:

```text
train.log
metrics.jsonl
run_config.json
checkpoints/{step}/pretrained_model/
```

Проверка последних записей обучения и checkpoint:

```bash
tail -n 20 "$RUN_DIR/train.log"
tail -n 5 "$RUN_DIR/metrics.jsonl"
find "$RUN_DIR/checkpoints" -maxdepth 3 -type f | head
```

После обучения оценивайте именно тот checkpoint, который только что был создан, и записывайте фактический путь к модели в результат каждого эпизода и в конфигурацию запуска.

## 7. Перечень результатов по бенчмаркам

| Бенчмарк | Формальный протокол | Вход в учебнике |
| --- | --- | --- |
| Every Embodied | 30 seed для красной и синей задачи | Задачи 02–13 |
| RoboCasa365 | 16 задач × 50 эпизодов | [Учебник RoboCasa365](./11-robocasa365-rocm-download-training-and-evaluation.md) |
| DexJoCo | 11 задач с официальными фиксированными seed | [Учебник DexJoCo](./12-dexjoco-pi05-rocm-jax-migration-training-and-evaluation.md) |
| DISCOVERSE | Проверки задач, экспертные данные и замкнутая политика считаются раздельно | [Учебник DISCOVERSE](./13-discoverse-rocm-data-generation-training-and-multi-view-video.md) |
| RoboWits | 10 задач × 50 seed/mutation [исходные/мутированные условия] | [Учебник RoboWits](./14-robowits-rocm-download-training-and-creative-task-evaluation.md) |
| Unitree G1 | Фиксированные seed бросков мяча и безопасный зазор | [Учебник Unitree G1](./15-unitree-g1-prediction-cbf-rocm-reproduction.md) |

## 8. Итоговая структура архива

```text
release/{benchmark}/{model}/
├── README.md
├── environment.txt
├── run_config.json
├── train_metrics.jsonl
├── checkpoint_link.txt
├── eval_summary.json
├── per_episode/
└── videos/
```

В `README.md` описываются команды загрузки, обучения и оценки; в `checkpoint_link.txt` записывается адрес для скачивания модели. Большие модели, данные и кэш в Git-репозиторий не коммитятся.

## 9. Критерии завершённости

Учебник по бенчмарку считается завершённым, если одновременно выполнены условия:

1. на новой машине по документу можно загрузить исходный код, ресурсы, данные и модели;
2. среда выполняет цикл `reset → step → render → MP4`;
3. вход обучения даёт реальные обновления градиентов и checkpoint;
4. оценка читает этот checkpoint и выводит полный знаменатель;
5. процент успеха можно заново вычислить по файлам результатов по эпизодам;
6. успешные и типичные неудачные видео можно сопоставить с соответствующими эпизодами.
