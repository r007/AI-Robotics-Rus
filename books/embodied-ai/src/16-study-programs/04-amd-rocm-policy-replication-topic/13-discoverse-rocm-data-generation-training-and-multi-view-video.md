# DISCOVERSE: генерация данных, обучение и многовидовое видео на ROCm

DISCOVERSE — это фреймворк симуляции для обучения роботов по схеме Real2Sim2Real [от реальности к симуляции и обратно], охватывающий AIRBOT, MMK2, экспертные конечные автоматы, имитационное обучение, LiDAR и 3DGS (3D Gaussian Splatting). В этой главе мы начинаем с загрузки исходного кода и моделей, создаём экспертные данные, конвертируем их в формат LeRobot, обучаем ACT и Diffusion Policy [диффузионная стратегия], выполняем замкнутую оценку и экспортируем видео с трёх камер в высоком разрешении.

## Результаты главы

- записи о запуске сцен AIRBOT и MMK2;
- успешные экспертные траектории и список сгенерированных эпизодов;
- набор данных LeRobot, соответствие действий и наблюдений и видео воспроизведения;
- модель ACT или Diffusion Policy, результаты замкнутой оценки и многовидовое видео;
- необязательный динамический рендеринг 3DGS.

После установки зафиксируйте версию вышестоящего проекта:

```bash
git -C "$SRC_ROOT/DISCOVERSE" rev-parse HEAD
```

## 1. Загрузка и установка

```bash
cd "$SRC_ROOT"
git lfs install
git clone https://github.com/discoverse-dev/DISCOVERSE.git
cd DISCOVERSE

export DISCOVERSE_ENV="$ENV_ROOT/discoverse-rocm"
python3.10 -m venv "$DISCOVERSE_ENV"
source "$DISCOVERSE_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/setup_submodules.py
python scripts/check_installation.py
```

Дополнительные зависимости ставятся по назначению:

```bash
python -m pip install -e '.[act_full]'
python -m pip install -e '.[lidar,visualization]'
python -m pip install -e '.[gs]'
```

Модели 3DGS (3D Gaussian Splatting) публикуются в `tatp/DISCOVERSE-models`. При первом запуске они скачиваются автоматически, но можно выполнить вход заранее:

```bash
hf auth login
```

## 2. Базовая проверка запуска

```bash
export MUJOCO_GL=egl
python discoverse/robots_env/airbot_play_base.py
python discoverse/robots_env/mmk2_base.py

python examples/mocap_ik/mocap_ik_manipulator.py \
  -r airbot_play \
  -t place_block \
  --record \
  --camera-names global_camera wrist_camera
```

Приёмка миграции охватывает 18 проверок запуска, 12 задач AIRBOT и 8 фиксированных сцен MMK2. Генерировать обучающие данные нужно только после того, как базовые проверки пройдены.

## 3. Генерация экспертных траекторий

Официальные скрипты задач уже содержат экспертный конечный автомат. Для примера с укладкой кубика на AIRBOT:

```bash
python examples/tasks_airbot_play/place_block.py --help
```

Экспертный конечный автомат должен одновременно соблюдать кинематические и физические ограничения. Если возникает недостижимость IK [обратная кинематика], сначала проверьте, не выходит ли целевая поза за рабочее пространство манипулятора, не приводит ли ориентация концевого звена к пределам суставов и нет ли скачков позы между соседними состояниями. Для задач укладки стоит разбить путь эксперта на фазы: предзахват, захват, подъём, предукладка, опускание, освобождение и отход; между соседними состояниями выполнять интерполяцию. Если опорный объект опрокидывается, меняйте направление подхода и отхода, а не увеличивайте массу объекта.

Названия параметров могут различаться в разных версиях, поэтому сначала смотрите `--help`. При пакетной генерации фиксируйте каталог данных, диапазон seed [random seed] и требуемое число успешных эпизодов:

```bash
export DISCOVERSE_DATA_ROOT="$DATA_ROOT/discoverse"
mkdir -p "$DISCOVERSE_DATA_ROOT/place_block"

for seed in $(seq 0 499); do
  python examples/tasks_airbot_play/place_block.py \
    --data_idx "$seed" \
    --data_set_size 1 \
    --auto
done

rsync -a data/place_block/ "$DISCOVERSE_DATA_ROOT/place_block/"
```

Официальный скрипт сначала пишет данные в `data/place_block` внутри каталога исходного кода DISCOVERSE; последняя строка переносит на диск данных большого объёма только те эпизоды, которые прошли проверку успеха. Перед повторным запуском с тем же seed проверьте, не существует ли уже целевой эпизод, чтобы не генерировать его дважды.

Каждую траекторию проверяйте минимум по следующим пунктам:

- условие успеха в симуляции истинно;
- число состояний, действий и кадров камер совпадает;
- временные метки строго возрастают;
- хвостовое заполнение действий не переходит в следующий эпизод;
- при воспроизведении захват, перенос и укладка выполняются полностью.

Архивный корпус экспертных данных `place_block` содержит 500 принятых эпизодов и 44 439 выровненных кадров.

## 4. Конвертация в обучающие данные

Исходные данные DISCOVERSE обычно включают `obs_action.json`, изображения или видео камер и метаданные задачи. При конвертации в LeRobot [библиотека робототехнического обучения] создаются единые поля:

```text
observation.state
observation.images.{camera}
action
task
episode_index
frame_index
timestamp
```

Сначала конвертируйте одну траекторию, затем пакет:

```bash
python "$EVERY_EMBODIED_ROOT/<topic-folder>/code/convert_discoverse_to_lerobot.py" \
  --input-root "$DISCOVERSE_DATA_ROOT/place_block" \
  --output-root "$DATA_ROOT/discoverse-lerobot/place_block" \
  --repo-id local/discoverse-place-block \
  --task 'Place the green block in the pink bowl' \
  --fps 20 \
  --image-size 256
```

Здесь `<topic-folder>` — каталог этой темы в исходном репозитории `every-embodied`.

Конвертер читает `obs_action.json`, `cam_0.mp4` и `cam_1.mp4` каждого успешного эпизода и проверяет, что оба изображения и 7-мерные суставные состояния и 7-мерные действия имеют одинаковую длину покадрово. DISCOVERSE записывает наблюдение после выполнения действия, поэтому по умолчанию используется `--action-bias=-1`, которое выравнивает момент наблюдения со следующим действием. В файле `discoverse_conversion_manifest.json` записываются исходные эпизоды, число кадров и диапазоны индексов действий. Не ограничивайтесь сравнением форм массивов: проверьте временную согласованность, воспроизведя действия.

## 5. Обучение ACT

```bash
lerobot-train \
  --dataset.repo_id=local/discoverse-place-block \
  --dataset.root="$DATA_ROOT/discoverse-lerobot/place_block" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --steps=100000 \
  --batch_size=32 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/discoverse/act_place_block"
```

На ROCm [открытая платформа вычислений AMD] запись `--policy.device=cuda` остаётся правильной. Сначала запустите 10–50 шагов, чтобы проверить данные и обратное распространение, и только потом начинайте полное обучение.

## 6. Обучение Diffusion Policy

```bash
lerobot-train \
  --dataset.repo_id=local/discoverse-place-block \
  --dataset.root="$DATA_ROOT/discoverse-lerobot/place_block" \
  --policy.type=diffusion \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=16 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/discoverse/diffusion_place_block"
```

Точное имя стратегии зависит от версии LeRobot, поэтому ориентируйтесь на вывод `lerobot-train --help`. В конфигурации обучения обязательно сохраняйте отображение наблюдений, нормализацию действий и временное окно.

## 7. Замкнутая оценка

Цикл оценки должен реально продвигать симулятор действиями стратегии:

```python
observation = env.reset()
for step in range(max_steps):
    action = policy.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    recorder.append(observation, action, info)
    if terminated or truncated:
        break
```

Зафиксируйте два набора seed: похожий на обучающий и неизвестный. Для каждого отдельно подсчитывайте приближение, захват, подъём, перенос, выравнивание и итоговую укладку. Результаты оценки и видео записываются в один каталог эпизода.

## 8. Многовидовой вывод в высоком разрешении

Задача `kiwi_pick.py` для MMK2 по умолчанию уже записывает три камеры, но с разрешением 640×480. Скопируйте скрипт задачи и измените только разрешение записи — официальный конечный автомат эксперта при этом сохраняется:

```bash
cd "$SRC_ROOT/DISCOVERSE"
mkdir -p "$RUN_ROOT/discoverse/mmk2_kiwi_hd"
cp examples/tasks_mmk2/kiwi_pick.py \
  "$RUN_ROOT/discoverse/mmk2_kiwi_hd/kiwi_pick_hd.py"

python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["RUN_ROOT"]) / "discoverse/mmk2_kiwi_hd/kiwi_pick_hd.py"
text = path.read_text()
text = text.replace('"width"  : 640', '"width"  : 1920')
text = text.replace('"height" : 480', '"height" : 1080')
path.write_text(text)
PY

python "$RUN_ROOT/discoverse/mmk2_kiwi_hd/kiwi_pick_hd.py" \
  --data_idx 0 \
  --data_set_size 1 \
  --auto
```

После успешного выполнения три исходных видео лежат в `data/pick_kiwi/000/cam_0.mp4`, `cam_1.mp4` и `cam_2.mp4`. Скрипт записывает их на диск только после выполнения условий успеха, поэтому видео и соответствующие экспертные действия относятся к одному эпизоду.

Чтобы сохранить исходные видео по одной камере и получить сводное видео с трёх видов:

```bash
ffmpeg \
  -i data/pick_kiwi/000/cam_0.mp4 \
  -i data/pick_kiwi/000/cam_1.mp4 \
  -i data/pick_kiwi/000/cam_2.mp4 \
  -filter_complex '[0:v][1:v][2:v]hstack=inputs=3[v]' \
  -map '[v]' -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$RUN_ROOT/discoverse/mmk2_kiwi_hd/composite.mp4"
```

## 9. Рендеринг 3DGS

```bash
python -m pip install -e '.[gs]'
hf auth login
```

В сценах, поддерживающих Gaussian renderer [гауссов рендерер], задайте `cfg.use_gaussian_renderer = True`; в интерактивном окне переключение выполняется сочетанием `Ctrl+g`. При проверке сохраняйте:

- путь к фактически загруженному облаку точек `.ply`;
- число точек и разрешение рендеринга;
- минимум один динамический повтор;
- сетку MuJoCo для той же задачи как контрольный вариант.

## 10. Ключевые моменты переноса на AMD

| Модуль | Что переносится |
| --- | --- |
| Стратегии PyTorch | Интерфейс `torch.cuda` сохраняется, устанавливается колесо ROCm |
| MuJoCo | Вывод видео через EGL [интерфейс рендеринга без окна] |
| 3DGS (3D Gaussian Splatting) | Замена расширений, специфичных для CUDA, или использование реализации, собирающейся под ROCm |
| Данные | Исправление границ эпизодов, блоков действий и временной синхронизации камер |
| ACT | Длина блока действий одинакова при обучении и выполнении |
| Несколько роботов | Отдельная проверка MJCF робота, контроллера и условия успеха |

## 11. Что считается результатом

Запуск среды, успешность экспертов и успешность обученной стратегии фиксируются раздельно. Примеры результатов миграции: проверки запуска `18/18`, AIRBOT `12/12`, MMK2 `8/8`, задачи высокой чёткости `4/4` и 500 экспертных траекторий. Модели обучения оцениваются по собственным замкнутым задачам и собственным знаменателям.

## 12. Официальные материалы

- [Исходный код DISCOVERSE](https://github.com/discoverse-dev/DISCOVERSE)
- [Страница проекта DISCOVERSE](https://air-discoverse.github.io/)
- [Модели DISCOVERSE](https://huggingface.co/tatp/DISCOVERSE-models)
