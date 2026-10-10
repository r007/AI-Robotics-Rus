# Воспроизведение танца робота с гала-концерта Весеннего фестиваля: полный практический отчёт (версия 2026)

Этот документ фиксирует ключевые шаги репозитория — установку, поиск неисправностей и итоговый рабочий запуск. Цель — помочь новым участникам избежать большинства ошибок.

Видеоурок (сайт Datawhale): https://www.datawhale.cn/learn/content/258/6228

Видеоурок (Bilibili): https://www.bilibili.com/video/BV1gBfCBpEEb

Дополнительное видео: https://www.bilibili.com/video/BV1mjcXzLEGC/?share_source=copy_web&vd_source=cc2570142da6f3f02e6aba37bd88f313

Дополнительный документ: https://my.feishu.cn/docx/OzdvdvZ3LoSUfSxziNCcFWYFnqa?from=from_copylink

## 1. Цели проекта

- Вход: текстовое описание движения или готовое видео.
- Выход: движение робота (`robot_motion*.pkl`) и визуализация (Web / viser / MuJoCo).
- Поддерживаются траектории нескольких людей (режим all tracks) и запись видео с несколькими роботами в MuJoCo.

## 2. Основной конвейер

```text
Prompt / Video -> PromptHMR -> SMPL-X -> GMR -> Robot Motion
```

Частые команды:

```bash
python scripts/generate_video.py --model seedance --action "последовательность действий: персонаж делает четыре шага вперёд"
python scripts/extract_pose.py --project data/video_001
python scripts/convert_to_robot.py --project data/video_001 --all-tracks
python scripts/visualize.py --project data/video_001 --robot-viser --robot-all
```

Строка `--action` в первой команде — русский перевод: «Последовательность действий: персонаж делает четыре шага вперёд».

## 3. Установка окружения (рекомендуется)

### 0) Развёртывание на новой машине (сначала сделайте этот шаг)

```bash
git clone https://github.com/datawhalechina/every-embodied.git
cd every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot
cd third_party
# 3. Клонируем GMR
git clone --depth 1 https://github.com/taeyoun811/GMR.git GMR
# 4. Клонируем PromptHMR
git clone --depth 1 https://github.com/taeyoun811/PromptHMR.git PromptHMR
cd ..
```

Пояснения:

- Все команды этой главы по умолчанию выполняются в каталоге `video2robot` (путь `every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot`).

Если у вас архив с патчами, примените три патча:

```bash
git apply patches/main.patch
git -C third_party/PromptHMR apply ../../patches/prompthmr.patch
git -C third_party/GMR apply ../../patches/gmr.patch
```

Затем создайте окружения из YAML-файлов:

```bash
conda env create -f envs/gmr.yml
conda config --add envs_dirs /path/to/data/conda_envs
# Первое окружение ставится на системный диск, второе — на диск данных, чтобы не переполнить систему (системный диск 30 ГБ, диск данных 50 ГБ)
conda env create -f envs/phmr.yml
```

Если окружения с такими именами уже существуют:

```bash
conda env update -n gmr -f envs/gmr.yml --prune
conda env update -n phmr -f envs/phmr.yml --prune
```

### 1) Окружение GMR

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot
pip install -e .
# Для gmr дополнительно нужно установить:

pip install loop-rate-limiters
pip install smplx
pip install imageio
pip install mink
pip install rich
pip install imageio[ffmpeg]
```

### 2) Окружение PromptHMR (сценарий с 4090)

```bash
conda create -n phmr python=3.10 -y
conda activate phmr
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR
# bash scripts/install.sh --pt_version=2.4  — не рекомендуется использовать установку одной командой по умолчанию: в ней много ошибок
```

Если автоматическая установка не удалась, используйте ручной вариант (см. раздел «Ручная настройка» ниже).

## 4. Запуск веб-интерфейса (стабильная версия)

```bash
conda activate phmr
python -m pip install -U fastapi "uvicorn[standard]" jinja2 python-multipart

pkill -f "video2robot/visualization/robot_viser.py"
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot
export VISER_FIXED_PORT=8789
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Если в GPFree или фоновой задаче `conda run -n phmr` не находит окружение, нет прав на запись или имя окружения разбирается неверно, укажите абсолютный путь к интерпретатору:

```bash
VISER_FIXED_PORT=8789 /path/to/data/conda_envs/phmr/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Адрес для доступа: `http://localhost:8000`

После запуска сначала проверьте работу веб-сервиса через API — не ограничивайтесь тем, что `uvicorn` запустился:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/projects
curl http://127.0.0.1:8000/api/viser/status
```

Пояснения:

- Фиксированный `VISER_FIXED_PORT` нужен, чтобы iframe не получал отказ в соединении из-за случайного порта.
- Веб-интерфейс уже поддерживает китайский язык, Seedance, загрузку видео и переключение «жестяной» / цветной модели.

- Промпт для генерации видео робота в веб-версии Doubao: «Помоги сгенерировать изображение: помоги оптимизировать промпт и создай видео китайского боевого искусства, персонаж должен полностью находиться в кадре».
- Можно просто загрузить видео и запустить обработку — подробности в видеоуроке.

- При первом запуске автоматически загружаются небольшие модели torch. Если загрузка не удалась, выполните вручную:
  ```bash
  mkdir -p ~/.cache/torch/hub/checkpoints && rm -f ~/.cache/torch/hub/v0.10.0.zip.*.partial && wget -O ~/.cache/torch/hub/v0.10.0.zip https://github.com/pytorch/vision/zipball/v0.10.0 && wget -O ~/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth && ls -lh ~/.cache/torch/hub/v0.10.0.zip ~/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth
  ```

## 5. Многотрековые траектории и визуализация

### 1) Генерация движений роботов по нескольким трекам

```bash
conda activate gmr
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot
python scripts/convert_to_robot.py --project data/video_001 --all-tracks
```

### 2) Отображение всех траекторий в robot-viser

```bash
python scripts/visualize.py --project data/video_001 --robot-viser --robot-all
```

## 6. Экспорт видео в MuJoCo

### Один робот

```bash
conda activate gmr
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/GMR
python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/data/video_005/robot_motion.pkl \
  --record_video \
  --video_path /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/data/video_005/mujoco_robot.mp4
```

### Несколько роботов (новый скрипт)

```bash
python scripts/vis_robot_motion_multi.py \
  --robot unitree_g1 \
  --robot_motion_paths \
    /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/data/video_005/robot_motion_track_1.pkl \
    /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/data/video_005/robot_motion_track_2.pkl \
  --record_video \
  --max_seconds 10 \
  --camera_azimuth 0 \
  --video_path /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/data/video_005/mujoco_multi_robot_10s_front.mp4
```

Пояснения:

- Исправлена проблема «0 секунд / сверхкороткое видео»: длительность теперь строго задаётся количеством кадров.
- Исправлена проблема «видна только земля, робота не видно»: камера следует за центром группы роботов.

## 7. Трудности внедрения и перспективы симуляции в IsaacSim

### Практические трудности

1. **Точность переноса движения**: отображение позы SMPL-X, извлечённой из видео, на суставы робота (retargeting) ограничено физическими возможностями робота.
2. **Отсутствие восприятия среды**: сейчас это в основном воспроизведение действий по разомкнутому циклу, нет обратной связи в реальном времени о трении пола, контакте и препятствиях.
3. **Динамические ограничения**: движения человека в видео могут превышать крутящий момент двигателей робота или пределы равновесия. Например, в выступлении робота-ушу на весеннем гала-концерте по случаю года Лошади высокие прыжки и нагрузки предъявляют очень высокие требования к отклику сервоприводов.

### Перспективы IsaacSim

В будущем для обучения с подкреплением планируется использовать **IsaacSim**:

- **Преимущества**: очень высокая точность физической симуляции, поддержка сложных моделей контакта, трения и датчиков.
- **Ценность**: данный проект в основном решает задачу «переноса движений», а IsaacSim будет отвечать за обучение устойчивости движений.

## 8. Быстрый справочник по частым проблемам и отладке

### Q1: не найден `conda` (задача веб-интерфейса мгновенно завершается)

Исправлено в основном раннере задач репозитория: теперь автоматически определяется переменная `CONDA_EXE`.

### Q2: robot-viser иногда выдаёт `localhost refused connection`

Используйте фиксированный порт: `export VISER_FIXED_PORT=8789`.

### Q3: при записи MuJoCo ошибка «отсутствует backend imageio»

Выполните: `pip install -U "imageio[ffmpeg]"`.

### Q4: ошибки компиляции (lietorch / droid_backends)

Для PyTorch 2.4 и новее замените `.type()` на `.scalar_type()`.

### Q5: не удаётся скачать файлы — скачайте их локально

1. Этап моделей (главная проблема).
   При запуске `python scripts/extract_pose.py --project data/video_001` часто появляются ошибки одного типа:
   - invalid load key, 'v'
   - UnpicklingError
   - weights_only ... Unsupported operand 118
   - Cannot load file ... allow_pickle=False

   Это означает, что скачанный файл не может быть правильно прочитан.

2. Локальная загрузка и проверка (пример).

   1) Загрузка одного файла с Hugging Face:
   ```bash
   $env:HF_ENDPOINT="https://hf-mirror.com"
   $env:HF_HUB_DISABLE_XET="1"

   python - << 'PY'
   from huggingface_hub import hf_hub_download
   p = hf_hub_download(
       repo_id="Datawhale/spring-festival-wushu-robot-replication-model",
       filename="data/pretrain/phmr/checkpoint.ckpt",
       repo_type="model",
       local_dir=r"E:\...\model_tmp",
       force_download=True
   )
   print("saved:", p)
   PY
   ```
   Если такой код не запускается прямо в терминале, сохраните его в отдельный файл:
   ```python
   from huggingface_hub import hf_hub_download
   p = hf_hub_download(
       repo_id="Datawhale/spring-festival-wushu-robot-replication-model",
       filename="data/pretrain/phmr/checkpoint.ckpt",
       repo_type="model",
       local_dir=r"E:\...\model_tmp",
       force_download=True
   )
   print("saved:", p)
   ```
   Сохраните как `test.py` и запустите `python test.py`.

   2) Локальная проверка (обязательна).
   ```python
   import torch
   p=r"E:\...\model_tmp\data\pretrain\phmr\checkpoint.ckpt"
   with open(p,'rb') as f:
       print("first16:", f.read(16))
   obj=torch.load(p, map_location='cpu', weights_only=False)
   print("torch.load OK:", type(obj))
   ```
   Критерии проверки:
   - первые 16 байт не должны начинаться с `b'version https://'` (так выглядит LFS-указатель вместо настоящего файла);
   - `torch.load` выводит «OK» без ошибок.

   3) Проверка на сервере выполняется аналогично.

### Q6: `git lfs clone` работает очень медленно или раз за разом обрывается

Лучше запускать долгие загрузки в фоне через `tmux` и следить, что каталог загрузки правильный. Если загрузка частично выполнена, перейдите в каталог модели и выполните `git lfs pull`, чтобы продолжить. При нестабильной сети в Китае можно попробовать зеркало `hf-mirror.com`, но убедитесь, что трафик терминала действительно идёт через прокси: некоторые прокси-программы обслуживают только браузер и не действуют на удалённый терминал.

### Q7: веб-интерфейс «зависает» на 96% или 99%

Это не обязательно проблема фронтенда. Сначала проверьте журналы бэкенда, процессы задач, полноту весов модели и создание выходных файлов проекта. При нескольких людях в видео или отсутствии весов модели прогресс в интерфейсе тоже может долго не меняться.

### Q8: в GPFree `conda run -n phmr/gmr` не работает

Если окружения находятся в `/path/to/data/conda_envs/phmr` или `/path/to/data/conda_envs/gmr`, используйте абсолютные пути:

```bash
conda run -p /path/to/data/conda_envs/phmr ...
conda run -p /path/to/data/conda_envs/gmr ...
```

## 9. Рекомендации по организации репозитория (слить с основным, старый отменить)

Рекомендуется вести `video2robot` как обычный каталог основного репозитория, а не как submodule (это единственный рекомендуемый вариант на данный момент).

Причины:

- Старый репозиторий `hope5hope/video2robot` скоро будет выведен из обращения, и submodule повысит риск поломки.
- Команда хочет, чтобы учебные материалы и код находились в одном репозитории, — так новым участникам не нужно переключаться между репозиториями.
- Изменения можно сразу фиксировать, проверять и публиковать в основном репозитории.

### Зафиксированные базовые коммиты

- Основной репозиторий: `030f3410dac3cb15a2570376dca6a0f46c2d158c`
- PromptHMR: `4f8915c5b9603344c56e95fadb9a01a23ba2272d`
- GMR: `069b4fd48f440e813b2b4d69255c70f53e5f83fb`

## 10. Статус репозитория

`hope5hope/video2robot` сохранён как исторический репозиторий для справки и больше не используется как основной для разработки и поставки.

## 11. Экспорт YAML и перенос воспроизведения на другую машину

На «исходной» машине:

```bash
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot
mkdir -p envs
conda env export -n gmr > envs/gmr.yml
conda env export -n phmr > envs/phmr.yml
```

На «целевой» машине:

```bash
git clone https://github.com/datawhalechina/every-embodied.git
cd every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot
conda env create -f envs/gmr.yml
conda env create -f envs/phmr.yml
```

Затем примените три патча (если ваше воспроизведение зависит от локальных изменений):

```bash
git apply patches/main.patch
git -C third_party/PromptHMR apply ../../patches/prompthmr.patch
git -C third_party/GMR apply ../../patches/gmr.patch
```

Пояснения:

- YAML-файлы не сохраняют «незакоммиченные изменения исходного кода» и «файлы весов модели».
- Поэтому рекомендуется передавать вместе три компонента: `YAML + патчи + скрипт загрузки моделей`. Так воспроизведение удаётся чаще всего.

---

## Приложение: справочник по ручной настройке (окружение phmr)

```bash
# Клонирование основного репозитория
git clone https://github.com/datawhalechina/every-embodied.git
cd every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot


conda create -n phmr python=3.10 -y
conda activate phmr
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR
pip install -r requirements.txt

mkdir -p python_libs
git clone https://github.com/Arthur151/chumpy python_libs/chumpy
python -m pip install -e python_libs/chumpy --no-build-isolation

# Добавьте в bashrc
export PYTHONPATH=$PYTHONPATH:/path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR
echo 'export PYTHONPATH=$PYTHONPATH:/path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR' >> ~/.bashrc
source ~/.bashrc


conda install -c conda-forge eigen -y

# export CPATH="$CONDA_PREFIX/include/eigen3:${CPATH:-}" && cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR/pipeline/droidcalib && python setup.py install
# Компиляция droidcalib
source /opt/conda/etc/profile.d/conda.sh
conda activate phmr

conda install -c conda-forge eigen -y
export CPATH="$CONDA_PREFIX/include/eigen3:${CPATH:-}"

cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR/pipeline/droidcalib
python setup.py install

export LD_LIBRARY_PATH="/path/to/data/conda_envs/phmr/lib/python3.10/site-packages/torch/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/lib64:/usr/local/lib"
# Эту строку добавьте в bashrc
echo 'export LD_LIBRARY_PATH="/path/to/data/conda_envs/phmr/lib/python3.10/site-packages/torch/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/lib64:/usr/local/lib"' >> ~/.bashrc
source ~/.bashrc

# Компиляция lietorch
mkdir -p python_libs
cd python_libs
git clone https://github.com/princeton-vl/lietorch.git
cd lietorch
git submodule update --init --recursive
python setup.py install
cd ../..

# Установка detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd /path/to/data/detectron2
pip install -e . --no-build-isolation

# Установка SAM2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e . --no-build-isolation

# Адаптация кода
sed -i 's/load_video_frames, load_video_frames_from_np/load_video_frames/g' /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR/pipeline/detector/sam2_video_predictor.py

# Дополнительные пакеты pip

python -m pip install -U torch-scatter --no-build-isolation  # Эта команда работает долго, наберитесь терпения
python -m pip install -U xformers
```

Обновите скрипт загрузки:
https://github.com/hope5hope/video2robot/blob/main/%E6%96%87%E4%BB%B6%E6%9B%B4%E6%96%B0%E8%AF%B4%E6%98%8E.md

Загрузка весов модели (рекомендуемый вариант):

```bash
git-lfs install
git lfs clone https://huggingface.co/Datawhale/spring-festival-wushu-robot-replication-model
# или зеркало в Китае
git lfs clone https://hf-mirror.com/Datawhale/spring-festival-wushu-robot-replication-model
```

Загрузка больших моделей занимает много времени, поэтому лучше запускать её в фоне в `tmux`, чтобы обрыв терминала не прервал загрузку. Перед загрузкой убедитесь, что текущий каталог:

```bash
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR
tmux new-session -d -s phmr_lfs_download 'git lfs clone https://hf-mirror.com/Datawhale/spring-festival-wushu-robot-replication-model'
tmux attach -t phmr_lfs_download
```

Если репозиторий уже частично загружен, но файлы LFS неполные, не удаляйте его и не начинайте заново: перейдите в каталог модели и продолжите загрузку:

```bash
git lfs pull
```

Альтернативный вариант загрузки весов:

```bash
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot/third_party/PromptHMR
bash scripts/fetch_smplx.sh
bash scripts/fetch_data.sh
```

Внимание: `fetch_smplx.sh` загружает не только SMPL-X. Скрипт также скачивает SMPL и дополнительные файлы. Наличие лишь `data/body_models/smplx/SMPLX_NEUTRAL.npz` ещё не означает, что все файлы моделей готовы.

### Команда объединения патчей (быстрая синхронизация изменений)

Если вы синхронизируетесь с исходным репозиторием, объедините патчи так:

```bash
cd /path/to/every-embodied/07-robot-operation-and-motion-control/Locomotion/video2robot

# 1. Убедитесь, что подмодули инициализированы
git submodule update --init --recursive

# 2. Применить патч основного репозитория
git apply patches/main.patch

# 3. Применить патч PromptHMR
git -C third_party/PromptHMR apply ../../patches/prompthmr.patch

# 4. Применить патч GMR
git -C third_party/GMR apply ../../patches/gmr.patch
```
