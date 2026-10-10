# RoboWits: загрузка, обучение и оценка творческих задач на ROCm

RoboWits — бенчмарк для роботов в неожиданных условиях и творческого решения задач. Среда построена на Genesis, данные хранятся в формате LeRobot [библиотека робототехнического обучения], а официальные скрипты поддерживают обучение ACT, Pi0 и Pi0.5. В этой главе описаны получение ограниченных данных, подготовка ресурсов, обучение ACT на одном GPU AMD, продолжение обучения с контрольной точки, оценка 10 задач с исходными и мутированными условиями и вывод видео.

## Результаты главы и задачи

Официальное подмножество из 10 задач:

| ID | Задача |
| --- | --- |
| 01 | `align_blocks` |
| 02 | `retrieve_cube` |
| 03 | `gap_retrieve` |
| 04 | `pinch_card` |
| 06 | `dominos` |
| 09 | `hold_cup` |
| 13 | `cover_with_lid` |
| 16 | `stand_bulb` |
| 17 | `ball_onto_tower` |
| 25 | `water_into_mug` |

Для каждой задачи используются 50 официальных начальных состояний. Среда определяет успех по отношению целевых объектов и итоговому состоянию. Оценка с мутациями сохраняет смысл задачи, но заменяет указанные объекты, компоновку или физические условия. После установки зафиксируйте версию:

```bash
git -C "$SRC_ROOT/RoboWits" rev-parse HEAD
```

## 1. Заявка и загрузка данных

Сначала подайте заявку на доступ к [данным RoboWits в формате LeRobot](https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset), после одобрения выполните вход:

```bash
hf auth login
hf auth whoami

hf download XHRlyb2001/RoboWits_lerobot_dataset \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/RoboWits_lerobot_dataset"
```

Полный набор содержит около 1 207 эпизодов, 1 194 300 кадров и 24 задачи. После загрузки проверьте метаданные и каталог видео:

```bash
find "$DATA_ROOT/RoboWits_lerobot_dataset" -type f | wc -l
find "$DATA_ROOT/RoboWits_lerobot_dataset/meta" -maxdepth 2 -type f | head
find "$DATA_ROOT/RoboWits_lerobot_dataset/videos" -type f | head
```

## 2. Установка исходного кода и ресурсов

```bash
cd "$SRC_ROOT"
git clone https://github.com/UMass-Embodied-AGI/RoboWits.git
cd RoboWits
git submodule update --init --recursive
```

Оставьте уже установленный PyTorch для ROCm и не давайте зависимостям вышестоящего проекта заново установить колёса CUDA:

```bash
python -m pip install -e . --no-deps
python -m pip install accelerate datasets pyav wandb
python -m pip install -e third_party/lerobot --no-deps
```

Для части полных сцен нужны ресурсы BlenderKit. Войдите через плагин Blender и загрузите их в каталог, который указан в вышестоящем проекте; ключи в репозиторий не записывайте.

## 3. Проверка среды и формы данных

```bash
export PYTHONPATH="$SRC_ROOT/RoboWits/third_party/lerobot/src:$SRC_ROOT/RoboWits"
python scripts/robowits/examples/run_env.py
```

У RoboWits два основных интерфейса:

| Режим | Размерность состояния и действия | Назначение |
| --- | ---: | --- |
| EE / EE_ABS | 14 | Стратегия по позам концевого звена |
| JOINT / JOINT_ABS | 16 | Официальные данные и оценка ACT |

Официальное обучение ACT использует 16-мерные суставные состояния JOINT и абсолютные суставные действия JOINT_ABS. Обучение и оценка должны использовать один и тот же интерфейс.

## 4. Обучение ACT на одном GPU AMD

Официальный скрипт `scripts/robowits/train/train_act.sh` по умолчанию запускает 8 GPU через Accelerate [запуск распределённого обучения]. На устройстве с несколькими GPU можно задать набор данных, каталог вывода и число процессов напрямую:

```bash
cd "$SRC_ROOT/RoboWits"
HF_DATASET=XHRlyb2001/RoboWits_lerobot_dataset \
OUTPUT_DIR="$RUN_ROOT/robowits/act_100k" \
NUM_PROCESSES=8 \
STEPS=100000 \
SAVE_FREQ=5000 \
  bash scripts/robowits/train/train_act.sh
```

При запуске на одном GPU AMD вызывайте `lerobot-train` напрямую, сохраняя те же параметры модели и валидации, что и в официальном скрипте:

```bash
lerobot-train \
  --dataset.repo_id=local/robowits \
  --dataset.root="$DATA_ROOT/RoboWits_lerobot_dataset" \
  --dataset.video_backend=pyav \
  --dataset.val_frac=0.04 \
  --dataset.image_transforms.enable=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.vision_backbone=resnet18 \
  --policy.dim_model=512 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=4 \
  --save_freq=5000 \
  --log_freq=100 \
  --eval_freq=0 \
  --val_freq=500 \
  --val_max_samples=5000 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/robowits/act_100k"
```

Сначала запустите `--steps=10`, чтобы проверить обратное распространение, и только затем начинайте длительное обучение на 100k шагов. Checkpoint сохраняется каждые 5k шагов, поэтому обучение можно продолжить после обрыва на удалённом устройстве.

## 5. Продолжение обучения ACT

```bash
lerobot-train \
  --resume=true \
  --config_path="$RUN_ROOT/robowits/act_100k/checkpoints/last/pretrained_model/train_config.json" \
  --steps=100000 \
  --batch_size=32 \
  --num_workers=4 \
  --wandb.enable=false
```

Перед продолжением убедитесь, что в последнем checkpoint есть модель, состояние оптимизатора и файл `train_config.json`.

## 6. Обучение Pi0 и Pi0.5

Pi0:

```bash
lerobot-train \
  --dataset.repo_id=local/robowits \
  --dataset.root="$DATA_ROOT/RoboWits_lerobot_dataset" \
  --policy.type=pi0 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.pretrained_revision=26b99b9439acb1e352439e34ee9c67af0d76efa3 \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=5000 \
  --val_freq=500 \
  --val_max_samples=5000 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/robowits/pi0_100k"
```

Pi0.5:

```bash
lerobot-train \
  --dataset.repo_id=local/robowits \
  --dataset.root="$DATA_ROOT/RoboWits_lerobot_dataset" \
  --policy.type=pi05 \
  --policy.device=cuda \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.pretrained_revision=9e55186ad36e66b95cda57bc47818d9e6237ae30 \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=5000 \
  --val_freq=500 \
  --val_max_samples=5000 \
  --wandb.enable=false \
  --output_dir="$RUN_ROOT/robowits/pi05_100k"
```

Для Pi0 и Pi0.5 сначала нужен доступ к соответствующим базовым весам. Если памяти не хватает, уменьшите размер батча, но обязательно запишите отклонение в `run_config.json`.

## 7. Официальная оценка 10 задач

```bash
cd "$SRC_ROOT/RoboWits"
export CHECKPOINT_PATH="$RUN_ROOT/robowits/act_100k/checkpoints/last/pretrained_model"
export TASK_IDS='01 02 03 04 06 09 13 16 17 25'
export CONFIG=JOINT_ABS
export OBSERVATION_MODE=JOINT
export ENV_DEVICE=cpu
export N_EPISODES=50
export OUTPUT_DIR="$RUN_ROOT/robowits/act_100k_eval"

bash scripts/robowits/eval/eval.sh
```

Оценка с мутациями:

```bash
export OUTPUT_DIR="$RUN_ROOT/robowits/act_100k_mutation_eval"
bash scripts/robowits/eval/eval_mutation.sh
```

Для каждой задачи используются 50 начальных состояний из официального `eval_dataset_50/{task}.json`. Оценка сохраняет `eval_info.json`, статистику по задачам и видео.

## 8. Разделение задач между Genesis и ROCm

Сеть стратегии работает на GPU AMD, а сцены Genesis для начала можно стабильно выполнять на бэкенде CPU. Параметр PyTorch остаётся `cuda`, потому что ROCm [открытая платформа вычислений AMD] использует эту строку устройства:

```bash
export ROBO_WITS_TORCH_DEVICE=cuda
export ROBO_WITS_GENESIS_BACKEND=cpu
```

Если построение сцены на GPU-бэкенде Genesis падает, это не мешает сначала проверить обучение стратегии на AMD. Физический и стратегический бэкенды в результатах фиксируются отдельно.

## 9. Журнал обучения

При длительном обучении записывайте: номер шага, loss [значение функции потерь], learning rate [скорость обучения], grad norm [норма градиента], param norm [норма параметров], время шага, видеопамять, температуру, путь к checkpoint и задачи оценки. Значение loss используется только для диагностики обучения; итоговые выводы делаются по замкнутой успешности и показателям прогресса по этапам.

## 10. Официальные материалы

- [Проект RoboWits](https://umass-embodied-agi.github.io/RoboWits/)
- [Исходный код RoboWits](https://github.com/UMass-Embodied-AGI/RoboWits)
- [Данные RoboWits](https://huggingface.co/datasets/XHRlyb2001/RoboWits_lerobot_dataset)
- [Checkpoint RoboWits ACT для AMD](https://huggingface.co/Datawhale/robowits-act-amd-rocm)
