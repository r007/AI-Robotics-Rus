# Настройка GenieSim

> Скриншоты из исходного документа в этой версии опущены; ссылки на них сохранены как текст. Документ частично основан на официальной документации GenieSim (AgiBot) и на записях практических запусков.

GenieSim — симуляционная платформа от AgiBot, которая даёт разработчикам эффективную генерацию данных и бенчмарки для оценки, чтобы ускорить разработку воплощённого ИИ. GenieSim организует замкнутый цикл: генерация траекторий, обучение моделей, бенчмарки и проверка развёртывания. С помощью этой цепочки инструментов можно быстро проверять качество алгоритмов и улучшать модели. От простого захвата до сложного телеуправления GenieSim предоставляет высокореалистичную среду и точные метрики, помогая разработчикам эффективно вести разработку и итерации роботов.

GenieSim Benchmark — открытая версия оценки GenieSim; её цель — точное тестирование производительности моделей воплощённого ИИ и поддержка их оптимизации.

Сначала можно скачать ассеты:

```bash
sudo apt install git-lfs
git lfs install   # При запросе пароля используйте токен доступа с правом записи.
# Создать токен можно в настройках: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/agibot-world/GenieSimAssets
```

```bash
# Настройка репозитория NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && \
    sudo apt-get update
# Установка пакетов NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
# Настройка контейнерного runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Проверка NVIDIA Container Toolkit
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

> Системные требования Isaac Sim: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

### 2.2 Загрузка сцен и ассетов с Hugging Face

Откройте https://huggingface.co/datasets/agibot-world/GenieSimAssets и следуйте инструкциям.

### 2.3 Установка

#### 2.3.1 Контейнер Docker (рекомендуется)

1. Для разработки используйте контейнер Docker.
   - Установите Docker по [документации Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html).
2. Подготовьте образ Docker.

```bash
# Перейдите в корень genie_sim и создайте образ Docker из Dockerfile
docker build -f ./scripts/dockerfile -t registry.agibot.com/genie-sim/open_source:latest .
```

Если возникают сетевые проблемы, мой способ такой:

```bash
DOCKER_HOST_IP=$(ip addr show docker0 | grep "inet\s" | awk '{print $2}' | cut -d/ -f1)
docker build \
  --add-host=host.docker.internal:"$DOCKER_HOST_IP" \
  --build-arg http_proxy="http://host.docker.internal:7890" \
  --build-arg https_proxy="http://host.docker.internal:7890" \
  --build-arg no_proxy="localhost,127.0.0.1" \
  -f ./scripts/dockerfile \
  -t registry.agibot.com/genie-sim/open_source:latest .
```

- Запустите контейнер Docker и сервер.

```bash
# Запуск нового контейнера в корневом каталоге
# Замените ~/assets на папку GenieSimAssets
SIM_ASSETS=~/robo/GenieSimAssets/ ./scripts/start_gui.sh
./scripts/into.sh
# Запуск сервера
omni_python server/source/genie.sim.lab/raise_standalone_sim.py --enable_curobo True  # Траектории захвата в этой демонстрации генерирует Curobo
```

Дождитесь готовности приложения (App ready) и только потом выполняйте следующую команду.

Когда демонстрация супермаркета завершится, симулятор на этой стороне тоже закроется.

```bash
# Запуск контейнера в корневом каталоге
./scripts/into.sh
# Запуск клиента внутри контейнера
omni_python benchmark/task_benchmark.py --task_name=curobo_restock_supermarket_items --env_class=DemoEnv
omni_python benchmark/task_benchmark.py --task_name=iros_open_drawer_and_store_items --env_class=DemoEnv  # (пока выдаёт ошибку)
omni_python benchmark/task_benchmark.py --task_name=genie_task_home_microwave_food --env_class=DemoEnv  # (тоже не работает: выводит только step)
# GenieSim 2.1, как и 2.0, содержит только один пример задачи на Curobo. В открытой части настроена только задача с выкладкой товаров; остальные задачи iros требуют инференса модели или телеуправления.
```

Сначала появится зелёная строка info.
Затем подождите примерно минуту.

Если возникли ошибки, связанные с gRPC, см. issue и внесите правки по нему:

https://github.com/AgibotTech/genie_sim/issues/11

Внимание: в командной строке нельзя задавать никакие прокси. Также выполните `sudo ufw allow 50051`.

```bash
# Убедитесь, что сервер запущен (командой omni_python server/source/genie.sim.lab/raise_standalone_sim.py --enable_curobo True в том же контейнере).
# Убедитесь, что порт 50051 свободен. Если он занят, завершите процесс: найдите его через sudo lsof -i :50051 и выполните sudo kill -9 <PID>.
```

Если всё «зависло», подтяните свежий код с удалённого репозитория (git pull). Тогда зависание исчезнет.

Если появились крупные изменения файлов, перейдите в каталог assets и выполните:

```bash
git lfs pull  # Обновляет все файлы LFS до актуальной версии
```

https://github.com/AgibotTech/genie_sim/issues/29

#### 2.3.2 Хост-машина

1. Мы рекомендуем разработчикам использовать наш единый контейнер Docker.
2. Если хочется использовать собственное окружение, возьмите за основу наш `dockerfile` и установите перечисленные в нём зависимости.

#### 2.3.3 Руководство разработчика

#### 2.3.4 Включение хуков pre-commit (необязательно)

1. Установите и настройте `pre-commit`, чтобы включить автоматические форматировщики для Python / JSON / YAML и других файлов.

```bash
# Установка pre-commit в окружение Python
sudo apt install python3-pip
pip3 install pre-commit
# Включение предопределённых хуков pre-commit в репозитории
pre-commit install
```

2. Запустите форматировщик для всех отслеживаемых файлов:

```bash
pre-commit run --all-files
```

### 2.4 Задачи бенчмарка

#### 2.4.1 Запуск бенчмарка

1. Запустите контейнер Docker.

```bash
# Запуск нового контейнера в корневом каталоге
# Нужно заменить ~/assets на папку GenieSimAssets
SIM_ASSETS=~/17robot/GenieSimAssets ./scripts/start_gui.sh
```

2. Запустите бенчмарк.

Выполните следующую команду в корневом каталоге `genie_sim` вне контейнера Docker.

Функция автозапуска (Auto run) удобна, но нужен параметр infer:

```bash
./scripts/autorun.sh genie_task_home_pour_water infer
```

https://github.com/AgibotTech/genie_sim/issues/34

Поддерживаются такие имена задач:

**Имена задач**

- genie_task_cafe_espresso
- genie_task_cafe_toast
- genie_task_home_clean_desktop
- genie_task_home_collect_toy
- genie_task_home_microwave_food
- genie_task_home_open_drawer
- genie_task_home_pass_water
- genie_task_home_pour_water
- genie_task_home_wipe_dirt
- genie_task_supermarket_cashier_packing
- genie_task_supermarket_stock_shelf
- genie_task_supermarket_pack_fruit

---

#### 2.4.2 Фреймворк оценки бенчмарка

Конфигурация

Для оценки используется ADER (Action Domain Evaluation Rule) — правило оценки области действий.

**Пример**

```json
{
  "Acts": [
    {
      "ActionList": [
        {
          "ActionSetWaitAny": [
            {
              "Follow": "beverage_bottle_002|[0.2,0.2,0.2]|right"
            },
            {
              "Timeout": 120
            },
            {
              "Onfloor": "beverage_bottle_002|0.0"
            }
          ]
        },
        {
          "ActionSetWaitAny": [
            {
              "PickUpOnGripper": "beverage_bottle_002|right"
            },
            {
              "Timeout": 120
            },
            {
              "Onfloor": "beverage_bottle_002|0.0"
            }
          ]
        },
        {
          "ActionSetWaitAny": [
            {
              "Follow": "handbag_000|[0.4,0.4,0.4]|right"
            },
            {
              "Timeout": 120
            }
          ]
        },
        {
          "ActionSetWaitAny": [
            {
              "Inside": "beverage_bottle_002|handbag_000|1"
            },
            {
              "StepOut": 1000
            }
          ]
        }
      ]
    }
  ],
  "Init": [],
  "Objects": [
    {
      "bottle": [
        "beverage_bottle_002"
      ],
      "handbag": [
        "handbag_000"
      ]
    }
  ],
  "Problem": "pack_in_the_supermarket"
}
```

**Текущие возможности оценки**

| Действие | Описание | Базовый класс | Синтаксис |
| --- | --- | --- | --- |
| **Общие (COMMON)** | | | |
| ActionList | Очередное действие: внутренние действия выполняются по порядку. | ActionBase | `"ActionList":[]` |
| ActionSetWaitAny | Условное очередное действие: завершается, когда завершается любое из внутренних действий. | ActionBase | `"ActionSetWaitAny":[]` |
| ActionWaitForTime | Действие ожидания по времени: похоже на sleep, но не блокирует поток. | ActionBase | `"ActionWaitForTime": 3.0` |
| TimeOut | Проверка тайм-аута: проверяет, наступил ли тайм-аут. | ActionCancelBase | `"Timeout": 60` |
| StepOut | Проверка ограничения шагов: проверяет, достигнуто ли ограничение числа шагов. | ActionCancelBase | `"StepOut": 100` |
| ActionSetWaitAll | Выход, когда выполнены все условия. | ActionBase | `"ActionSetWaitAll":[]` |
| **Пользовательские (CUSTOM)** | | | |
| Ontop | Один объект находится над другим. | EvaluateAction | `"Ontop": "active_obj` |
| Inside | Один объект находится внутри другого. | EvaluateAction | `"Inside": "active_obj` |
| PushPull | Проверяет, находится ли призматический сустав объекта с суставами в пределах порога [min, max]; нужно, чтобы понять, открыт или закрыт ящикоподобный объект. | EvaluateAction | `"PushPull": "obj_id` |
| Follow | Проверяет, следует ли левый или правый захват за конкретным объектом в пределах, заданных рамкой [x, y, z]. | EvaluateAction | `"Follow": "obj_id` |
| PickUpOnGripper | Захват удерживает объект. | EvaluateAction | `"PickUpOnRightGripper": "object` |
| OnShelf | Объект находится в определённой области. | EvaluateAction | |
| Onfloor | Проверяет, упал ли указанный объект ниже опорной высоты ref_z; если да, выходит. | ActionCancelBase | `"Inside": "obj_id` |
| Cover | Объект A закрывает объект B. | EvaluateAction | `"Cover": "active_obj` |

**Структура данных результата оценки**

```json
// Вывод
[
  {
    "task_type": "benchmark",
    "model_path": "",
    "task_uid": "13513cf3-2d88-421e-b3e5-dc0998a60970",
    "task_name": "genie_task_supermarket",
    "stage": "",
    "result": {
      "code": -1,
      "step": 0,
      "msg": "",
      "progress": [],
      "scores": []
    },
    "start_time": "2025-06-05 15:04:31",
    "end_time": "2025-06-05 15:05:40"
  }
]
```

**Коды ошибок**

```python
from enum import Enum

class ErrorCode(Enum):
    INIT_VALUE = -1
    SUCCESS = 0
    ABNORMAL_INTERRUPTION = 1
    OUT_OF_MAX_STEP = 2
    UNKNOWN_ERROR = 500
```

---

### 2.5 Задачи манипуляции челленджа AgiBot World

#### 2.5.1 Запуск инференса базовой модели

1. Скачайте репозиторий Agibot World и базовую модель:

   a. [GitHub — OpenDriveLab/AgiBot-World, ветка manipulation-challenge](https://github.com/OpenDriveLab/AgiBot-World/tree/manipulation-challenge)

После загрузки, пока образ Docker не обновлён, agibot-world остаётся в Docker в ветке main.

```bash
# Инициализация репозитория Agibot World для запуска базовой модели
git clone -b <branch> <repo_url>
git submodule update --init --recursive
```

Потом каждый раз можно подтягивать обновления:

```bash
user@ubuntu22:~/workspace/main/AgiBot-World# ls
experiments       InternVL             __pycache__       scripts
genie_sim_ros.py  latent_action_model  README.md
hubconf.py        prismatic            requirements.txt
user@ubuntu22:~/workspace/main/AgiBot-World# git pull
fatal: detected dubious ownership in repository at '/path/to/workspace/main/AgiBot-World'
To add an exception for this directory, call:
        git config --global --add safe.directory /path/to/workspace/main/AgiBot-World
user@ubuntu22:~/workspace/main/AgiBot-World# # Нужно выполнить:
git config --global --add safe.directory /path/to/workspace/main/AgiBot-World
# затем
git pull
git submodule update --init --recursive
```

```text
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/finetuned/action_decoder.pt'

There was an error running python
```

Эта ошибка возникает, потому что не добавлена базовая модель.

3. Переместите все файлы базовой модели и код в каталог Agibot-World:

```bash
cp -r UniVLA/latent_action_model/ ./latent_action_model/
cp -r UniVLA/scripts/ ./scripts/
cp -r UniVLA/prismatic/ ./prismatic/
cp -r UniVLA/experiments/ ./experiments/
mkdir -p checkpoints/finetuned
cd checkpoints/finetuned
git lfs clone https://huggingface.co/qwbu/univla-iros-manipulation-challenge-baseline
mv univla-iros-manipulation-challenge-baseline/* ./
```

Чтобы оставаться в актуальном состоянии:

```bash
git clone -b manipulation-challenge https://github.com/OpenDriveLab/AgiBot-World.git AgiBot-World-manipulation-challenge
cp -r AgiBot-World-manipulation-challenge/UniVLA/* AgiBot-World/
```

```text
main
├── AgiBot-World
│   ├── InternVL
│   ├── experiments/robot
│   ├── latent_action_model
│   ├── prismatic
│   ├── robot
│   ├── scripts
│   │   └── infer.py
│   │   └── ...
│   │   └── ...
│   ├── checkpoints
│   │   └── finetuned
│   │       └── readme.txt
│   ├── genie_sim_ros.py
│   ├── hubconf.py
│   └── requirements.txt
```

В пятой строке infer.py есть место, которое нужно поправить. Позже я отправлю PR; будем рады, если кто-нибудь тоже внесёт правку (сейчас эта проблема уже исправлена, поэтому её трогать не нужно).

4. Запустите контейнер Docker:

```bash
# Запуск нового контейнера в корневом каталоге
# Замените ~/assets на локальную папку assets
SIM_ASSETS=~/robo/GenieSimAssets ./scripts/start_gui.sh
```

4. Запустите инференс базовой модели.

Выполните эти команды в корневом каталоге `genie_sim` вне контейнера Docker:

```text
./scripts/autorun.sh iros_open_drawer_and_store_items infer
./scripts/autorun.sh iros_clear_the_countertop_waste infer
```

Поддерживаются такие имена задач:

| Название задачи |
| --- |
| iros_clear_the_countertop_waste |
| iros_open_drawer_and_store_items |
| iros_heat_the_food_in_the_microwave |
| iros_pack_moving_objects_from_conveyor |
| iros_pickup_items_from_the_freezer |
| iros_restock_supermarket_items |
| iros_pack_in_the_supermarket |
| iros_make_a_sandwich |
| iros_clear_table_in_the_restaurant |
| iros_stamp_the_seal |

---

Получилось: видеопамяти 24 ГБ хватает.

Открыть ящик и захватить предмет удалось, но процесс ещё содержит много проблем: в конце кубик Рубика упал на пол.

#### 2.5.2 Интеграция собственной политики

1. Постройте свой код по инструкции ниже.

2. a. Начните интеграцию модели с шаблона по такому пути. Не меняйте структуру функции `infer`.

```text
main
├── model
│   ├── demo_infer.py
```

b. При кастомизации строго соблюдайте темы ROS-узла, которые указаны для задачи.

*Обмен данными между моделью инференса и симуляционной средой стандартизирован и реализован на ROS2. Ниже перечислены темы, выделенные для задач.*

| Название темы | Публикатор | Подписчик | Тип сообщения | Содержание |
| --- | --- | --- | --- | --- |
| /joint_command | Модель | Симуляция | JointState | Команды суставов, которые выдаёт инференс модели |
| /joint_states | Симуляция | Модель | JointState | Текущее состояние суставов робота из симуляции |
| /sim/head_img | Симуляция | Модель | CompressedImage | Текущее RGB-изображение головы из симуляции |
| /sim/left_wrist_img | Симуляция | Модель | CompressedImage | Текущее RGB-изображение левого запястья из симуляции |
| /sim/right_wrist_img | Симуляция | Модель | CompressedImage | Текущее RGB-изображение правого запястья из симуляции |
| /sim/head_depth_img | Симуляция | Модель | CompressedImage | Текущая карта глубины головы из симуляции |
| /sim/left_wrist_depth_img | Симуляция | Модель | CompressedImage | Текущая карта глубины левого запястья из симуляции |
| /sim/right_wrist_depth_img | Симуляция | Модель | CompressedImage | Текущая карта глубины правого запястья из симуляции |

c. После кастомизации проверьте, что структура файлов соответствует заданной, и убедитесь, что есть `infer.py` и `genie_sim_ros.py`.

```text
main
├── AgiBot-World
│   ├── scripts
│   │   └── infer.py
│   │   └── ...
│   │   └── ...
│   ├── genie_sim_ros.py
│   ├── ...
│   └── ...
```

2. Запустите контейнер Docker:

```bash
# Замените ~/assets на папку GenieSimAssets
SIM_ASSETS=~/assets ./scripts/start_gui.sh
```

3. Запустите инференс модели.

4. Выполните в корневом каталоге genie_sim вне контейнера Docker:

```text
./scripts/autorun.sh {TASK_NAME} infer
```

4. Зависимости для инференса модели.

Часто используемые библиотеки Python перечислены в файле `requirements.txt` репозитория Genie Sim Benchmark и предустановлены в образе Docker на нашем тестовом сервере. Если для запуска политики нужны дополнительные библиотеки Python, вместе с файлами модели на тестовый сервер можно загрузить файл `requirements.txt` с дополнительными библиотеками.

---

#### 2.5.3 Фреймворк оценки челленджа AgiBot World

См. 2.4.2 Фреймворк оценки бенчмарка.

---

### 2.6 Телеуправление

#### 2.6.1 PICO

Поддерживает управление поясом, головой, левым и правым концевыми эффекторами и перемещение базы с помощью контроллера.

**Руководство пользователя**

| Номер | Функция (левый / правый) |
| --- | --- |
| ① | Джойстик: перемещение базы робота; нажатие: сброс позы базы |
| ② | Сброс левой руки |
| ③ | Включить отслеживание позы |
| ④ | Сброс правой руки |
| ⑤ | Сброс корпуса и головы |
| ⑥ | Управление левым захватом |

**Настройка Pico**

1. Подключитесь к той же **локальной сети (LAN)**, что и компьютер.
2. В библиотеке запустите **приложение AIDEA Vision App**.
3. Выберите **беспроводное подключение** и введите **IP-адрес** компьютера.

**Запуск**

1. Запустите сервер:

```bash
# Выполнить внутри контейнера
omni_python server/source/genie.sim.lab/raise_standalone_sim.py
```

2. Запустите управление PICO в контейнере:

```bash
# Выполнить внутри контейнера
omni_python teleop/teleop.py --task_name genie_task_home_microwave_food --mode pico --host_ip x.x.x.x
```

---

#### 2.6.2 Клавиатура

Поддерживает управление поясом, головой, левым и правым концевыми эффекторами и перемещение базы с клавиатуры.

Руководство пользователя

| Клавиша | Функция | Клавиша | Функция |
| --- | --- | --- | --- |
| w | Концевой эффектор вперёд | i | Крен + |
| s | Концевой эффектор назад | k | Крен - |
| a | Концевой эффектор влево | j | Тангаж + |
| d | Концевой эффектор вправо | l | Тангаж - |
| q | Концевой эффектор вверх | u | Рысканье + |
| e | Концевой эффектор вниз | o | Рысканье - |
| ↑ | База вперёд | shift+↑ | Тангаж головы + |
| ↓ | База назад | shift+↓ | Тангаж головы - |
| ← | База поворот влево | shift+← | Рысканье головы + |
| → | База поворот вправо | shift+→ | Рысканье головы - |
| ctrl+↑ | Пояс вверх | ctrl+tab | Переключить руку |
| ctrl+↓ | Пояс вниз | r | Сброс |
| ctrl+← | Тангаж пояса - | c | Закрыть захват |
| ctrl+→ | Тангаж пояса + | ctrl+c | Открыть захват |

**Запуск**

1. Запустите сервер:

```bash
SIM_ASSETS=~/robo/GenieSimAssets/ ./scripts/start_gui.sh
./scripts/into.sh
# Выполнить внутри контейнера
omni_python server/source/genie.sim.lab/raise_standalone_sim.py
```

2. Запустите клавиатурное управление в контейнере:

```bash
# Выполнить внутри контейнера
omni_python teleop/teleop.py --task_name genie_task_home_microwave_food --mode "keyboard"
```

Управление поясом пока стоит использовать осторожно: кажется, оно ещё не доведено. В целом управление концевыми эффекторами работает очень хорошо.

---

### 2.7 Запись и воспроизведение

Здесь ещё раз перечислены задачи:

- genie_task_cafe_espresso
- genie_task_cafe_toast
- genie_task_home_clean_desktop
- genie_task_home_collect_toy
- genie_task_home_microwave_food
- genie_task_home_open_drawer
- genie_task_home_pass_water
- genie_task_home_pour_water
- genie_task_home_wipe_dirt
- genie_task_supermarket_cashier_packing
- genie_task_supermarket_stock_shelf
- genie_task_supermarket_pack_fruit

Для повышения эффективности сначала записывают траекторию, а затем выполняют воспроизведение, чтобы записать видео.

Вся информация о сцене (позы суставов робота, позы объектов, позы камер и т. д.) записывается в файл `state.json`.

- Запустите сервер:

```bash
# Выполнить внутри контейнера
omni_python server/source/genie.sim.lab/raise_standalone_sim.py
```

- Запустите клиент:

```bash
# Выполнить внутри контейнера
omni_python teleop/teleop.py --task_name genie_task_home_pour_water --mode keyboard  --record
```

Информация о сцене записывается в `./output/recording_data/{TASK_NAME}/state.json`.

Здесь создаётся /path/to/workspace/main/output/recording_data/genie_task_home_pour_water/state.json.

#### 2.7.1 Воспроизведение

Воспроизведите траекторию и запишите видео.

- Запустите сервер:

```bash
# Выполнить внутри контейнера
omni_python server/source/genie.sim.lab/raise_standalone_sim.py --disable_physics --record_img --record_video
```

- Запустите клиент:

```bash
# Например: TASK_NAME=genie_task_home_pour_water
# Выполнить внутри контейнера
omni_python teleop/replay_state.py --task_file teleop/tasks/${TASK_NAME}.json --state_file output/recording_data/${TASK_NAME}/state.json --record
```

Изображения и видео записываются в `./output/recording_data/{TASK_NAME}/{IDX}/`.

---

## 3. Примеры использования

### 3.1 Запуск симуляции одной строкой кода

1. Запустите контейнер Docker:

```bash
SIM_ASSETS=~/robo/GenieSimAssets ./scripts/start_gui.sh
```

2. Выполните shell-скрипты в корневом каталоге `genie_sim` вне контейнера Docker.

3. a. Запуск раскладки задачи:

```bash
./scripts/autorun.sh {TASK_NAME}
```

b. Запуск телеуправления PICO:

```bash
./scripts/autorun.sh {TASK_NAME} pico {HOST_IP}
```

c. Запуск клавиатурного телеуправления:

```bash
./scripts/autorun.sh {TASK_NAME} keyboard
```

d. Запуск воспроизведения сцены:

```bash
./scripts/autorun.sh {TASK_NAME} replay {STATE_FILE_PATH}
```

e. Запуск инференса модели:

```bash
./scripts/autorun.sh {TASK_NAME} infer
```

f. Запуск очистки:

```bash
./scripts/autorun.sh clean
```

3. Нажмите `q` или `Q` в терминале, чтобы нормально остановить задачу. По умолчанию скрипт `autorun` включает запись топиков ROS. Если она не нужна, удалите соответствующий код в скрипте `autorun` и регулярно очищайте папку вывода. (Какую информацию сохранять и куда?)

---

### 3.2 Как настроить файл задачи бенчмарка

Каждый файл задачи бенчмарка содержит 6 ключевых элементов:

1. **task** — уникальное имя задачи.

2. **objects** — различные объекты задачи:
   1. `extra_objects`: неинтерактивные объекты;
   2. `fix_objects`: интерактивные объекты с фиксированной начальной позой;
   3. `task_related_objects`: интерактивные объекты со случайной начальной позой.

3. **recording_setting** — какие ракурсы камер записывать.

4. **robot** — ID робота, конфигурационный файл робота и начальная поза базы робота.

5. **scene** — набор информации о сцене:
   1. `scene_id`: уникальное имя сцены;
   2. `function_space_objects`: кубическая область, в которой случайно генерируются `task_related_objects`;
   3. `scene_info_dir`: путь к ассетам сцены;
   4. `scene_usd`: путь к файлу usd сцены.

6. **stages** — несколько подэтапов, спроектированных для планирования задачи:
   1. `action`: имя действия;
   2. `active`: активный объект, например захват;
   3. `passive`: пассивный объект, например бутылка.

---

### 3.3 Как создать полную задачу бенчмарка с моделью воплощённого ИИ

1. Сначала соберите шаблон задачи, например "genie_task_supermarket.json". Этот конфигурационный файл содержит робота, сцену, объекты и этапы.

```text
benchmark/bddl/eval_tasks/your_task.json
```

Правильно заполните пути к ассетам. Программа ищет ассеты в переменной окружения `SIM_ASSETS`, которую задаёт пользователь.

2. Обобщение сцен: сопоставьте задачу со сценами.

```text
Путь к файлу: benchmark/bddl/task_to_preselected_scenes.json
```

```json
{
  "genie_task_supermarket": [
    "scenes/genie/supermarket_02/Collected_emptyScene_01/emptyScene_01.usd"
  ]
}
```

3. Напишите код критериев оценки, например:

```text
Путь к файлу: benchmark/bddl/task_definitions/genie_task_supermarket/problem0.bddl
```

```lisp
(define (problem restock_shelves)
    (:domain isaac)
    (:objects
        benchmark_beverage_bottle_013 - bottle.n.01
    )
    (:init
        (onfloor benchmark_beverage_bottle_013 floor.n.01_2)
    )
    (:goal
        (onshelf ?benchmark_beverage_bottle_013)
    )
)
```

Теперь настройка бенчмарка завершена.

4. Подключите модель воплощённого ИИ в "yourpolicy.py". (См. demopolicy.py.)

```python
class YourPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        """Инициализация конфигурации и загрузка модели."""
        pass

    def reset(self):
        """Сброс."""
        pass

    def act(self, observations, **kwargs) -> np.ndarray:
        """Действие на основе наблюдений.

        Параметры: observations содержит изображения робота (Head_Camera_01 / Right_Camera_01 / Left_Camera_01) и текущую информацию о суставах.
        Возвращает: целевые суставы робота.
        """
        pass
```

5. Наконец, запустите клиент и оцените свою модель ИИ:

```bash
python3 benchmark/task_benchmark.py --task_name {task name} --policy_class {policy name} --env_class OmniEnv
```

---

### 3.4 Лучшие практики по параметрам ассетов для задач манипуляции

#### 3.4.1 Как отлаживать коллайдеры в IsaacSim

> Включите режим отладки коллайдеров по инструкции.

#### 3.4.2 Плохие примеры

- Сложные сетки приводят к ошибкам вычисления коллизий: чем проще, тем лучше.
- Слишком много треугольников в маленькой области обычно вызывает проблемы с сеточными коллизиями.

convexHull и convexDecomposition

#### 3.4.3 Хорошие примеры

Хорошо — значит, объект легко захватить и в сетке мало коллизий.

- Хорошие коллизионные сетки обычно выглядят простыми и аккуратными.
- Следите, чтобы важные для захвата области были выделены точно (например, у банки, бутылки, крышки).

Сетка convexDecomposition

Коллайдер convexDecomposition

convexHull

---

## 4. Справочник API

| Функция | Описание | Входы | Возврат |
| --- | --- | --- | --- |
| **Инициализация** | | | |
| client.InitRobot() | Инициализация робота и сцены | robot_cfg: путь к конфигурационному файлу (в папке 'robot_cfg'); scene_usd: путь к usd; init_position([x,y,z]); init_rotation([x,y,z]) | |
| **Контроллер** | | | |
| client.moveto() | Перемещение концевого эффектора робота в целевую позу | target_position([x,y,z]); target_quaternion([w,x,y,z]); is_backend: 1. True: прямолинейное движение ЭЭ; 2. False: движение с обходом препятствий; ee_interpolation: 1. True: интерполяция ruckig; 2. False: интерполяция целевой точки; distance_frame: плотность интерполяции | |
| client.set_joint_positions() | Перемещение суставов робота в целевое состояние | target_joint_position: [N], rad/s; joint_indices: [N], индексы; is_trajectory: 1. True: суставы выполняют действие; 2. False: установить состояние суставов | |
| client.set_gripper_state() | Установить состояние захвата (открыт или закрыт) | gripper_command: "open" или "close"; is_right(bool); opened_width(float) | |
| client.SetTrajectoryList() | Задать траекторию списком поз ЭЭ | trajectory_list(list): [[position(xyz), rotation(wxyz)]] | |
| **Объекты** | | | |
| client.add_object() | Добавить USD-объект в сцену | usd_path(str): относительный путь к ассету от SIM_ASSETS; prim_path(str): путь prim объекта в сцене, например "/World/Object/obj_01"; lable_name(str): имя метки объекта для семантической сегментации; target_position([x,y,z]); target_quaternion([w,x,y,z]); target_scale([x,y,z]); mass:[kg]; **add_particle(bool):** добавить дополнительные частицы жидкости, например воду в чайнике; particle_size:([x,y,z]); particle_position:([x,y,z]) | |
| client.SetMaterial() | Задать материал XFormPrim | material_info(list): например material_info = [{"object_prim": "/obj1", "material_name": "wood", "material_path": "materials/wood"}] | |
| client.SetLight() | Задать освещение сцены | light_info(list): например light_info = [{"light_type": "Distant", "light_prim": "/World/DistantLight", "light_temperature": 2000, "light_intensity": 1000, "rotation": [1,0.5,0.5,0.5], "texture": ""}] | |
| client.SetObjectPose() | Задать позу твёрдых тел и сочленённых объектов за один физический шаг | object_info(list): например object_info = [{"prim_path": "/World/obj1", "position": [x,y,z], "rotation": [w,x,y,z]}]; object_joint_info(list): информация о суставах сочленённых объектов | |
| **Датчики** | | | |
| client.AddCamera() | Добавить камеру на сцену | camera_prim(str): путь prim камеры; camera_position(xyz): положение камеры; camera_rotation(wxyz): поворот камеры; Width, Height, focus_length, horizontal_aperture, vertical_aperture — внутренние параметры камеры: fx = Width * focus_length / horizontal_aperture; fy = Height * focus_length / vertical_aperture; is_local(bool): локальная или мировая поза | |
| client.capture_frame() | Снять кадр rgb / глубины | camera_prim(str): путь prim камеры | response.color_image.data; response.depth_image.data |
| client.capture_semantic_frame() | Снять семантический кадр | camera_prim_path(str): путь prim камеры | response.semantic_mask.data |
| **Наблюдение** | | | |
| client.get_observation() | Получить данные камеры / суставов / tf за один кадр | data_keys(dict): например data_keys = { 'camera': {'camera_prim_list': ["/camera"], 'render_depth': True, 'render_semantic': True}, 'pose': ["/object1"], 'joint_position': True, 'gripper': True} | observation = {"camera": camera_datas, "joint": joint_datas, "pose": object_datas, "gripper": gripper_datas} |
| client.GetIKStatus() | Вычисление обратной кинематики | target_poses(list): [{"position": xyz, "rotation": wxyz}]; is_right(bool): тип руки для двуруких; ObsAvoid(bool): вычислять IK с обходом препятствий | "status"([bool]): успешность IK; "Jacobian"([double]): оценка якобиана состояния суставов IK; "joint_positions"([list]): положения суставов IK; "joint_names"([list]): имена суставов IK |
| client.GetEEPose() | Получить мировую позу концевого эффектора | is_right(bool): выбор руки | state.ee_pose.position; state.ee_pose.rpy |
| client.get_object_pose() | Получить мировую позу объекта | prim_path(str): путь prim объекта | object_pose.position; object_pose.rpy |
| client.get_joint_positions() | Получить текущие положения суставов робота | | result.states.name; result.states.position |
| **Особые функции Curobo** | | | |
| client.AttachObj() | Прикрепить пассивный объект | prim_paths[list[str]]: пути prim прикрепляемых объектов | |
| client.DetachObj() | Отсоединить все объекты | | |
| **Настройка записи** | | | |
| client.start_recording() | Начать запись фрагмента | data_keys(dict): настройки записи, например data_keys = {'camera': {'camera_prim_list': ['/World/base_link/Head_Camera'], 'render_depth': True, 'render_semantic': True}, 'pose': ['/World/obj1'], 'joint_position': True, 'gripper': True}; fps(int): частота кадров записи; task_name(str) | |
| client.stop_recording() | Остановить запись | | |
| client.reset() | Сбросить сцену, робота и объекты | | |
| client.Exit() | Выйти из приложения | | |
