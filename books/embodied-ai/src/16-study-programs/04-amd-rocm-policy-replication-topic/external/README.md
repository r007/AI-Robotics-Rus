# Вышестоящий проект MuJoCo для запуска

Блокноты сбора, обучения и замкнутой оценки используют `PROJECT_ROOT` для связи с вышестоящим проектом MuJoCo. Репозиторий темы содержит учебные материалы, лёгкие скрипты и небольшие демонстрационные медиафайлы; наборы данных, кэши моделей и полные выходные данные обучения в него не копируются.

## Рекомендуемый способ

Укажите прямо на учебный каталог вышестоящего проекта в Every Embodied:

~~~bash
export EVERY_EMBODIED_ROOT=/path/to/every-embodied
export PROJECT_ROOT="$EVERY_EMBODIED_ROOT/<папка-главы-06>/<папка-темы-ACT-Pi0-SmolVLA>"
~~~

Здесь `<папка-главы-06>` и `<папка-темы-ACT-Pi0-SmolVLA>` — каталоги главы 06 и подраздела «MuJoCo: ACT, Pi0, SmolVLA» в исходном репозитории.

Вышестоящая точка входа: [LeRobot MuJoCo: обучение ACT, SmolVLA, Pi0](https://github.com/datawhalechina/every-embodied/tree/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA).

## Минимальная проверка

Разные версии вышестоящего проекта могут иметь разную структуру файлов. Для текущего процесса нужны как минимум:

~~~text
$PROJECT_ROOT/
├── asset/example_scene_y2.xml
├── mujoco_env/
├── train_model.py
└── eval_policy_success.py
~~~

Проверьте наличие:

~~~bash
test -f "$PROJECT_ROOT/asset/example_scene_y2.xml"
test -d "$PROJECT_ROOT/mujoco_env"
test -f "$PROJECT_ROOT/train_model.py"
test -f "$PROJECT_ROOT/eval_policy_success.py"
~~~

Пакетная точка входа темы находится в `code/run_closed_loop.py`, её не нужно копировать в вышестоящий каталог.

## Данные и модели

~~~bash
export DATA_ROOT=/path/to/datasets/every_embodied
export MODEL_ROOT=/path/to/checkpoints/every_embodied
export RUN_ROOT=/path/to/runs/every_embodied
~~~

| Каталог | Содержимое |
| --- | --- |
| `PROJECT_ROOT` | Исходный код вышестоящего проекта, сцены и небольшие конфигурации |
| `DATA_ROOT` | Данные LeRobot и метаданные |
| `MODEL_ROOT` | Модели, загруженные или полученные в результате обучения |
| `RUN_ROOT` | Журналы, результаты по эпизодам и видео |

Эти крупные каталоги не входят в Git-репозиторий учебника. При смене машины достаточно заново задать переменные окружения.

## Автономный пакет запуска

Если нужно собрать автономный учебный пакет, скопируйте в `external/mujoco_pnp/` только исходный код, XML сцены, сетки и конфигурации, которые нужны текущему заданию. В Notebook путь по-прежнему задаётся через `PROJECT_ROOT`:

~~~bash
export PROJECT_ROOT=/path/to/04-amd-rocm-policy-replication-topic/external/mujoco_pnp
~~~

После копирования повторите проверку устройства, импорт окружения и один цикл `reset → step → render`, чтобы убедиться, что зависимости полные.
