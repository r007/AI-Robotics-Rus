# Точка входа в Notebook

Этот каталог содержит три типа Notebook: шаблоны запуска, учебные Notebook с сохранёнными выводами и сквозные рабочие процессы, которые зависят от вышестоящего проекта и локальных моделей. Типы различаются по состоянию выполнения.

## Настройка перед запуском

~~~bash
export PROJECT_ROOT=/path/to/mujoco_pnp
export DATA_ROOT=/path/to/datasets/every_embodied
export MODEL_ROOT=/path/to/checkpoints/every_embodied
export RUN_ROOT=/path/to/runs/every_embodied
export OUTPUT_ROOT=$RUN_ROOT/notebooks

cd /path/to/<topic-folder>
jupyter lab
~~~

`<topic-folder>` — каталог этой темы в исходном репозитории `every-embodied`.

`PROJECT_ROOT` указывает на вышестоящий проект MuJoCo. Данные, модели и выходные каталоги располагаются вне репозитория темы.

## Классы Notebook

### 01–06: шаблоны запуска

Notebook 01–06 хранят код проверок, логику графиков и шаблоны команд; в версии для репозитория они не выполнялись заранее. Читатель запускает их на своём устройстве и получает результаты для своего окружения и своей модели.

| Notebook | Назначение |
| --- | --- |
| [01_device_env_check.ipynb](./01_device_env_check.ipynb) | ROCm, PyTorch, видеопамять, температура и каталоги |
| [02_physical_success_review.ipynb](./02_physical_success_review.ipynb) | Условия физического успеха и проверка ключевых кадров |
| [03_act_dagger_diagnostics.ipynb](./03_act_dagger_diagnostics.ipynb) | Кривые ACT, диагностика действий и DAgger |
| [04_smolvla_weighted_sampling.ipynb](./04_smolvla_weighted_sampling.ipynb) | Группировка по красной и синей кружке и веса выборки |
| [05_pi0_smoke_gate.ipynb](./05_pi0_smoke_gate.ipynb) | Права доступа, загрузка модели и конфигурация короткого обучения |
| [06_rocm_debug_playbook.ipynb](./06_rocm_debug_playbook.ipynb) | Послойный журнал отладки |

### 07–13: разделённые процессы с учебными выводами

Notebook 07–13 содержат сохранённые выводы учебного запуска, таблицы и изображения можно читать сразу. Интерактивный сбор, длительное обучение и пакетная замкнутая оценка управляются переменными `RUN_*`: открытие файла не запускает долгие задачи автоматически.

| Notebook | Назначение | Состояние сохранения |
| --- | --- | --- |
| [07_data_collection_and_audit.ipynb](./07_data_collection_and_audit.ipynb) | Сбор с клавиатуры, запись с четырёх видов и аудит данных | 11/11 ячеек кода с сохранёнными выводами |
| [08_act_training_rocm.ipynb](./08_act_training_rocm.ipynb) | Короткое обучение ACT и полная конфигурация | 11/11 |
| [09_smolvla_training_rocm.ipynb](./09_smolvla_training_rocm.ipynb) | Обучение SmolVLA и разбиение по задачам | 11/11 |
| [10_pi0_training_rocm.ipynb](./10_pi0_training_rocm.ipynb) | Права доступа Pi0 и конфигурация обучения | 12/12 |
| [11_mujoco_closed_loop_deploy.ipynb](./11_mujoco_closed_loop_deploy.ipynb) | Загрузка модели, результаты по эпизодам и видео | 10/10 |
| [12_pi0_strict_input_end_to_end.ipynb](./12_pi0_strict_input_end_to_end.ipynb) | Проверка входа Pi0 и рандомизации среды | 10/10 |
| [13_pi05_random_position_eef_delta.ipynb](./13_pi05_random_position_eef_delta.ipynb) | EEF-delta, блоки действий и данные восстановления | 13/13 |

### 14–16: сквозные рабочие процессы

Notebook 14–16 объединяют конфигурацию модели, обучение, оценку и видео в одном файле. В репозитории сохранены параметры и исторические учебные выводы; для реального обучения и оценки нужны вышестоящий проект, данные и каталоги моделей.

| Notebook | Модель | Пояснение |
| --- | --- | --- |
| [14_smolvla_end_to_end.ipynb](./14_smolvla_end_to_end.ipynb) | SmolVLA | Установите `RUN_LONG_TRAIN=1` для запуска обучения и `RUN_EVAL=1` для оценки этой модели |
| [15_pi0_end_to_end.ipynb](./15_pi0_end_to_end.ipynb) | Pi0 | Нужны базовая модель с подтверждённым доступом и соответствующие данные |
| [16_act_end_to_end.ipynb](./16_act_end_to_end.ipynb) | ACT | Обычный рецепт и рецепт repair выбираются переменной `ACT_RECIPE` |

Переключатели сквозных Notebook по умолчанию выключены. Если пути вышестоящего проекта не заданы, ячейки конфигурации и пояснений всё равно выполняются, а ячейки обучения и оценки выводят список нужных переменных.

## Обычное и защищённое обучение

В [workflows/README.md](./workflows/README.md) приведены отдельные копии обычного и защищённого обучения. Оба варианта используют разные подкаталоги `RUN_ROOT`, а оценка читает модель, полученную в текущем запуске Notebook.

## Повторное выполнение

После установки Jupyter-ядра лучше выполнять Notebook штатным способом Jupyter. Если в учебном образе не установлен `nbclient`, можно использовать облегчённый исполнитель для ячеек без интерактивных окон:

~~~bash
python code/execute_tutorial_notebooks.py \
  notebooks/07_data_collection_and_audit.ipynb \
  notebooks/08_act_training_rocm.ipynb \
  notebooks/09_smolvla_training_rocm.ipynb \
  notebooks/10_pi0_training_rocm.ipynb \
  notebooks/11_mujoco_closed_loop_deploy.ipynb \
  notebooks/12_pi0_strict_input_end_to_end.ipynb \
  notebooks/13_pi05_random_position_eef_delta.ipynb
~~~

Во время длительного обучения индикатор прогресса обновляется в одной и той же области вывода; пошаговые метрики записываются в `metrics.jsonl`, чтобы не создавать отдельный блок вывода на каждом шаге.

## Проверка

После выполнения сквозного процесса для одной модели проверьте:

1. данные, наблюдения и действия в конфигурации обучения совпадают со средой оценки;
2. путь к оцениваемой модели находится в `checkpoints/` текущего запуска;
3. число результатов по эпизодам равно знаменателю оценки в конфигурации;
4. успешные и неуспешные видео воспроизводятся и связаны со своими строками результатов;
5. в Notebook нет ошибок и абсолютных путей, относящихся к конкретной машине.
