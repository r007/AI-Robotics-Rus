# Фиксация окружения AMD ROCm для LeRobot

Рабочая станция AMD 395 и облачная среда Radeon Cloud / W7900 используют один и тот же базовый набор версий LeRobot. Блокноты зависят от устаревшего пакета `lerobot.common`, поэтому нельзя смешивать новый checkout `lerobot.policies` в одном процессе Python.

## Зафиксированная база

| Параметр | Версия или путь |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.11.0+rocm7.13.0a20260425 |
| torchvision | 0.26.0+rocm7.13.0a20260425 |
| HIP | 7.13.26162 |
| LeRobot | `10b7b3532543b4adfb65760f02a49b4c537afde7` |
| Корень импорта LeRobot | `/path/to/lerobot_10b7_legacy` |
| Ключевые API | `lerobot.common.datasets`, `lerobot.common.policies` |

Перед обучением, оценкой и запуском Notebook задайте:

~~~bash
export LEROBOT_SRC=/path/to/lerobot_10b7_legacy
export PYTHONPATH=/path/to/lerobot_10b7_legacy:$PYTHONPATH
~~~

Либо выполните:

~~~bash
source code/rocm/activate_lerobot_rocm.sh
python code/rocm/verify_lerobot_rocm_env.py
~~~

Запускайте Notebook только после вывода `ENVIRONMENT_OK`. Если выводится `ENVIRONMENT_MISMATCH`, не начинайте обучение: сначала исправьте версии Python, ROCm или LeRobot.

## Распределение работы между платформами

- AMD 395: локальная отладка, короткие дымовые тесты, проверка данных, пошаговое воспроизведение в Notebook.
- Radeon Cloud / W7900: при той же фиксации окружения — длительное обучение, 14 замкнутых оценок и генерация видео.

Веса моделей, наборы данных и код Notebook можно использовать совместно; пакеты Python `site-packages` между машинами не копируются. В облаке окружение нужно заново установить по этому списку или использовать тот же шаблон контейнера или окружения.

## Известные расхождения

SSH-порт W7900 временно не завершает рукопожатие, поэтому онлайн-проверка в облаке ещё не выполнена. Когда соединение восстановится, первой командой должна быть:

~~~bash
python code/rocm/verify_lerobot_rocm_env.py --json
~~~

Окружения считаются согласованными, только если основные версии, коммит LeRobot и пути импорта в JSON на обоих концах полностью совпадают.
