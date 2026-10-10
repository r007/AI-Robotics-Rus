# Тема AMD ROCm: воспроизведение стратегий (policy)

Эта тема предназначена для читателей, которые хотят на GPU AMD обучить воплощённые стратегии, провести замкнутую оценку и перенести их в симуляцию. Материал начинается с проверки устройства ROCm, сбора данных в MuJoCo и обучения стратегий в LeRobot, а затем расширяется до RoboCasa365, DexJoCo, DISCOVERSE, RoboWits и Unitree G1. Каждый эксперимент сопровождается входными данными, командами запуска, каталогами результатов и правилами оценки.

## Входы в тему

Выберите один маршрут в зависимости от цели; не обязательно читать файлы подряд по номерам.

| Маршрут | Для кого | Рекомендуемый порядок | Итоговый результат |
| --- | --- | --- | --- |
| Обучение стратегии захвата | Первое обучение робототехнической стратегии на AMD | 00 → 01 → 07 → 08/09/10 → 11 → 02 | Датасет, модель, результаты по эпизодам и видео |
| Диагностика стратегии | Есть готовая модель, нужно найти причину сбоев в замкнутом цикле | 02 → 03/04/05 → 06 | Статистика по фазам, графики действий и журнал исправлений |
| Перенос на несколько бенчмарков | Воспроизведение бытовых задач, ловкой руки и безопасного управления | 09 → 10 → 11/12/13/14/15 → 16 | Результаты при фиксированном протоколе, многовидовое видео и архив каталога |

Если локального устройства AMD нет, начните с [облачной платформы AUP Learning Cloud и AMD Developer Cloud](./00-amd-aup-free-cloud-platform-usage-guide.md). Подробности работы в веб-интерфейсе платформы, JupyterHub и Code Server вынесены в [приложение об AUP, Jupyter и Code Server](./appendices/AUP_JUPYTER_CODE_SERVER.md); основные главы содержат только окружение, хранилище и шаги запуска.

## Длинные примеры

| Двуручная Ханойская башня в DexJoCo | Длинная уборка по RoboCasa365 |
| --- | --- |
| [![Двуручная Ханойская башня в DexJoCo](./assets/competition_showcase/dexjoco_bimanual_hanoi_poster.jpg)](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/dexjoco/recovery/bimanual-hanoi.mp4) | [![Длинная уборка по RoboCasa365](./assets/competition_showcase/robocasa_pack_identical_lunches_poster.jpg)](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/robocasa-recovery/pack-success.mp4) |
| Pi0.5, двуручная координация, три вида, 47,6 с | GR00T N1.5, длительные бытовые операции, четыре вида, 195 с |

Больше задач, моделей и протоколов см. в [AMD Physical AI: бенчмарки симуляции и длинные видео](./09-amd-physical-ai-simulation-benchmarks-and-long-term-video-reproduction.md).

## Первый запуск

1. По [проверке устройства и окружения](./01-amd-rocm-device-and-environment-confirmation.md) убедитесь, что ROCm, PyTorch и постоянный каталог настроены.
2. Откройте [предобученные веса и опыт без обучения](./08-pre-trained-weights-and-zero-training-experience.md): скачайте модель или посмотрите четырёхвидовой повтор из репозитория.
3. По главе [сбор данных и обучение](./07-rocm-end-to-end-collection-training-and-deployment.md) выполните запись демонстраций, аудит данных и короткое обучение.
4. Запустите [пакетный замкнутый скрипт](./code/README.md) с фиксированными random seed и сохраните `results.jsonl` и видео.
5. По [единой оценке и архивации](./16-unify-the-evaluation-videos-and-result-archiving.md) соберите процент успеха и конфигурацию запуска.

`5000 steps` — это конфигурация короткого учебного обучения. Пригодность модели подтверждается не только ею: оценка проводится на случайных seed, не участвовавших в обучении, с проверкой физического успеха и воспроизведением видео.

## Оглавление глав

### Основы и стратегии захвата

| Номер | Глава | Практическая точка входа |
| --- | --- | --- |
| 00 | [Облачные платформы, постоянное хранилище и удалённая разработка](./00-amd-aup-free-cloud-platform-usage-guide.md) | Команды платформы |
| 01 | [Проверка устройства AMD ROCm и окружения](./01-amd-rocm-device-and-environment-confirmation.md) | [Notebook 01](./notebooks/01_device_env_check.ipynb) |
| 02 | [Оценка физического успеха и проверка видео](./02-physical-success-evaluation-and-video-review.md) | [Notebook 02](./notebooks/02_physical_success_review.ipynb) |
| 03 | [Перенос ACT на ROCm и диагностика DAgger](./03-act-rocm-migration-and-dagger-diagnosis.md) | [Notebook 03](./notebooks/03_act_dagger_diagnostics.ipynb) |
| 04 | [Перенос SmolVLA и взвешивание выборки](./04-smolvla-rocm-migration-and-sampling-weighting.md) | [Notebook 04](./notebooks/04_smolvla_weighted_sampling.ipynb) |
| 05 | [Pi0/Pi0.5: права, короткое обучение и оценка](./05-pi0-rocm-permission-smoke-and-training-gatekeeping.md) | [Notebook 05](./notebooks/05_pi0_smoke_gate.ipynb) |
| 06 | [Справочник по отладке ROCm](./06-rocm-debugging-review-and-troubleshooting-cases.md) | [Notebook 06](./notebooks/06_rocm_debug_playbook.ipynb) |
| 07 | [Сбор данных, обучение и развёртывание в MuJoCo](./07-rocm-end-to-end-collection-training-and-deployment.md) | [Notebook 07–13](./notebooks/README.md) |
| 08 | [Загрузка моделей и опыт без обучения](./08-pre-trained-weights-and-zero-training-experience.md) | Модели Hugging Face |

### Перенос на несколько бенчмарков

| Номер | Глава | Основной результат |
| --- | --- | --- |
| 09 | [Бенчмарки симуляции и длинные видео](./09-amd-physical-ai-simulation-benchmarks-and-long-term-video-reproduction.md) | Карта проектов и репрезентативные видео |
| 10 | [Единый каталог, загрузка и авторизация](./10-simulation-benchmark-download-and-unified-directory.md) | Переменные каталогов и проверка окружения |
| 11 | [RoboCasa365](./11-robocasa365-rocm-download-training-and-evaluation.md) | Обучение бытовым задачам, оценка 16 задач, видео с четырёх видов |
| 12 | [DexJoCo](./12-dexjoco-pi05-rocm-jax-migration-training-and-evaluation.md) | 11 задач ловкой руки, ROCm JAX, Pi0.5 |
| 13 | [DISCOVERSE](./13-discoverse-rocm-data-generation-training-and-multi-view-video.md) | Экспертные траектории, конвертация в LeRobot, ACT/DP/3DGS |
| 14 | [RoboWits](./14-robowits-rocm-download-training-and-creative-task-evaluation.md) | Ограниченные данные, обучение и оценка творческих задач |
| 15 | [Unitree G1 с предсказательным CBF](./15-unitree-g1-prediction-cbf-rocm-reproduction.md) | Входы восприятия, награда безопасности и повтор позы всего тела |
| 16 | [Единая оценка, видео и архив](./16-unify-the-evaluation-videos-and-result-archiving.md) | `eval_info.json`, видео и сводные таблицы |

## Notebook и скрипты

- [Описание Notebook](./notebooks/README.md): разделяет готовые к чтению Notebook с результатами, шаблоны запуска и сквозные рабочие процессы, зависящие от вышестоящих проектов.
- [Рабочие процессы моделей](./notebooks/workflows/README.md): обычное обучение и защищённое обучение используют отдельные каталоги, а оценка читает модель, полученную в этом же запуске.
- [Скрипты Python](./code/README.md): пакетная замкнутая оценка, воспроизведение данных, демонстрационные видео и конвертация данных DISCOVERSE.
- [Структура вышестоящих проектов](./external/README.md): исходный код, данные, модели и результаты запуска выносятся за пределы каталога темы.

Notebook удобен для пошагового просмотра изображений, состояний и действий. Долгое обучение, пакетную оценку и массовую запись видео лучше запускать через скрипты: их легче возобновлять и сравнивать на удалённом устройстве.

## Единые артефакты

После каждого официального эксперимента сохраняйте как минимум следующие файлы:

| Файл | Содержимое |
| --- | --- |
| `run_config.json` | Версия вышестоящего проекта, модель, задача, наблюдения, действия, камеры и random seed |
| `metrics.jsonl` | Шаги обучения, функция потерь, learning rate, время и метрики устройства |
| `checkpoints/` | Загружаемые модели и конфигурации |
| `eval/<task>/eval_info.json` | Результаты по эпизодам и критерии успеха задачи |
| `eval/<task>/videos/` | Видео успешных и неуспешных попыток |
| `summary.json` | Процент успеха и статистика по фазам, агрегированные из результатов эпизодов |

Большие наборы данных, модели, кэши и пакетные видео храните на постоянном диске или в публичном репозитории моделей. Git-репозиторий должен содержать исходный код, небольшие конфигурации, короткие демонстрационные медиафайлы и документацию.

## Приложения

- [Работа с AUP, Jupyter и Code Server](./appendices/AUP_JUPYTER_CODE_SERVER.md)
- [Записи экспериментов с Pi0/Pi0.5](./appendices/PI0_PI05_EXPERIMENT_NOTES.md)
- [Указатель случаев отладки ROCm](./appendices/ROCM_DEBUG_CASES.md)

Приложения содержат ход экспериментов и углублённую диагностику. При первом изучении сначала пройдите основные главы, а затем переходите к приложению, относящемуся к вашей проблеме.
