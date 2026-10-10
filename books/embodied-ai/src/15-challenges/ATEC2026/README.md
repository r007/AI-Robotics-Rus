# ATEC2026, онлайн-этап: разбор после соревнования, публичные решения и точки входа для воспроизведения

Этот каталог — общая точка входа по ATEC2026 в 15-й главе Every Embodied. Здесь собраны определение задач, практика L0, разбор Task B по сбору мусора, анализ публичного решения Task D, воспроизведение Task E по уборке стола, лёгкий код и описания внешних открытых ресурсов.

Это не чемпионское решение и не гарантия какого-либо результата в онлайн-рейтинге. Цель главы — разложить реальное соревнование по воплощённому ИИ на материалы, которые можно читать, проверять и воспроизводить, а также явно отделить официальные доказательства, локальные результаты отладки и самостоятельные заявления внешних участников.

## Структура каталога

```text
15-challenges/ATEC2026/
├── README.md
├── l0-robot-walking/                 # Task A / воспроизведение и оценка B2Piper locomotion
├── l0-desktop-organization-taske/    # Task E / Piper: захват на столе и разбор ACT
├── TaskB-B2Piper/                    # Task B / B2-Piper: спецвыпуск по сбору мусора
├── docs/                             # Общие границы и документы о публикации для задач A/B/D/E
├── code/                             # Лёгкие адаптеры, аудит внешних референсных репозиториев и скрипты проверки
└── references/                       # Индекс внешних публичных решений, фиксированных commit и лицензий
```

## Обзор задач

| Задача | Робот / возможность | Вход в этом каталоге | Границы публичных выводов |
|---|---|---|---|
| Task A / L0 | B2Piper, прогулка по пересечённой местности | [`l0-robot-walking/`](./l0-robot-walking/) | основной акцент на официальном baseline, локальной оценке, отслеживании траектории и выборе checkpoint |
| Task B | B2-Piper, сбор мусора | [`TaskB-B2Piper/`](./TaskB-B2Piper/) | публично раскрыты полная отладка и границы наблюдений; проверенной полной финальной policy нет |
| Task D | толкание ящика и преодоление препятствий | [`docs/03-public-scheme-and-logic-tars.md`](./docs/03-public-scheme-and-logic-tars.md) | основной акцент на конечном автомате, LiDAR и адаптации действий публичного решения Logic-TARS |
| Task E / L0 | Piper, уборка стола | [`l0-desktop-organization-taske/`](./l0-desktop-organization-taske/) | лучший онлайн-результат ACT seed1 — `15.00`, это не решение с максимальным баллом |

## Рекомендуемый порядок чтения

1. [Определение задач и границы наблюдений](./docs/01-task-definition-and-observation-boundaries.md)
2. [Список проблем и решения](./docs/02-problem-list-and-solutions.md)
3. [Публичное решение и разбор Logic-TARS](./docs/03-public-scheme-and-logic-tars.md)
4. [Воспроизведение, публикация на HF и GitHub](./docs/04-reproduction-hf-and-github-release.md)
5. [Сверка архива и очистка локальных файлов](./docs/05-arrangement-of-reconciliation-and-local-cleanup.md)
6. [Архив по Task B](./TaskB-B2Piper/README.md)

## Публичные ресурсы

- Официальный репозиторий симуляции: [atecup/ATEC2026_Simulation_Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)
- Пакет модели B2Piper L0: [Datawhale/atec2026-b2piper-l0](https://huggingface.co/Datawhale/atec2026-b2piper-l0)
- Публичный архив воспроизведения Task B: [Datawhale/atec2026-task-b-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-b-reproducibility)
- Модель Task E: [Datawhale/atec2026-task-e-act-seed1-best](https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best)
- Данные и логи Task E: [Datawhale/atec2026-task-e-reproducibility](https://huggingface.co/datasets/Datawhale/atec2026-task-e-reproducibility)

## Соглашения о воспроизведении

- В GitHub хранятся только учебные материалы, лёгкий код, границы экспериментов, зафиксированные commit внешних репозиториев и обезличенный Workspace Memory.
- Крупные данные, веса моделей, снимки исходного кода официальных проектов, видео и логи выкладываются на Hugging Face с учётом лицензий и записью SHA-256.
- Для полной симуляции всё равно нужны подходящие Isaac Sim/Isaac Lab, GPU, зависимости Python, официальный runner и оценщик. Учебный материал с GitHub нельзя считать окружением для запуска в одну команду.
- Среда-камеры, истинные состояния, oracle prefix, скриптовый захват и конечный автомат стадий можно использовать для диагностики, но нужно явно указывать границы протокола. Их нельзя выдавать за официальный вход policy или полное доказательство успеха.
- Сырые кеши обучения, приватные логи, токены, учётные данные серверов и неавторизованные SDK в публичный репозиторий не попадают.
