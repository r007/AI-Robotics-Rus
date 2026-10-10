# Выбор платформы роботизированной симуляции и навигатор по разделу

Эта страница помогает читателю сначала решить, какую симуляционную платформу изучать, а затем перейти к соответствующему руководству по установке или практике. Раньше здесь были смешаны отдельные заметки по установке DISCOVERSE; теперь страница сделана как сравнение платформ и навигатор по путям обучения, чтобы базовая настройка главы не выглядела привязанной к одному проекту.

## Сравнение распространённых платформ симуляции

| Платформа | Основное назначение | Подходящие задачи | Место в проекте |
| --- | --- | --- | --- |
| MuJoCo | Лёгкая, эффективная, удобна для управления | Управление, манипуляторы, имитационное обучение, небольшие замкнутые эксперименты | [examples/README.md](https://github.com/datawhalechina/every-embodied/tree/main/examples), [README.md](https://github.com/datawhalechina/every-embodied) |
| Robosuite + LIBERO | Связка на основе MuJoCo для исследований манипуляции и VLA | Манипуляция, VLA, бенчмарк непрерывного обучения | [SmolVLA-LIBERO](../06-manipulation-and-vla/large-model-control-vla-vlm/01SmolVLA-LIBERO/01SmolVLA-libero.md), [бенчмарк LIBERO](../09-data-and-benchmarks/01-libero.md) |
| Isaac Sim | Фотореалистичный рендеринг и сложная симуляция роботов | Синтетические данные, ROS2, Sim2Real, сложные сцены | [Навигатор по развёртыванию Isaac Sim, Isaac Lab и GR00T](01-isaac-deployment-and-gr00t-practices/00-isaac-deployment-navigation.md), [Руководство по локальной и облачной настройке Isaac Sim](01-isaac-deployment-and-gr00t-practices/01-isaac-sim-local-and-cloud-configuration.md) |
| Isaac Lab | Основная линия RL / IL в экосистеме Isaac | RL, IL, обучение политик, построение задач | [Руководство по локальной и облачной настройке Isaac Sim](01-isaac-deployment-and-gr00t-practices/01-isaac-sim-local-and-cloud-configuration.md), [Полное руководство по Isaac Lab + GR00T в Alibaba Cloud](01-isaac-deployment-and-gr00t-practices/02-complete-tutorial-on-deploying-isaac-lab-gr00t-in-alibaba-cloud.md) |
| Habitat-Sim / Habitat-Lab | Платформа для навигации в помещениях и VLN | PointNav, ObjectNav, VLN | [Среда навигации Habitat](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-base.md) |
| ManiSkill | Платформа параллельной GPU-манипуляции и RL | Манипуляция, PPO, сбор данных, построение задач | [Конфигурация симуляции ManiSkill](02-maniskill-environment-simulation-configuration.md), [Подробная документация ManiSkill](maniskill-detailed-documentation) |
| GenieSim / GenieSim3 | Высокореалистичная симуляция манипуляции и бенчмарк | Pi0, оценка политик, задачи высокой точности | [Конфигурация GenieSim](07-geniesim-configuration.md), [Конфигурация GenieSim3](08-geniesim3-configuration.md) |
| OmniGibson / BEHAVIOR | Длинные домашние задачи и сложные взаимодействия | Домашние сцены, мобильная манипуляция, длинные задачи | [Конфигурация среды домашнего робота](04-household-robot-environment-configuration.md) |
| Genesis | Новая высокоскоростная платформа физической симуляции | Быстрая симуляция, многофизические задачи, исследования | [Конфигурация Genesis](genesis-simulation-environment-configuration/01-environment-configuration-and-testing.md), [Визуализация и рендеринг в Genesis](genesis-simulation-environment-configuration/02-visualization-and-rendering.md) |
| Gazebo | Системная проверка ROS / ROS2 | Симуляция датчиков, интеграция мобильных роботов | Не является основной линией этого репозитория, но часто встречается в инженерных цепочках |
| AirSim | Историческая платформа для дронов | Дроны, демонстрации визуальной навигации | [Практика мультимодальной большой модели для дронов](../13-frontier-project-reproduction/drone-large-model-groundingdino-practice/unmanned-aerial-vehicle-multi-modal-large-model.md) |
| DISCOVERSE | MuJoCo + рендеринг 3DGS высокой точности + Real2Sim2Real | Высокоточный рендеринг, реконструкция реальных сцен, исследования Real2Sim2Real | Полезна как расширенная платформа для ознакомления, страница проекта: [DISCOVERSE](https://air-discoverse.github.io/) |
| SIM1 | Манипуляция двумя руками с мягкими телами и генерация данных | Работа с тканью, телеуправление, воспроизведение, генерация траекторий диффузией | [SIM1: симуляция мягких тел и генерация данных](09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md) |

## Рекомендации по выбору

Если цель — быстро разобраться в управлении манипулятором, захвате и базовой реализации политик, начните с MuJoCo, Robosuite + LIBERO или ManiSkill. Порог входа у них относительно низкий, и они подходят для обучения и проверки алгоритмов.

Если цель — высокоточный рендеринг, синтетические данные, интеграция с ROS2 или сложные роботизированные системы, начните с Isaac Sim и Isaac Lab. Они требовательнее к видеокарте, драйверам и согласованности окружения, зато ближе к реальным инженерным развёртываниям.

Если цель — навигация в помещениях, языковая навигация и планирование путей, начните с Habitat-Sim / Habitat-Lab. Навигацию дронов тоже можно отнести сюда, потому что основные задачи — понимание среды, локализация цели и планирование пути.

Если цель — домашние роботы, длинные задачи, взаимодействие со многими объектами и домашние сцены, смотрите OmniGibson / BEHAVIOR или конфигурацию среды домашнего робота.

Если цель — знакомство с новыми платформами или исследовательский поиск, можно изучить Genesis, DISCOVERSE, SIM1, RoboTwin, MetaSim / RoboVerse и другие расширенные платформы.

## Рекомендуемый путь изучения

1. Сначала прочитайте эту страницу и определите, нужны ли вам манипуляция, навигация, системное развёртывание или обзор новых платформ.
2. Для направления манипуляции и обучения с подкреплением: сначала [конфигурация симуляции ManiSkill](02-maniskill-environment-simulation-configuration.md), затем [подробная документация ManiSkill](maniskill-detailed-documentation).
3. Для направления высокой точности и системной интеграции: сначала [навигатор по развёртыванию Isaac Sim, Isaac Lab и GR00T](01-isaac-deployment-and-gr00t-practices/00-isaac-deployment-navigation.md), затем выберите путь для локального Windows, рабочей станции Linux, облачного сервера или воспроизведения GR00T.
4. Для направления навигации: сначала [среда навигации Habitat](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-base.md) и [воспроизведение ETPNav](../08-navigation-and-vln/03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md).
5. Для прикладного направления: в зависимости от задачи выберите [конфигурацию среды домашнего робота](04-household-robot-environment-configuration.md), [мультимодальную большую модель для дронов](../13-frontier-project-reproduction/drone-large-model-groundingdino-practice/unmanned-aerial-vehicle-multi-modal-large-model.md) или [SIM1: симуляцию мягких тел и генерацию данных](09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md).

## Соглашения по оформлению

Следующие соглашения помогают сохранить правильный порядок разделов в веб-оглавлении:

- Каждый Markdown-файл содержит только один заголовок первого уровня, который отражает тему документа.
- Порядок глав задают имена файлов, например `01...md`, `02...md`; в боковой панели сайта заголовки лучше брать из имён файлов.
- Внутри текста можно использовать локальную нумерацию вида `1. Установка`, `2. Быстрый старт`, но не следует жёстко записывать сквозную нумерацию глав в заголовки.
- Сравнение платформ и выбор направления — на этой странице; конкретные команды установки — в документах по самим платформам.
- Внешние проекты вроде DISCOVERSE описываются здесь только в таблице сравнения, если для них не добавлено отдельное полное руководство.
