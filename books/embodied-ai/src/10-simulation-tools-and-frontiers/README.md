# Руководство по роботизированной симуляции

Этот каталог содержит описание выбора симуляционных платформ, руководства по настройке окружений, практические документы по платформам и материалы соревнований.

## Входные документы

- [Конвейер генерации ассетов PhysX-Omni с цветными текстурами](14-physx-omni-color-asset-generation-practice/README.md) — генерация деталей по одному изображению, совместимость с окружением Blackwell, проверка весов и границы приёмки по рендерингу Isaac Sim и коллизиям робота.

- [Выбор платформы симуляции и навигатор по каталогу](01-simulation-configuration-document.md) — сравнение MuJoCo, Isaac Sim, ManiSkill, Habitat, GenieSim, Genesis, AirSim и других платформ: их назначение и подходящие задачи.
- [Настройка симуляции ManiSkill](02-maniskill-environment-simulation-configuration.md) — руководство по созданию окружения симуляции ManiSkill.
- [Навигатор по развёртыванию Isaac Sim, Isaac Lab и GR00T](01-isaac-deployment-and-gr00t-practices/00-isaac-deployment-navigation.md) — единый обзор путей: локальный Windows, рабочая станция Linux, облачный сервер, Docker, micromamba/venv, Isaac Lab и GR00T.
- [Настройка среды домашнего робота](04-household-robot-environment-configuration.md) — настройка симуляции домашнего робота.
- [SIM1: симуляция мягких тел и генерация данных](09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md) — телеуправление двумя руками при работе с тканью, воспроизведение, генерация траекторий диффузией и процесс фильтрации.
- [UniLab + MotrixSim: воспроизведение RL-обучения гетерогенных роботов](11-unilab-motrixsim-heterogeneous-rl-training/README.md) — запуск бэкенда MotrixSim в UniLab на видеокарте 6 ГБ, обучение PPO, мониторинг ресурсов и воспроизведение видео.
- [HumanoidMimicGen: обзор генерации данных полнотелого планирования](12-humanoidmimicgen-full-body-planning-data-generation-introduction/README.md) — разбор метода, который по небольшому числу демонстраций телеуправления человекоподобного робота создаёт тысячи примеров loco-manipulation: аннотация навыков, полнотелая IK, планирование движения, бенчмарк G1 и совместное обучение на симуляции и реальности.
- [PhysicsNeMo: обзор решателя физического ИИ](13-physicsnemo-physical-ai-solver-introduction/README.md) — фреймворк Physics-ML от NVIDIA (после миграции из Modulus), границы между нейронными операторами / суррогатами PINN и симуляторами взаимодействия роботов.

## Платформенные разделы

- [Подробная документация ManiSkill](maniskill-detailed-documentation/01-task-construction.md) — подробное руководство по платформе ManiSkill.
- [Руководство по локальной и облачной настройке Isaac Sim](01-isaac-deployment-and-gr00t-practices/01-isaac-sim-local-and-cloud-configuration.md) — локальные рабочие станции, облачные серверы, Docker, pip и micromamba.
- [Развёртывание Isaac Lab + GR00T в облаке](01-isaac-deployment-and-gr00t-practices/02-complete-tutorial-on-deploying-isaac-lab-gr00t-in-alibaba-cloud.md) — исторический путь развёртывания Isaac Lab и GR00T в Alibaba Cloud.
- [Настройка Genesis](genesis-simulation-environment-configuration/01-environment-configuration-and-testing.md) — настройка платформы симуляции Genesis.
- [Визуализация и рендеринг в Genesis](genesis-simulation-environment-configuration/02-visualization-and-rendering.md) — практика визуализации и рендеринга в Genesis.
- [Genesis World 1.0: полный практический обзор](genesis-simulation-environment-configuration/03-genesis-world-1-0-complete-experience-and-robot-simulation-pipeline.md) — от официальной архитектуры, окружения Blackwell / CUDA и симуляции Franka до высококачественного рендеринга Nyx и границ импорта ассетов интерьера.
- [Настройка GenieSim](07-geniesim-configuration.md) — настройка окружения GenieSim.
- [Настройка GenieSim3](08-geniesim3-configuration.md) — настройка окружения GenieSim3.
- [UniLab + MotrixSim: гетерогенное обучение](11-unilab-motrixsim-heterogeneous-rl-training/README.md) — практика разделения симуляции на CPU и обучения на GPU.
- [HumanoidMimicGen: генерация данных полнотелого планирования](12-humanoidmimicgen-full-body-planning-data-generation-introduction/README.md) — генерация данных для полнотелых операций человекоподобного робота, бенчмарк MuJoCo / robosuite и связь с GR00T и обучением политик.
- [PhysicsNeMo: решатель физического ИИ](13-physicsnemo-physical-ai-solver-introduction/README.md) — расширенный инструмент для CFD, конструкций, погоды, суррогатов PDE и Physics-ML; не является основным направлением симуляции роботов.

## Ресурсы симуляции

- Ресурсы (`assets/`) — изображения и файлы ресурсов, связанные с симуляцией.
- Соревнования по симуляции (`challenge-competition/`) — материалы соревнований по симуляции роботов.

## Рекомендуемый путь изучения

1. Сначала прочитайте [выбор платформы симуляции и навигатор по каталогу](01-simulation-configuration-document.md), чтобы определить, относится ли ваша задача к манипуляции, навигации, системному развёртыванию или обзору новых платформ.
2. Для направления манипуляции и обучения с подкреплением изучите [настройку ManiSkill](02-maniskill-environment-simulation-configuration.md), затем [подробную документацию ManiSkill](maniskill-detailed-documentation/01-task-construction.md).
3. Для направления высокой точности рендеринга и системной интеграции изучите [навигатор по развёртыванию Isaac Sim, Isaac Lab и GR00T](01-isaac-deployment-and-gr00t-practices/00-isaac-deployment-navigation.md), затем выберите путь в зависимости от локального Windows, рабочей станции Linux, облачного сервера или воспроизведения GR00T.
4. Для направления домашних задач и сложных взаимодействий изучите [настройку среды домашнего робота](04-household-robot-environment-configuration.md).
5. Для направления работы с мягкими телами изучите [SIM1: симуляцию мягких тел и генерацию данных](09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md).
6. Для направления инфраструктуры RL-обучения изучите [UniLab + MotrixSim: воспроизведение RL-обучения гетерогенных роботов](11-unilab-motrixsim-heterogeneous-rl-training/README.md), чтобы понять, как разделяются физическая симуляция на CPU и обучение политик на GPU.
7. Для направления данных полнотелых операций человекоподобных роботов прочитайте [HumanoidMimicGen: обзор генерации данных полнотелого планирования](12-humanoidmimicgen-full-body-planning-data-generation-introduction/README.md), чтобы понять, как небольшое число демонстраций телеуправления расширяется до данных для обучения политик через ограничения навыков, полнотелое планирование и рандомизацию.
8. Для направления расширения физического ИИ прочитайте [PhysicsNeMo: обзор решателя физического ИИ](13-physicsnemo-physical-ai-solver-introduction/README.md), чтобы понять различие между нейронными операторами, PINN-суррогатами и симуляторами роботов.

## Требования к окружению

Требования разных платформ сильно различаются; ориентируйтесь на документы по настройке каждой из них. В общем случае нужны:

- Python 3.8 или выше либо версия, указанная платформой.
- CUDA и драйвер NVIDIA для GPU-симуляции или рендеринга высокого качества.
- Зависимости платформ MuJoCo, Isaac Sim, ManiSkill, Habitat и других.
- Видеокарта NVIDIA; для сложного рендеринга и крупномасштабной параллельной симуляции рекомендуется больше видеопамяти.
