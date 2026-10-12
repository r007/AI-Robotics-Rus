# Путь групповой учёбы, версия 6 (Datawhale)

#### **9) Введение в воплощённый ИИ для начинающих, v6**

- Открытое содержание: [Every-Embodied: Zero to Hero in Embodied AI](https://github.com/datawhalechina/every-embodied)
- База знаний по учебным материалам: [Учебное руководство](https://datawhale-eai.feishu.cn/wiki/QvxQwV2Qyij1NakV9IdcjTl1nJd) *(внутренний документ Feishu; ссылка не перенесена)*
- Вопросы: [Записи вопросов и сводка](https://datawhale-eai.feishu.cn/wiki/DBSpwQysHiO9MbkA24qc4Xrxndd) *(внутренний документ Feishu; ссылка не перенесена)*
- Отметки о выполнении, отличные работы и таблица взаимодействия ассистентов — по объявлению группы Datawhale текущего выпуска.

Этот путь рассчитан на новый выпуск групповой учёбы продолжительностью 20 дней. После V5 в репозитории появилось очень много материалов. Если продолжать раскладывать их по темам «VLA / зрение / CAD / симуляция / аппаратура / модели мира», новичкам будет трудно понять, с чего начать. Поэтому V6 организован по четырём основным возможностям воплощённой системы:

| Направление | Решаемая задача | Для кого подходит |
| :--- | :--- | :--- |
| **Управление действиями** | Как робот движется, захватывает и исполняет стратегию действий | Тем, кто хочет работать с манипуляторами, ACT, Pi0, SmolVLA, OpenVLA, PhysBrain, PRTS, G0.5, Dexora, LWD, VisualThink-VLA, Agentic-VLA, Dexbotic-RLinf, DM0.5, EVA-Client, выравниванием наград, LeRobot и реальным управлением |
| **Восприятие и картографирование** | Как робот видит, находит цели, понимает 3D-сцены и ресурсы | Тем, кто хочет работать с SAM, Locate Anything, LingBot-Map, 3D/CAD, Shape Your Body, реконструкцией сцен и проектированием роботов |
| **Навигация и планирование** | Как робот перемещается в среде, планирует маршрут, активно выбирает, куда смотреть, выполняет VLN или задачи дронов | Тем, кто хочет работать с Habitat, ETPNav, VLN, активным восприятием, NBS, управлением дронами и планированием пути |
| **Модели мира** | Как робот предсказывает будущее, синтезирует данные и оценивает стратегии «в воображении» | Тем, кто хочет работать с LeWM, RISE, RAW-Dream, WoG, WALL-WM, RoboDream, Gamma-World, BWM, τ0-WM, GE-Sim 2.0 |

Эти четыре направления не исключают друг друга. Полная робототехническая система в итоге использует одновременно управление, восприятие, планирование и модели мира. Но на этапе групповой учёбы рекомендуется сначала выбрать одно основное направление и пройти один понятный путь целиком.

## Информация о курсе

- Для кого: студенты, инженеры и исследователи, интересующиеся воплощённым ИИ; желательны основы Python и базовые навыки командной строки.
- Продолжительность: 20 дней, пять заданий Task по 4 дня каждое.
- Сроки: Task01 — до **17 июля, 03:00**; последующие задания идут с интервалом 4 дня.
- Минимальная цель: выполнить Task01 и в Task02–Task04 пройти хотя бы одно приоритетное задание в выбранном направлении.
- Рекомендуемая цель: выполнить одно приоритетное воспроизведение и одно продвинутое разбор, а в Task05 сдать один отчёт, пригодный для повторного использования.

| Роль | Имя |
| --- | --- |
| Первооткрыватель маршрута (разработчик курса) | Чэнь Кэвэй, Ли Юньди, Чжан Тяньи |
| Лоцман (операционный ассистент) | по объявлению текущего выпуска |
| Штурман (профильный ассистент) | по объявлению текущего выпуска |

## Как выбрать направление

Если направления ещё нет, его можно выбрать по таблице:

| Что вы хотите делать | Рекомендуемое направление | Первое задание |
| :--- | :--- | :--- |
| Заставить манипулятор взять стакан, переместить предмет, запустить ACT / VLA | Управление действиями | MuJoCo ACT/Pi0/SmolVLA или OpenVLA |
| Научить робота узнавать цели, оценивать глубину, реконструировать сцены, создавать 3D-модели | Восприятие и картографирование | SAM / оценка глубины, Locate Anything или Build123d |
| Заставить робота или дрон перемещаться по комнате, карте или маршруту и активно выбирать, куда смотреть | Навигация и планирование | Habitat Sim/Lab, ETPNav или активное восприятие NBS |
| Прочитать недавние статьи о моделях мира в робототехнике | Модели мира | Базовые VAE/DDPM, LeWM или обзор RoboDream/RAW-Dream |

Тем, у кого нет оборудования и мощного GPU, лучше выбрать **восприятие и картографирование** или **навигацию и планирование**. Тем, у кого есть опыт работы с манипулятором, RDK-X5, LeRobot или MuJoCo, лучше выбрать **управление действиями**. Тем, кто хочет читать статьи и разбирать методы, лучше выбрать **модели мира**.

## Общие правила отметок

В каждом Task нужно сдать результат только по одному направлению. Не пытайтесь пройти все ссылки.

1. **Приоритетное воспроизведение**: скриншоты, журналы, видео, результаты или краткий отчёт о воспроизведении.
2. **Разбор статьи**: описание задачи, входов и выходов, основных модулей и границ применимости метода.
3. **Диагностика окружения**: если запуск не удался, укажите версию системы, GPU/CPU, среду Python/Conda, место ошибки и план дальнейшей проверки.
4. **Итоговый отчёт**: не перечисляйте ссылки. Чётко опишите выбранный путь, доказательства выполнения, точки отказа и дальнейшие шаги.

## Распределение заданий

| Задание | Дней | Срок | Цель | Рекомендуемая работа |
| :--- | :--- | :--- | :--- | :--- |
| **Task01: общая основа и выбор направления** | 4 | **17 июля, 03:00** | Построить общий обзор воплощённого ИИ и выбрать основное направление | Запустить [быстрый старт проекта](https://github.com/datawhalechina/every-embodied/tree/main/examples) или прочитать [обзор воплощённого ИИ](../01-embodied-ai-overview/01-overview-of-embodied-ai.md) и [описание пространства робота и преобразование координат](../02-robot-basics-control-and-hand-eye-coordination/01-robot-space-description-and-coordinate-transformation.md), затем написать «какое направление я выбрал и почему» |
| **Task02: вход в направление** | 4 | **21 июля, 03:00** | Выполнить минимальную проверяемую задачу в выбранном направлении | Одно из четырёх направлений; см. пояснение Task02 ниже |
| **Task03: практика в направлении** | 4 | **25 июля, 03:00** | Запустить или подробно разобрать одну основную задачу и получить показываемый результат | Продолжать работу в направлении Task02; не рекомендуется менять направление на ходу |
| **Task04: углублённый разбор** | 4 | **29 июля, 03:00** | Выполнить более глубокое воспроизведение, разбор статьи или сравнительный анализ | Ответить на вопрос «что доказывает этот метод и что он не доказывает» |
| **Task05: систематизация результатов и обмен** | 4 | **2 августа, 03:00** | Подготовить итоговый отчёт, видеодемонстрацию, заметки о воспроизведении или обзор направления | Сдать полный результат, который сможет повторно использовать следующий выпуск |

## Task02: вход в направление

Цель Task02 — не сделать большой проект, а получить у каждого участника первый результат в выбранном направлении.

| Направление | Приоритетная задача | Рекомендуемая форма сдачи |
| :--- | :--- | :--- |
| Управление действиями | [Воспроизведение ACT, Pi0, SmolVLA в MuJoCo](../06-manipulation-and-vla/large-model-control-vla-vlm/04-mujoco-reproduction-act-pi0-smolvla/README.md) или [воспроизведение OpenVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/02-openvla-reproduction/02-openvla-reproduction.md) | Скриншот запуска окружения, один smoke test обучения или вывода, либо описание цепочки входов и выходов VLA |
| Восприятие и картографирование | [SAM и оценка глубины](../04-computer-vision-and-3d-reconstruction/01-sam-and-depth-estimation.md), [Locate Anything: визуально-языковая локализация](../04-computer-vision-and-3d-reconstruction/04-locate-anything-visual-language-localization.md) или [Build123d: введение в код-моделирование](../21-robot-design/01-build123d-code-modeling-introduction/README.md) | Результат сегментации или локализации, результат глубины, либо экспортированная 3D-модель |
| Навигация и планирование | [Практика Habitat Sim](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-sim-basic-practices.md) или [практика Habitat Lab](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-basic-practices.md) | Скриншот запуска среды навигации, кадр наблюдения, траектория или описание конфигурации |
| Модели мира | [VAE: подробное описание](../17-world-models/02-vae-explanation-and-code/vae-explanation-and-code.md), [DDPM: подробное описание](../17-world-models/03-ddpm-explanation-and-code/ddpm-explanation-and-code.md) или [анализ и воспроизведение LeWorldModel](../17-world-models/leworldmodel-analysis-and-reproduction/leworldmodel-analysis-and-reproduction.md) | Одна страница с конспектом метода: что предсказывает модель мира, каковы входы и выходы |

## Task03: практика в направлении

Task03 переходит к более полному воспроизведению или чтению метода. Выбирается по-прежнему одно направление.

### Управление действиями

Приоритетный выбор:

- [Воспроизведение OpenVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/02-openvla-reproduction/02-openvla-reproduction.md)
- [Воспроизведение ACT, Pi0, SmolVLA в MuJoCo](../06-manipulation-and-vla/large-model-control-vla-vlm/04-mujoco-reproduction-act-pi0-smolvla/README.md)
- [Подключение манипулятора LeRobot к RDK-X5 для телеуправления](../03-robot-hardware-lerobot-and-rdk-x5/03-rdk-x5-connects-to-the-lerobot-robotic-arm-for-remote-manipulation.md)
- [EVA-Client: инженерная навигация по реальному развёртыванию и оценке роботов](../06-manipulation-and-vla/large-model-control-vla-vlm/19-eva-client-real-robot-deployment-and-evaluation-engineering-navigation/README.md)
- [Robots That Know What to Ask: обзор выравнивания наград](../05-deep-and-reinforcement-learning/04-robots-that-know-what-to-ask-reward-alignment-introduction/README.md)

Главное в сдаче — объяснить цепочку «наблюдение → модель или стратегия → действие → исполнение или оценка». Для реальных задач можно сдать фото или видео телеуправления, записи сбора данных; для симуляционных — скриншоты обучения и вывода, а также разбор неудачных случаев. Направление выравнивания наград не требует обязательного воспроизведения на реальном манипуляторе; важно объяснить связь между демонстрацией, дисперсией признаков, выявлением недоопределённости, объяснительными вопросами и обновлением награды.

### Восприятие и картографирование

Приоритетный выбор:

- [Locate Anything: визуально-языковая локализация](../04-computer-vision-and-3d-reconstruction/04-locate-anything-visual-language-localization.md)
- [LingBot-Map: потоковая трёхмерная реконструкция по видео](../04-computer-vision-and-3d-reconstruction/03-lingbot-map-video-streaming-3d-reconstruction.md)
- [Text-to-CAD: инженерное моделирование](../21-robot-design/02-text-to-cad-engineering-modeling-introduction/README.md)
- [ForgeCAD: визуальная обратная инженерия](../21-robot-design/03-forgecad-visual-reverse-engineering-introduction/README.md)
- [Shape Your Body: обзор проектирования роботов по градиенту ценности](../21-robot-design/04-shape-your-body-value-gradient-robot-design-introduction/README.md)

Главное в сдаче — объяснить, какие входы, что делает модель или инструмент и как выход служит роботу. Направление CAD должно правильно описывать реальный процесс: естественный язык или эталонное изображение обычно не превращаются напрямую в CAD. Сначала создаётся параметрический исходный код, затем среда CAD экспортирует PNG/GIF/STEP/STL/3MF. Направление Shape Your Body должно объяснять связь между параметрами тела, симуляционным управлением, функцией ценности и проектированием по градиенту.

### Навигация и планирование

Приоритетный выбор:

- [Практика Habitat Sim](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-sim-basic-practices.md)
- [Практика Habitat Lab](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-basic-practices.md)
- [Воспроизведение ETPNav](../08-navigation-and-vln/03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md)
- [NBS: обзор активного восприятия и планирования пути](../08-navigation-and-vln/03-frontier-vln-reproduction/03-actively-perceive-nbs-guide/README.md)
- [Практика мультимодальной большой модели для дрона](../13-frontier-project-reproduction/drone-large-model-groundingdino-practice/unmanned-aerial-vehicle-multi-modal-large-model.md)

Главное в сдаче — объяснить связь между картой, наблюдением, целью, маршрутом, информационной выгодой и действием. Для задач навигации недостаточно проверить, достигнут ли конечный пункт; важно записывать причины неудач: дрейф локализации, недостаточность наблюдения, обход по лишнему пути, неясность языковой цели или неточная оценка фронтира и информационной выгоды в активном восприятии.

### Модели мира

Приоритетный выбор:

- [Анализ и воспроизведение LeWorldModel](../17-world-models/leworldmodel-analysis-and-reproduction/leworldmodel-analysis-and-reproduction.md)
- [RISE: самоулучшающаяся стратегия робота](../17-world-models/rise-self-improvement-robot-policy-reproduction/README.md)
- [RoboDream: компонуемая модель мира для синтеза данных](../17-world-models/robodream-compositional-world-model-data-synthesis-guide/README.md)
- [Gamma-World: многоагентная модель мира](../17-world-models/gamma-world-multi-agent-world-model-introduction/README.md)
- [BWM: модель мира с условием по действиям](../17-world-models/bwm-action-condition-world-model-guide/README.md)
- [τ0-WM: единая видео-действие модель мира](../17-world-models/tau0-wm-unified-video-action-world-model-introduction/README.md)
- [GE-Sim 2.0: замкнутый видеосимулятор мира](../17-world-models/ge-sim-v2-closed-loop-video-world-simulator-introduction/README.md)

Главное в сдаче — определить, к какому типу относится модель мира: предсказывает латентное состояние, предсказывает видео, выполняет RL в воображении, усиливает VLA, синтезирует демонстрационные данные, создаёт общий мир для нескольких агентов, выполняет видеопрогноз с условием по траекториям действий, объединяет видеопрогноз, генерацию действий и оценку действий во время теста или упаковывает модель мира как замкнутый обучаемый симулятор. Направление моделей мира не требует обязательного воспроизведения экспериментов; достаточно чётко описать границы метода.

## Task04: углублённый разбор

Task04 нужен для сравнения или глубокого разбора. Ниже даны рекомендуемые темы для каждого направления.

| Направление | Рекомендуемая тема | Рекомендуемая форма сдачи |
| :--- | :--- | :--- |
| Управление действиями | [DiT4DiT-LIBERO](../06-manipulation-and-vla/large-model-control-vla-vlm/05DiT4DiT-LIBERO/01-dit4dit-libero-training-and-evaluation.md), [EventVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/06-eventvla-visual-evidence-memory-guide/README.md), [WALL-OSS](../06-manipulation-and-vla/large-model-control-vla-vlm/07-wall-oss-open-source-vla-model-introduction/README.md), [WALL-X](../06-manipulation-and-vla/large-model-control-vla-vlm/08-wall-x-open-source-engineering-framework-navigation/README.md), [3DVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/09-3dvla-3d-spatial-and-instance-understanding-guide/README.md), [PhysBrain 1.0](../06-manipulation-and-vla/large-model-control-vla-vlm/10-physbrain-physical-common-sense-enhancement-vla-guide/README.md), [PRTS](../06-manipulation-and-vla/large-model-control-vla-vlm/11-prts-reinforcement-learning-native-vla-guidance/README.md), [Galaxea G0.5](../06-manipulation-and-vla/large-model-control-vla-vlm/12-galaxea-g0-5-autoregressive-vla-guidance/README.md), [Dexora](../06-manipulation-and-vla/large-model-control-vla-vlm/13-dexora-high-dof-dual-arm-dexterous-vla-guidance/README.md), [LWD](../06-manipulation-and-vla/large-model-control-vla-vlm/14-lwd-real-robot-cluster-reinforcement-learning-introduction/README.md), [VisualThink-VLA](../06-manipulation-and-vla/large-model-control-vla-vlm/15-visualthink-vla-visual-evidence-reasoning-guide/README.md), [Agentic-VLA](../06-manipulation-and-vla/large-model-control-vla-vlm/16-agentic-vla-online-adaptation-guide/README.md), [Dexbotic-RLinf](../06-manipulation-and-vla/large-model-control-vla-vlm/17-dexbotic-rlinf-engineering-vla-post-training-guide/README.md), [DM0.5](../06-manipulation-and-vla/large-model-control-vla-vlm/18-dm0-5-high-performance-inference-and-opendm-guidance/README.md), [EVA-Client](../06-manipulation-and-vla/large-model-control-vla-vlm/19-eva-client-real-robot-deployment-and-evaluation-engineering-navigation/README.md), [Robots That Know What to Ask](../05-deep-and-reinforcement-learning/04-robots-that-know-what-to-ask-reward-alignment-introduction/README.md) | Сравнить классический и новый VLA: представление действий, память, понимание 3D-экземпляров, предобучение на физическом здравом смысле, предобучение досягаемости целей / CRL, авторегрессионную цепочку рассуждений с потоком токенов действий, обучение высокоуровневых двурукой и многопалой манипуляций с учётом массы, групповое RL на реальных роботах (DIVL / QAM), токены визуальных доказательств / маршрутизированные доказательства / низколатентный вывод, онлайн-адаптацию агента, инженерное пост-обучение Dexbotic + RLinf, высокопроизводительный вывод DM0.5, аудируемую оценку EVA-Client, выравнивание недоопределённых демонстраций и наград, инженерные каркасы, границы открытого кода |
| Восприятие и картографирование | [Locate Anything и YOLO](../04-computer-vision-and-3d-reconstruction/04-locate-anything-visual-language-localization.md), [потоковое картографирование LingBot-Map](../04-computer-vision-and-3d-reconstruction/03-lingbot-map-video-streaming-3d-reconstruction.md), [случаи ForgeCAD: 3D-принтер, клавиатура, кисть](../21-robot-design/03-forgecad-visual-reverse-engineering-introduction/README.md), [проектирование робота по градиенту ценности Shape Your Body](../21-robot-design/04-shape-your-body-value-gradient-robot-design-introduction/README.md) | Представить набор «вход — выход» и объяснить смысл каждого промежуточного результата; для Shape Your Body подробно описать multi-embodiment policy/value, VGDS и диагностику конструкции |
| Навигация и планирование | [базовые задачи Habitat](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-sim-basic-practices.md) против [ETPNav](../08-navigation-and-vln/03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md), либо [активное планирование NBS](../08-navigation-and-vln/03-frontier-vln-reproduction/03-actively-perceive-nbs-guide/README.md), либо цепочка [управления дронами](../18-drones/drone-control-tutorial.md) / [планирования дронов](../18-drones/drone-planning-tutorial.md) | Сдать траектории, конфигурацию, примеры успешных и неудачных случаев и разбор модулей планирования; для NBS подробно объяснить node-wise beam search, expected gain и RRAG |
| Модели мира | [RAW-Dream](../17-world-models/raw-dream-task-agnostic-world-model-reinforcement-for-vla/README.md), [WoG](../17-world-models/wog-conditional-space-world-model-guide/README.md), [WALL-WM](../17-world-models/wall-wm-event-level-world-action-model-guide/README.md), [RoboDream](../17-world-models/robodream-compositional-world-model-data-synthesis-guide/README.md), [Gamma-World](../17-world-models/gamma-world-multi-agent-world-model-introduction/README.md), [BWM](../17-world-models/bwm-action-condition-world-model-guide/README.md), [τ0-WM](../17-world-models/tau0-wm-unified-video-action-world-model-introduction/README.md), [GE-Sim 2.0](../17-world-models/ge-sim-v2-closed-loop-video-world-simulator-introduction/README.md) | Сравнить «модели мира с возможностью rollout», «модели мира в условном пространстве», «событийные модели мира действий», «модели мира синтеза данных», «многоагентные общие модели мира», «видеомодели мира с условием по действиям», «единые видео-действие модели мира» и «замкнутые видеосимуляторы мира» |

Если есть вычислительные ресурсы или аппаратура, можно выбрать такие продвинутые практики:

- [SIM1: симуляция мягких тел и генерация данных](../10-simulation-tools-and-frontiers/09-sim1-soft-body-simulation-and-data-generation/01-sim1-environment-configuration-and-operation.md)
- [UniLab + MotrixSim: гетерогенное RL для роботов](../10-simulation-tools-and-frontiers/11-unilab-motrixsim-heterogeneous-rl-training/README.md)
- [Genesis World 1.0: полный опыт и конвейер симуляции робота](../10-simulation-tools-and-frontiers/genesis-simulation-environment-configuration/03-genesis-world-1-0-complete-experience-and-robot-simulation-pipeline.md)
- [Тематика AMD ROCm: воспроизведение стратегий](../16-study-programs/04-amd-rocm-policy-replication-topic/README.md)
- [AGILE: loco-manipulation человекоподобного робота](../05-deep-and-reinforcement-learning/03-agile-humanoid-robot-loco-manipulation-reproduction/README.md)
- [HIMLoco: управление движением четырёхногого робота](../05-deep-and-reinforcement-learning/02-himloco-isaaclab-reproduction/README.md)
- [HumanoidMimicGen: генерация данных полнотелого планирования](../10-simulation-tools-and-frontiers/12-humanoidmimicgen-full-body-planning-data-generation-introduction/README.md)

## Task05: систематизация результатов и обмен

На последнем этапе не стоит открывать новые среды. Нужно оформить пройденное в виде результатов, пригодных для повторного использования.

Рекомендуемая структура отчёта:

1. Выбранное направление: управление действиями / восприятие и картографирование / навигация и планирование / модели мира.
2. Выполненное приоритетное задание: что запущено, что прочитано, какой результат.
3. Ключевые доказательства: скриншоты, журналы, видео, сгенерированные изображения, траектории, 3D-модели или таблицы.
4. Понимание метода: входы, выходы, основные модули, метрики оценки и границы применимости.
5. Заметки об ошибках: окружение, зависимости, видеопамять, данные, пути, оборудование или понимание статьи.
6. Советы следующему выпуску: что делать сначала, что потом, что можно пропустить.

## Четыре примера маршрутов

### Маршрут A: управление действиями

Task01 — запустить минимальную демонстрацию проекта; Task02 — выбрать MuJoCo или OpenVLA; Task03 — выполнить воспроизведение ACT/Pi0/SmolVLA или OpenVLA; Task04 — прочитать [DiT4DiT](../06-manipulation-and-vla/large-model-control-vla-vlm/05DiT4DiT-LIBERO/01-dit4dit-libero-training-and-evaluation.md), [EventVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/06-eventvla-visual-evidence-memory-guide/README.md), [WALL-OSS](../06-manipulation-and-vla/large-model-control-vla-vlm/07-wall-oss-open-source-vla-model-introduction/README.md), [3DVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/09-3dvla-3d-spatial-and-instance-understanding-guide/README.md), [PhysBrain 1.0](../06-manipulation-and-vla/large-model-control-vla-vlm/10-physbrain-physical-common-sense-enhancement-vla-guide/README.md), [PRTS](../06-manipulation-and-vla/large-model-control-vla-vlm/11-prts-reinforcement-learning-native-vla-guidance/README.md), [Galaxea G0.5](../06-manipulation-and-vla/large-model-control-vla-vlm/12-galaxea-g0-5-autoregressive-vla-guidance/README.md), [Dexora](../06-manipulation-and-vla/large-model-control-vla-vlm/13-dexora-high-dof-dual-arm-dexterous-vla-guidance/README.md), [LWD](../06-manipulation-and-vla/large-model-control-vla-vlm/14-lwd-real-robot-cluster-reinforcement-learning-introduction/README.md), [VisualThink-VLA](../06-manipulation-and-vla/large-model-control-vla-vlm/15-visualthink-vla-visual-evidence-reasoning-guide/README.md), [Agentic-VLA](../06-manipulation-and-vla/large-model-control-vla-vlm/16-agentic-vla-online-adaptation-guide/README.md), [Dexbotic-RLinf](../06-manipulation-and-vla/large-model-control-vla-vlm/17-dexbotic-rlinf-engineering-vla-post-training-guide/README.md), [DM0.5](../06-manipulation-and-vla/large-model-control-vla-vlm/18-dm0-5-high-performance-inference-and-opendm-guidance/README.md), [EVA-Client](../06-manipulation-and-vla/large-model-control-vla-vlm/19-eva-client-real-robot-deployment-and-evaluation-engineering-navigation/README.md) или [Robots That Know What to Ask](../05-deep-and-reinforcement-learning/04-robots-that-know-what-to-ask-reward-alignment-introduction/README.md); Task05 — сдать отчёт «VLA / стратегии манипуляций / инженерия пост-обучения / высокопроизводительный вывод / реальное развёртывание и оценка / разбор выравнивания наград».

### Маршрут B: восприятие и картографирование

Task01 — понять структуру воплощённой системы; Task02 — выполнить SAM / оценку глубины или Locate Anything; Task03 — выполнить LingBot-Map или ForgeCAD; Task04 — сделать [Locate Anything и YOLO](../04-computer-vision-and-3d-reconstruction/04-locate-anything-visual-language-localization.md), [Text-to-CAD](../21-robot-design/02-text-to-cad-engineering-modeling-introduction/README.md), [3D-модели](../21-robot-design/03-forgecad-visual-reverse-engineering-introduction/README.md) или [Shape Your Body](../21-robot-design/04-shape-your-body-value-gradient-robot-design-introduction/README.md); Task05 — сдать отчёт «как робот понимает сцену и формирует 3D-ресурсы или параметры конструкции».

### Маршрут C: навигация и планирование

Task01 — понять систему робота и преобразования координат; Task02 — запустить Habitat Sim/Lab; Task03 — выполнить ETPNav, [активное восприятие NBS](../08-navigation-and-vln/03-frontier-vln-reproduction/03-actively-perceive-nbs-guide/README.md) или мультимодальную навигацию дрона; Task04 — разобрать [Habitat/ETPNav](../08-navigation-and-vln/03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md), [NBS](../08-navigation-and-vln/03-frontier-vln-reproduction/03-actively-perceive-nbs-guide/README.md), [управление дронами](../18-drones/drone-control-tutorial.md) или [планирование дронов](../18-drones/drone-planning-tutorial.md): планирование пути, информационную выгоду, языковые цели и траектории неудач; Task05 — сдать разбор «от наблюдения до действия в системе навигации».

### Маршрут D: модели мира

Task01 — понять различие между моделью мира и VLA; Task02 — изучить VAE/DDPM или LeWM; Task03 — прочитать RISE, RoboDream, Gamma-World, BWM, τ0-WM или GE-Sim 2.0; Task04 — сравнить [RAW-Dream](../17-world-models/raw-dream-task-agnostic-world-model-reinforcement-for-vla/README.md), [WoG](../17-world-models/wog-conditional-space-world-model-guide/README.md), [WALL-WM](../17-world-models/wall-wm-event-level-world-action-model-guide/README.md), [Gamma-World](../17-world-models/gamma-world-multi-agent-world-model-introduction/README.md), [BWM](../17-world-models/bwm-action-condition-world-model-guide/README.md), [τ0-WM](../17-world-models/tau0-wm-unified-video-action-world-model-introduction/README.md) и [GE-Sim 2.0](../17-world-models/ge-sim-v2-closed-loop-video-world-simulator-introduction/README.md); Task05 — сдать отчёт «что именно предсказывает модель мира и как она служит роботу».

## Указатель материалов

| Направление | Рекомендуемые точки входа |
| :--- | :--- |
| Управление действиями | [MuJoCo ACT/Pi0/SmolVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/04-mujoco-reproduction-act-pi0-smolvla/README.md), [OpenVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/02-openvla-reproduction/02-openvla-reproduction.md), [DiT4DiT-LIBERO](../06-manipulation-and-vla/large-model-control-vla-vlm/05DiT4DiT-LIBERO/01-dit4dit-libero-training-and-evaluation.md), [EventVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/06-eventvla-visual-evidence-memory-guide/README.md), [WALL-OSS](../06-manipulation-and-vla/large-model-control-vla-vlm/07-wall-oss-open-source-vla-model-introduction/README.md), [WALL-X](../06-manipulation-and-vla/large-model-control-vla-vlm/08-wall-x-open-source-engineering-framework-navigation/README.md), [3DVLA](../06-manipulation-and-vla/large-model-control-vla-vlm/09-3dvla-3d-spatial-and-instance-understanding-guide/README.md), [PhysBrain 1.0](../06-manipulation-and-vla/large-model-control-vla-vlm/10-physbrain-physical-common-sense-enhancement-vla-guide/README.md), [PRTS](../06-manipulation-and-vla/large-model-control-vla-vlm/11-prts-reinforcement-learning-native-vla-guidance/README.md), [Galaxea G0.5](../06-manipulation-and-vla/large-model-control-vla-vlm/12-galaxea-g0-5-autoregressive-vla-guidance/README.md), [Dexora](../06-manipulation-and-vla/large-model-control-vla-vlm/13-dexora-high-dof-dual-arm-dexterous-vla-guidance/README.md), [LWD](../06-manipulation-and-vla/large-model-control-vla-vlm/14-lwd-real-robot-cluster-reinforcement-learning-introduction/README.md), [VisualThink-VLA](../06-manipulation-and-vla/large-model-control-vla-vlm/15-visualthink-vla-visual-evidence-reasoning-guide/README.md), [Agentic-VLA](../06-manipulation-and-vla/large-model-control-vla-vlm/16-agentic-vla-online-adaptation-guide/README.md), [Dexbotic-RLinf](../06-manipulation-and-vla/large-model-control-vla-vlm/17-dexbotic-rlinf-engineering-vla-post-training-guide/README.md), [DM0.5](../06-manipulation-and-vla/large-model-control-vla-vlm/18-dm0-5-high-performance-inference-and-opendm-guidance/README.md), [EVA-Client](../06-manipulation-and-vla/large-model-control-vla-vlm/19-eva-client-real-robot-deployment-and-evaluation-engineering-navigation/README.md), [Robots That Know What to Ask](../05-deep-and-reinforcement-learning/04-robots-that-know-what-to-ask-reward-alignment-introduction/README.md), [HumanoidMimicGen](../10-simulation-tools-and-frontiers/12-humanoidmimicgen-full-body-planning-data-generation-introduction/README.md) |
| Восприятие и картографирование | [SAM и оценка глубины](../04-computer-vision-and-3d-reconstruction/01-sam-and-depth-estimation.md), [Locate Anything](../04-computer-vision-and-3d-reconstruction/04-locate-anything-visual-language-localization.md), [LingBot-Map](../04-computer-vision-and-3d-reconstruction/03-lingbot-map-video-streaming-3d-reconstruction.md), [Build123d](../21-robot-design/01-build123d-code-modeling-introduction/README.md), [Text-to-CAD](../21-robot-design/02-text-to-cad-engineering-modeling-introduction/README.md), [ForgeCAD](../21-robot-design/03-forgecad-visual-reverse-engineering-introduction/README.md), [Shape Your Body](../21-robot-design/04-shape-your-body-value-gradient-robot-design-introduction/README.md) |
| Навигация и планирование | [Habitat Sim](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-sim-basic-practices.md), [Habitat Lab](../08-navigation-and-vln/02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-basic-practices.md), [ETPNav](../08-navigation-and-vln/03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md), [NBS: активное планирование пути](../08-navigation-and-vln/03-frontier-vln-reproduction/03-actively-perceive-nbs-guide/README.md), [мультимодальная большая модель для дрона](../13-frontier-project-reproduction/drone-large-model-groundingdino-practice/unmanned-aerial-vehicle-multi-modal-large-model.md), [управление дронами](../18-drones/drone-control-tutorial.md), [планирование дронов](../18-drones/drone-planning-tutorial.md) |
| Модели мира | [VAE](../17-world-models/02-vae-explanation-and-code/vae-explanation-and-code.md), [DDPM](../17-world-models/03-ddpm-explanation-and-code/ddpm-explanation-and-code.md), [LeWM](../17-world-models/leworldmodel-analysis-and-reproduction/leworldmodel-analysis-and-reproduction.md), [RISE](../17-world-models/rise-self-improvement-robot-policy-reproduction/README.md), [RAW-Dream](../17-world-models/raw-dream-task-agnostic-world-model-reinforcement-for-vla/README.md), [WoG](../17-world-models/wog-conditional-space-world-model-guide/README.md), [WALL-WM](../17-world-models/wall-wm-event-level-world-action-model-guide/README.md), [RoboDream](../17-world-models/robodream-compositional-world-model-data-synthesis-guide/README.md), [Gamma-World](../17-world-models/gamma-world-multi-agent-world-model-introduction/README.md), [BWM](../17-world-models/bwm-action-condition-world-model-guide/README.md), [τ0-WM](../17-world-models/tau0-wm-unified-video-action-world-model-introduction/README.md), [GE-Sim 2.0](../17-world-models/ge-sim-v2-closed-loop-video-world-simulator-introduction/README.md) |
