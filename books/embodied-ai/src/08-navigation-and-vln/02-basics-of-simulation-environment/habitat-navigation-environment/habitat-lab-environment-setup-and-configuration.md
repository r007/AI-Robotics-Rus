# Основы симуляции Habitat-Lab

### Habitat-Lab — это алгоритмический уровень поверх Habitat-Sim. Он инкапсулирует стандартизированные задачи, метрики оценки и API, поэтому исследователь может не вникать в детали нижнего уровня симуляции и сосредоточиться на проектировании, обучении и оценке алгоритмов воплощённого ИИ.

## 1. Особенности Habitat-Lab

### 1. Предопределённые воплощённые задачи

* Навигационные: PointGoalNav (навигация к точке), ObjectNav (навигация к объекту), VLN (визуально-языковая навигация).
* Интерактивные: Rearrange (перестановка объектов), PickPlace (захват и укладка), OpenDoor (открывание двери).
* Поддерживаются пользовательские задачи (например, совместная работа нескольких агентов или длинное планирование).

### 2. Стандартизированная система оценки

* Встроенные основные метрики (успешность SR, эффективность длины пути SPL, оценка выравнивания DTW и др.) позволяют напрямую сравнивать алгоритмы.
* Поддерживается согласование с публичными бенчмарками (например, Habitat Challenge), что упрощает воспроизведение статей и сопоставление результатов.

### 3. Модульная архитектура агента

* Цепочка «датчики → энкодер → стратегия → контроллер» разделена, поэтому компоненты легко заменить (например, использовать CNN или Transformer для визуального кодирования, а RL или IL — для принятия решений).
* Глубокая интеграция с PyTorch поддерживает обучение с подкреплением (RL), имитационное обучение (IL) и сквозное глубокое обучение.

### 4. Удобный API

* Лаконичный Python-интерфейс: одной командой загружается сцена, инициализируется агент и запускается цикл симуляции.

## 2. Установка окружения Habitat-Lab

Можно использовать conda-окружение, созданное для habitat-sim:

```python
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

При загрузке habitat-lab нужно следить за совпадением версии с установленным habitat-sim. В этом руководстве используется habitat-sim 0.2.5, значит, нужно скачать habitat-lab версии 0.2.5. При воспроизведении других статей тоже сверяйте версии habitat-sim и habitat-lab.

```python
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

Одновременно установите habitat-baselines:

```python
pip install -e habitat-baselines
```

Загрузите трёхмерные данные сцен и данные навигации к точке:

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
```

## 3. Введение в habitatlab_test.py и базовая конфигурация YAML

### В этом разделе на примере интерактивного навигационного кода habitatlab_test.py из проекта Habitat-Lab подробно разбирается логика написания YAML-конфигураций и порядок их загрузки. Это даст базовый подход к чтению более сложных YAML-файлов.

### 1. Базовая конфигурация YAML в habitatlab_test.py и её реализация

Порядок работы habitatlab_test.py:

1) Загрузка YAML-конфигурации.

2) Инициализация окружения.

3) Сброс окружения и получение наблюдения.

4) Управление агентом с клавиатуры.

5) Завершение цикла по заданному максимальному числу шагов.

Видно, что по сравнению с Habitat-Sim основное отличие — загрузка и настройка YAML. После импорта конфигурации инициализация всё равно выполняется через Habitat-Sim. Ниже подробно разобран файл pointnav_habitat_test.yaml.

```python
# @package _global_

defaults:
  - pointnav_base
  - /habitat/dataset/pointnav: habitat_test
  - _self_

habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agents:
      main_agent:
        sim_sensors:
          rgb_sensor:
            width: 256
            height: 256
          depth_sensor:
            width: 256
            height: 256

```

Входные данные YAML-конфигурации:
1. Загружается pointnav_base.yaml (базовая конфигурация PointNav).
2. /habitat/dataset/pointnav переопределяется на habitat_test.
3. Параметры habitat.environment.max_episode_steps: 500 и simulator.agents.main_agent.sim_sensors из текущего файла перекрывают базовую конфигурацию.

**Новичков часто смущает вопрос: неужели на этом вся конфигурация закончена? На самом деле нет. В коде много значений по умолчанию, которые в итоге разбираются Habitat-Lab и передаются в Habitat-Sim для создания окружения и агента. Ниже всё подробно разобрано, чтобы снять эти вопросы.**

Доступные параметры YAML можно посмотреть в документе: https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md

Все приведённые ниже фрагменты с конфигурацией взяты из функций файла default_structured_configs.py в Habitat-Lab; их можно посмотреть в исходном файле.

**Подробное описание YAML:**

**1) Настройка набора данных в YAML**

```python
class DatasetConfig(HabitatBaseConfig):
    type: str = "PointNav-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets"
    content_scenes: List[str] = field(default_factory=lambda: ["*"])
    data_path: str = (
        "data/datasets/pointnav/"
        "habitat-test-scenes/v1/{split}/{split}.json.gz"
    )
    # TODO: Make this field a structured dataclass.
    metadata: Optional[Any] = None
```

Значения по умолчанию ключевых параметров:

* habitat.dataset.type: по умолчанию PointNav-v1, то есть набор данных задачи навигации к точке.
* habitat.dataset.split: по умолчанию train, то есть обучающая выборка.
* habitat.dataset.scene_dir: по умолчанию data/scene_datasets, корневой каталог наборов сцен.
* habitat.dataset.data_path: по умолчанию data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz, путь к файлу набора; {split} заменяется фактическим значением split.
* Через поле content_scenes можно задать конкретный список загружаемых сцен; по умолчанию загружаются все доступные сцены.

**2) Настройка задачи в YAML**

```python
@dataclass
class TaskConfig(HabitatBaseConfig):
    physics_target_sps: float = 60.0
    reward_measure: Optional[str] = None
    success_measure: Optional[str] = None
    success_reward: float = 2.5
    slack_reward: float = -0.01
    end_on_success: bool = False
    # NAVIGATION task
    type: str = "Nav-v0"
    # Temporary structure for sensors
    lab_sensors: Dict[str, LabSensorConfig] = field(default_factory=dict)
    measurements: Dict[str, MeasurementConfig] = field(default_factory=dict)
    # Measures to only construct in the first environment of the first rank for
    # vectorized environments.
    rank0_env0_measure_names: List[str] = field(
        default_factory=lambda: ["habitat_perf"]
    )
    # Measures to only record in the first rank for vectorized environments.
    rank0_measure_names: List[str] = field(default_factory=list)
    goal_sensor_uuid: str = "pointgoal"
    # REARRANGE task
    count_obj_collisions: bool = True
    settle_steps: int = 5
    constraint_violation_ends_episode: bool = True
    constraint_violation_drops_object: bool = False
    # Forced to regenerate the starts even if they are already cached
    force_regenerate: bool = False
    # Saves the generated starts to a cache if they are not already generated
    should_save_to_cache: bool = False
    object_in_hand_sample_prob: float = 0.167
    min_start_distance: float = 3.0
    gfx_replay_dir = "data/replays"
    render_target: bool = True
    # Spawn parameters
    filter_colliding_states: bool = True
    num_spawn_attempts: int = 200
    spawn_max_dist_to_obj: float = 2.0
    base_angle_noise: float = 0.523599
    spawn_max_dist_to_obj_delta: float = 0.02
    # Factor to shrink the receptacle sampling volume when predicates place
    # objects on top of receptacles.
    recep_place_shrink_factor: float = 0.8
    # EE sample parameters
    ee_sample_factor: float = 0.2
    ee_exclude_region: float = 0.0
    base_noise: float = 0.05
    spawn_region_scale: float = 0.2
    joint_max_impulse: float = -1.0
    desired_resting_position: List[float] = field(
        default_factory=lambda: [0.5, 0.0, 1.0]
    )
    use_marker_t: bool = True
    cache_robot_init: bool = False
    success_state: float = 0.0
    # Measurements for composite tasks.
    should_enforce_target_within_reach: bool = False
    # COMPOSITE task CONFIG
    task_spec_base_path: str = "habitat/task/rearrange/pddl/"
    task_spec: str = ""
    # PDDL domain params
    pddl_domain_def: str = "replica_cad"
    obj_succ_thresh: float = 0.3
    # Disable drop except for when the object is at its goal.
    enable_safe_drop: bool = False
    art_succ_thresh: float = 0.15
    robot_at_thresh: float = 2.0

    # The minimum distance between the agents at start. If < 0
    # there is no minimal distance
    min_distance_start_agents: float = -1.0
    actions: Dict[str, ActionConfig] = MISSING
```

Значения по умолчанию ключевых параметров:

* habitat.task.type: по умолчанию Nav-v0, то есть задача навигации.
* habitat.task.physics_target_sps: по умолчанию 60.0, частота обновления физического движка — 60 раз в секунду.
* habitat.task.reward_measure: по умолчанию None; метрику награды нужно указать в конкретной конфигурации.
* habitat.task.success_measure: по умолчанию None; метрику успеха нужно указать в конкретной конфигурации.
* habitat.task.success_reward: по умолчанию 2.5, награда за успешное завершение задачи.
* habitat.task.slack_reward: по умолчанию -0.01, базовая награда (штраф) за каждый шаг.
* habitat.task.end_on_success: по умолчанию False, то есть после успеха эпизод не завершается автоматически.
* Конфигурация задачи также содержит многие значения по умолчанию, относящиеся к задачам перестановки, например settle_steps (по умолчанию 5 шагов), constraint_violation_ends_episode (по умолчанию True) и др.

**3) Настройка окружения в YAML**

```python
class EnvironmentConfig(HabitatBaseConfig):
    max_episode_steps: int = 1000
    max_episode_seconds: int = 10000000
    iterator_options: IteratorOptionsConfig = IteratorOptionsConfig()
```

Значения по умолчанию ключевых параметров:

* habitat.environment.max_episode_steps: по умолчанию 1000, ограничение максимального числа шагов эпизода.
* habitat.environment.max_episode_seconds: по умолчанию 10000000, ограничение максимальной длительности эпизода.
* habitat.environment.iterator_options: набор параметров итератора по умолчанию, например режим циклического прохода (cycle) по умолчанию True, перемешивание (shuffle) по умолчанию True и др.

**4) Настройка симулятора в YAML**

```python
class SimulatorConfig(HabitatBaseConfig):
    type: str = "Sim-v0"
    forward_step_size: float = 0.25  # in metres
    turn_angle: int = 10  # angle to rotate left or right in degrees
    create_renderer: bool = False
    requires_textures: bool = True
    # Sleep options
    auto_sleep: bool = False
    step_physics: bool = True
    concur_render: bool = False
    # If markers should be updated at every step:
    needs_markers: bool = True
    # If the articulated_agent camera positions should be updated at every step:
    update_articulated_agent: bool = True
    scene: str = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    # The scene dataset to load in the metadatamediator,
    # should contain simulator.scene:
    scene_dataset: str = "default"
    # A list of directory or config paths to search in addition to the dataset
    # for object configs. should match the generated episodes for the task:
    additional_object_paths: List[str] = field(default_factory=list)
    # Use config.seed (can't reference Config.seed) or define via code
    # otherwise it leads to circular references:
    #
    seed: int = II("habitat.seed")
    default_agent_id: int = 0
    debug_render: bool = False
    debug_render_articulated_agent: bool = False
    kinematic_mode: bool = False
    # If False, will skip setting the semantic IDs of objects in
    # `rearrange_sim.py` (there is overhead to this operation so skip if not
    # using semantic information).
    should_setup_semantic_ids: bool = True
    # If in render mode a visualization of the rearrangement goal position
    # should also be displayed
    debug_render_goal: bool = True
    robot_joint_start_noise: float = 0.0
    # Rearrange agent setup
    ctrl_freq: float = 120.0
    ac_freq_ratio: int = 4
    load_objs: bool = True
    # Rearrange agent grasping
    hold_thresh: float = 0.15
    grasp_impulse: float = 10000.0
    # we assume agent(s) to be set explicitly
    agents: Dict[str, AgentConfig] = MISSING
    # agents_order specifies the order in which the agents
    # are stored on the habitat-sim side.
    # In other words, the order to return the observations and accept
    # the actions when using the environment API.
    # If the number of agents is greater than one,
    # then agents_order has to be set explicitly.
    agents_order: List[str] = MISSING

    # Simulator should use default navmesh settings from agent config
    default_agent_navmesh: bool = True
    # if default navmesh is used, should it include static objects
    navmesh_include_static_objects: bool = False

    habitat_sim_v0: HabitatSimV0Config = HabitatSimV0Config()
    # ep_info is added to the config in some rearrange tasks inside
    # merge_sim_episode_with_object_config
    ep_info: Optional[Any] = None
    # The offset id values for the object
    object_ids_start: int = 100
    # Configuration for rendering
    renderer: RendererConfig = RendererConfig()
```

Значения по умолчанию ключевых параметров:

* habitat.simulator.type: по умолчанию Sim-v0.
* habitat.simulator.forward_step_size: по умолчанию 0.25 метра, длина шага при движении вперёд.
* habitat.simulator.turn_angle: по умолчанию 10 градусов, угол поворота.
* habitat.simulator.create_renderer: по умолчанию False.
* habitat.simulator.requires_textures: по умолчанию True.
* habitat.simulator.lag_observations: по умолчанию 0.
* habitat.simulator.auto_sleep: по умолчанию False.
* habitat.simulator.step_physics: по умолчанию True.

**5) Настройка агента в YAML**

```python
class AgentConfig(HabitatBaseConfig):
    height: float = 1.5
    radius: float = 0.1
    max_climb: float = 0.2
    max_slope: float = 45.0
    grasp_managers: int = 1
    sim_sensors: Dict[str, Any] = field(default_factory=dict)
    is_set_start_state: bool = False
    start_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    start_rotation: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    joint_start_noise: float = 0.1
    joint_that_can_control: Optional[List[int]] = None
    # Hard-code the robot joint start. `joint_start_noise` still applies.
    joint_start_override: Optional[List[float]] = None
    articulated_agent_urdf: Optional[str] = None
    articulated_agent_type: Optional[str] = None
    ik_arm_urdf: Optional[str] = None
    # File to motion data, used to play pre-recorded motions
    motion_data_path: str = ""
    auto_update_sensor_transform: bool = True
```

* habitat.simulator.agents.main_agent.height: по умолчанию 1.5 метра — базовая высота агента; она напрямую влияет на угол наблюдения датчиков (RGB, глубинной камеры) и по умолчанию соответствует высоте взгляда человека.
* habitat.simulator.agents.main_agent.radius: по умолчанию 0.1 метра — радиус столкновения агента, используется для обнаружения столкновений в симуляторе, чтобы агент не проходил сквозь объекты сцены.
* habitat.simulator.agents.main_agent.auto_update_sensor_transform: по умолчанию True — положение датчиков относительно агента обновляется каждый кадр; если выключить, обновлять его нужно вручную, что подходит для пользовательских схем размещения датчиков.

**Объединив значения по умолчанию с pointnav_habitat_test.yaml и pointnav_base.yaml, можно получить примерный полный YAML-файл (в реальном запуске такой файл не создаётся — он приведён только для справки):**

```python
# @package _global_
defaults:
  # Базовая конфигурация PointNav (содержит основную логику по умолчанию; её приоритет ниже последующих настроек)
  - pointnav_base
  # Ссылка на предопределённую конфигурацию набора habitat_test
  - /habitat/dataset/pointnav: habitat_test
  # Подключаем дискретные действия для навигации (вперёд, поворот, остановка и т. д.)
  - /habitat/task/actions:
    - move_forward
    - turn_left
    - turn_right
    - stop
    - look_up
    - look_down
  # Подключаем основные метрики навигации (число шагов, расстояние, успешность, SPL и др.)
  - /habitat/task/measurements:
    - num_steps
    - distance_to_goal
    - success
    - spl
    - distance_to_goal_reward
  # Подключаем сенсоры, которые нужны коду (GPS + компас + сенсор цели)
  - /habitat/task/lab_sensors:
    - pointgoal_with_gps_compass_sensor
    - compass_sensor
    - gps_sensor
  # Собственная конфигурация (наивысший приоритет, перекрывает значения по умолчанию выше)
  - _self_

# Основной блок конфигурации
habitat:
  # Настройки окружения (пользовательский лимит 500 шагов; остальные значения по умолчанию сохраняются)
  environment:
    max_episode_steps: 500
    max_episode_seconds: 10000000  # значение по умолчанию, защита от тайм-аута
    iterator_options:
      cycle: true  # циклический перебор эпизодов
      shuffle: true  # перемешивание эпизодов при обучении (включено по умолчанию)

  # Набор данных (параметры habitat_test, согласованные с предопределённой конфигурацией)
  dataset:
    type: PointNav-v1  # тип набора по умолчанию для задачи PointNav
    split: train  # выборка по умолчанию (можно динамически менять на val/test в коде)
    scenes_dir: data/scene_datasets  # каталог сцен по умолчанию
    data_path: data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz  # путь к данным по умолчанию
    content_scenes: []  # пустой список — загружаются все сцены; можно указать конкретные имена сцен

  # Конфигурация задачи (параметры навигации по умолчанию, согласованные с наградой и критерием успеха)
  task:
    type: Nav-v0  # тип задачи навигации по умолчанию
    physics_target_sps: 60.0  # частота обновления физики (по умолчанию 60 Гц)
    reward_measure: distance_to_goal_reward  # метрика награды (на основе изменения расстояния)
    success_measure: success  # метрика критерия успеха
    success_distance: 0.2  # порог успеха (расстояние до цели < 0.2 м считается успехом; совпадает с логикой кода)
    end_on_success: false  # после успеха эпизод не завершается автоматически (нужно вручную вызвать действие stop, нажав F)
    slack_reward: -0.01  # базовый штраф за шаг (предотвращает бессмысленное блуждание; значение по умолчанию)

  # Конфигурация симулятора (сохранены разрешения сенсоров пользователя, дополнены параметры физики и агента по умолчанию)
  simulator:
    type: Sim-v0  # тип симулятора по умолчанию
    forward_step_size: 0.25  # шаг движения вперёд (по умолчанию 0.25 м)
    turn_angle: 10.0  # угол поворота за один шаг (по умолчанию 10 градусов)
    auto_sleep: false  # автоматический сон отключён (сохраняем интерактивность)
    step_physics: true  # шаги физики включены (обязательно для навигации)
    requires_textures: true  # рендеринг текстур включён (RGB-изображение выводится корректно)
    agents:
      main_agent:
        height: 1.5  # высота агента (по умолчанию 1.5 м, соответствует взгляду человека)
        radius: 0.1  # радиус столкновения агента (по умолчанию 0.1 м)
        sim_sensors:
          # RGB-сенсор (пользовательские 256x256; дополнены параметры вроде угла обзора по умолчанию)
          rgb_sensor:
            width: 256
            height: 256
            hfov: 90.0  # горизонтальный угол обзора (по умолчанию 90 градусов)
            sensor_type: COLOR
            position: [0.0, 1.5, 0.0]  # положение сенсора (совпадает с высотой агента)
          # Глубинный сенсор (пользовательские 256x256; дополнены значения по умолчанию)
          depth_sensor:
            width: 256
            height: 256
            hfov: 90.0
            sensor_type: DEPTH
            position: [0.0, 1.5, 0.0]
            min_depth: 0.0  # минимальная глубина обнаружения
            max_depth: 10.0  # максимальная глубина обнаружения (по умолчанию 10 м)

  # Дополнительная настройка сенсоров (соответствует lab_sensors, чтобы наблюдения выводились корректно)
  task:
    lab_sensors:
      pointgoal_with_gps_compass_sensor:
        type: PointGoalWithGPSCompassSensor  # интегрированный сенсор цели + GPS + компас
      compass_sensor:
        type: EpisodicCompassSensor  # компас (выдаёт разницу угла относительно начальной ориентации)
      gps_sensor:
        type: EpisodicGPSSensor  # GPS (выдаёт разницу координат относительно начальной позиции)

  # Дополнительная настройка метрик (чтобы логика награды и успеха была корректной)
  task:
    measurements:
      distance_to_goal:
        distance_to: POINT  # расстояние до точки цели (по умолчанию)
      success:
        success_distance: 0.2  # совпадает с конфигурацией задачи, чтобы критерий был единым
      spl:
        success_distance: 0.2  # порог успеха, от которого зависит расчёт SPL
```

**Если вы уже успешно запустили habitatlab_test.py и поняли базовый принцип YAML-конфигурации, можно пропустить остальную часть. Тем, кто хочет разобраться, как код загружает YAML-файл, пригодятся следующие разделы.**

### 2. Порядок выполнения конфигурации YAML

**1) Получение конфигурационного файла и слияние с базовой конфигурацией**

Конфигурационный объект строится функцией get_config, которая находится в файле habitat-lab/habitat/config/default.py. Порядок построения такой:

```python
def get_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    configs_dir: str = _HABITAT_CFG_DIR,
) -> DictConfig:
    r"""Returns habitat config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat.seed=1"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_HABITAT_CFG_DIR`).
    :return: composed config object.
    """
    register_configs()
    config_path = get_full_config_path(config_path, configs_dir)
    # If get_config is called from different threads, Hydra might
    # get initialized twice leading to issues. This lock fixes it.
    with lock, initialize_config_dir(
        version_base=None,
        config_dir=osp.dirname(config_path),
    ):
        cfg = compose(
            config_name=osp.basename(config_path),
            overrides=overrides if overrides is not None else [],
        )

    return patch_config(cfg)
```

1. Разрешение пути: сначала функция get_full_config_path вычисляет абсолютный путь к конфигурационному файлу; она проверяет, существует ли указанный путь, а если нет — ищет файл в каталоге конфигураций.
2. Композиция конфигурации: compose объединяет конфигурацию; её аргументы — имя конфигурации (базовое имя файла) и список переопределений.
3. Патч конфигурации: перед возвратом объекта вызывается patch_config, которая дополняет отсутствующие ключи и проверяет, что некоторые ключи существуют и совместимы друг с другом.

**2) Инициализация Env конфигурацией**

После get_config в файле habitat-lab/habitat/core/env.py создаются симулятор, задача и датчики. Класс habitat.Env — основной класс окружения Habitat-Lab; он объединяет три компонента: набор данных, симулятор и задачу.

Подробные шаги инициализации класса Env:

1. Обработка аргументов конструктора:

```python
def __init__(self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None) -> None:
```
* Параметр config: конфигурация окружения; она должна содержать идентификатор симулятора и имя задачи, которые передаются функциям make_sim и make_task.

* Параметр dataset: ссылка на набор данных с информацией об отдельных экземплярах задач; может быть None.

2. Предобработка конфигурации:

```python
if "habitat" in config:
    config = config.habitat
self._config = config
```

Если в конфигурации есть ключ habitat, то в качестве рабочей конфигурации используется его подраздел.

3. Инициализация набора данных:

```python
self._dataset = dataset
if self._dataset is None and config.dataset.type:
    self._dataset = make_dataset(
        id_dataset=config.dataset.type,
        config=config.dataset
    )
```
Если набор данных не передан, а в конфигурации указан его тип, набор создаётся через make_dataset.

4. Инициализация текущего эпизода и итератора эпизодов:

```python
self._current_episode = None
self._episode_iterator = None
self._episode_from_iter_on_reset = True
self._episode_force_changed = False
```

5. Настройка итератора эпизодов (при наличии набора данных):

```python
if self._dataset:
    assert (
        len(self._dataset.episodes) > 0
    ), "dataset should have non-empty episodes list"
    self._setup_episode_iterator()
    self.current_episode = next(self.episode_iterator)
    with read_write(self._config):
        self._config.simulator.scene_dataset = (
            self.current_episode.scene_dataset_config
        )
        self._config.simulator.scene = self.current_episode.scene_id

    self.number_of_episodes = len(self.episodes)
else:
    self.number_of_episodes = None
```

* Вызывается метод _setup_episode_iterator для настройки итератора эпизодов.
* Получается первый эпизод, он становится текущим.
* В конфигурации обновляются набор сцен и идентификатор сцены.
* Задаётся общее число эпизодов.

6. Инициализация симулятора:

```python
self._sim = make_sim(
    id_sim=self._config.simulator.type,
    config=self._config.simulator
)
```

Симулятор создаётся по типу и конфигурации из файла настроек.

7. Инициализация задачи:

```python
self._task = make_task(
    self._config.task.type,
    config=self._config.task,
    sim=self._sim,
    dataset=self._dataset,
)
```

Задача создаётся по типу, конфигурации задачи, экземпляру симулятора и набору данных.

8. Настройка пространства наблюдений и пространства действий:

```python
self.observation_space = spaces.Dict({
    **self._sim.sensor_suite.observation_spaces.spaces,
    **self._task.sensor_suite.observation_spaces.spaces,
})
self.action_space = self._task.action_space
```

Пространство наблюдений объединяет пространства из набора датчиков симулятора и набора датчиков задачи; пространство действий берётся из задачи.

9. Настройка ограничений по шагам и времени:

```python
self._max_episode_seconds = self._config.environment.max_episode_seconds
self._max_episode_steps = self._config.environment.max_episode_steps
self._elapsed_steps = 0
self._episode_start_time: Optional[float] = None
self._episode_over = False
```

Из конфигурации читаются максимальное число шагов и время эпизода, а также инициализируются связанные служебные переменные.

В описанном процессе создание компонентов по конфигурации — ключевой механизм Habitat-Lab: через файл конфигурации можно гибко создавать окружения разных типов.

Ниже показаны код и пути для создания симулятора (make_sim), задачи (make_task) и датчиков:

* Создание симулятора

(habitat-lab/habitat/sims/registration.py)

```python
def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    _sim = registry.get_simulator(id_sim)
    assert _sim is not None, "Could not find simulator with name {}".format(id_sim)
    return _sim(**kwargs)
```

* Создание задачи

(habitat-lab/habitat/tasks/registration.py)

```python
def make_task(id_task, **kwargs):
    logger.info("Initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(id_task)
    return _task(**kwargs)
```

* Создание датчиков

(habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py)

Отдельной функции для создания датчиков нет: они задаются прямо в habitat_simulator.py при настройке окружения Habitat-Sim.

```python
sim_sensors = []
for agent_config in self.habitat_config.agents.values():
    for sensor_cfg in agent_config.sim_sensors.values():
        sensor_type = registry.get_sensor(sensor_cfg.type)
        assert sensor_type is not None, "invalid sensor type {}".format(sensor_cfg.type)
        sim_sensors.append(sensor_type(sensor_cfg))
```

Из конфигурации читается настройка датчиков каждого агента, затем с помощью registry.get_sensor получается класс датчика, и по конфигурации создаётся экземпляр датчика.

**3) Настройка окружения Habitat-Sim по конфигурации**

После обработки YAML полученная конфигурация используется для настройки окружения Habitat-Sim.

1. Структура конфигурации симулятора

```python
habitat.simulator
├── type: "Sim-v0"  # тип симулятора
├── scene_dataset: "data/scene_datasets/habitat-test-scenes"  # путь к набору сцен
├── scene: "van-gogh-room.glb"  # текущая сцена
├── agents:  # настройки агента
│   └── rgbd_agent:
│       ├── height: 1.5
│       ├── radius: 0.1
│       └── sim_sensors:  # настройки датчиков
│           ├── rgb_sensor:
│           └── depth_sensor:
├── forward_step_size: 0.25  # шаг движения вперёд
└── turn_angle: 10  # угол поворота
```

В методе create_sim_config класса HabitatSim (habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py) конфигурация превращается во внутренний формат Habitat-Sim.

```python
def create_sim_config(self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
    sim_config = habitat_sim.SimulatorConfiguration()

    # Копируем из конфигурации Habitat-Lab в конфигурацию Habitat-Sim
    overwrite_config(
        config_from=self.habitat_config.habitat_sim_v0,
        config_to=sim_config,
        ignore_keys={"gpu_gpu"},
    )

    sim_config.scene_dataset_config_file = self.habitat_config.scene_dataset
    sim_config.scene_id = self.habitat_config.scene

    # Настройка агента
    agent_config = habitat_sim.AgentConfiguration()
    overwrite_config(
        config_from=lab_agent_config,
        config_to=agent_config,
        # Эти ключи используются только в Habitat-Lab, их пропускаем
        ignore_keys={
            "is_set_start_state", "sensors", "sim_sensors", "start_position",
            "start_rotation", "articulated_agent_urdf", "articulated_agent_type",
            ...
        },
    )

    # Настройка датчиков
    sensor_specifications = []
    for sensor in _sensor_suite.sensors.values():
        sim_sensor_cfg = sensor._get_default_spec()
        overwrite_config(
            config_from=sensor.config,
            config_to=sim_sensor_cfg,
            ignore_keys=sensor._config_ignore_keys,
            trans_dict={
                "sensor_model_type": lambda v: getattr(habitat_sim.FisheyeSensorModelType, v),
                "sensor_subtype": lambda v: getattr(habitat_sim.SensorSubType, v),
            },
        )
        sensor_specifications.append(sim_sensor_cfg)

    agent_config.sensor_specifications = sensor_specifications

    return habitat_sim.Configuration(sim_config, [agent_config])
```

2. Структура конфигурации задачи (на примере задачи PointNav)

```python
habitat.task
├── type: "Nav-v0"  # тип задачи
├── reward_measure: "distance_to_goal_reward"  # метрика награды
├── success_measure: "spl"  # метрика успеха
├── success_reward: 2.5  # награда за успех
├── slack_reward: -0.01  # штраф за шаг
├── end_on_success: true  # завершать ли эпизод при успехе
└── lab_sensors:  # датчики Habitat-Lab
    └── pointgoal_with_gps_compass_sensor:
        ├── type: "PointGoalWithGPSCompassSensor"
        └── goal_format: "POLAR"
```

Во время инициализации класса NavigationTask (habitat-lab/habitat/tasks/nav/nav.py) конфигурация используется для настройки различных свойств задачи:

```python
@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):
                agent_config = get_agent_config(config.simulator)
                agent_config.start_position = episode.start_position
                agent_config.start_rotation = [
                    float(k) for k in episode.start_rotation
                ]
                agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
```

3. Структура конфигурации агента

```python
habitat.simulator.agents.rgbd_agent
├── height: 1.5  # высота агента
├── radius: 0.1  # радиус агента
├── sim_sensors:  # настройки датчиков
│   ├── rgb_sensor:
│   │   ├── type: "HabitatSimRGBSensor"
│   │   ├── width: 256
│   │   └── height: 256
│   └── depth_sensor:
│       ├── type: "HabitatSimDepthSensor"
│       ├── width: 256
│       └── height: 256
├── start_position: [0.0, 0.0, 0.0]  # начальная позиция
└── start_rotation: [0.0, 0.0, 0.0, 1.0]  # начальный поворот (кватернион)
```

В методе create_sim_config класса HabitatSim (habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py) конфигурация агента превращается во внутренний формат Habitat-Sim:

```python
def create_sim_config(self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
    # ... прочая обработка конфигурации ...

    # Настройка агента
    agent_config = habitat_sim.AgentConfiguration()

    # Копируем из конфигурации Habitat-Lab в конфигурацию Habitat-Sim
    overwrite_config(
        config_from=lab_agent_config,
        config_to=agent_config,
        ignore_keys={
            "is_set_start_state", "sensors", "sim_sensors", "start_position",
            "start_rotation", "articulated_agent_urdf", "articulated_agent_type",
            ...
        },
    )

    agent_config.sensor_specifications = sensor_specifications

    agent_config.action_space = {
        0: habitat_sim.ActionSpec("stop"),
        1: habitat_sim.ActionSpec(
            "move_forward",
            habitat_sim.ActuationSpec(
                amount=self.habitat_config.forward_step_size
            ),
        ),
        2: habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(
                amount=self.habitat_config.turn_angle
            ),
        ),
        3: habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(
                amount=self.habitat_config.turn_angle
            ),
        ),
    }

    # ... обработка конфигурации датчиков ...

    return habitat_sim.Configuration(sim_config, [agent_config])
```

4. Структура конфигурации датчика

```python
habitat.simulator.agents.rgbd_agent.sim_sensors.rgb_sensor
├── type: "HabitatSimRGBSensor"  # тип датчика
├── width: 256  # ширина изображения
├── height: 256  # высота изображения
├── position: [0.0, 1.25, 0.0]  # положение датчика
├── orientation: [0.0, 0.0, 0.0]  # ориентация датчика
└── hfov: 90  # горизонтальный угол обзора
```

В реализации класса датчика (habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py) конфигурация используется для настройки свойств датчика:

```python
@registry.register_sensor
class HabitatSimRGBSensor(RGBSensor, HabitatSimSensor):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)
        self._width = config.width
        self._height = config.height
        self._hfov = config.hfov
        self._position = config.position
        self._orientation = config.orientation
```

Во время инициализации датчиков в классе HabitatSim конфигурация датчика превращается в спецификацию датчика Habitat-Sim:

```python
for sensor in _sensor_suite.sensors.values():
    sim_sensor_cfg = sensor._get_default_spec()

    # Копируем из конфигурации датчика Habitat-Lab в конфигурацию датчика Habitat-Sim
    overwrite_config(
        config_from=sensor.config,
        config_to=sim_sensor_cfg,
        ignore_keys=sensor._config_ignore_keys,
        trans_dict={
            "sensor_model_type": lambda v: getattr(habitat_sim.FisheyeSensorModelType, v),
            "sensor_subtype": lambda v: getattr(habitat_sim.SensorSubType, v),
        },
    )

    sim_sensor_cfg.uuid = sensor.uuid
    sim_sensor_cfg.resolution = [sensor.observation_space.shape[1], sensor.observation_space.shape[0]]
    sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
    sensor_specifications.append(sim_sensor_cfg)
```

### Описание базовой конфигурации Habitat-Lab и её выполнения в коде завершено. Видно, что система конфигураций Habitat-Lab позволяет гибко создавать самые разные воплощённые окружения ИИ — от простой навигации до сложных задач манипуляции.

Справочные материалы:

https://github.com/facebookresearch/habitat-lab
