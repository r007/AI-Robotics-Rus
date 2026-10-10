import os
import imageio  # Обработка изображений и видео (чтение, запись, склейка кадров)
import numpy as np  # Численные вычисления (массивы, операции с матрицами)
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt  # Библиотека построения графиков

import habitat_sim  # Основная библиотека Habitat-Sim (ядро симуляции)
from habitat_sim.utils import common as utils  # Общие служебные функции (преобразование координат, обработка форматов данных)
from habitat_sim.utils import viz_utils as vut  # Функции визуализации (отрисовка сцены и траектории

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



def display_map(topdown_map, key_points=None):
    global img_counter
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    
    image_name = f"navmesh_{img_counter}.png"
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    img_counter +=1
    plt.show(block=False)
    plt.pause(3)  # Показываем 3 секунды (число секунд можно изменить)
    plt.close()



def get_topdown_map(pathfinder, height, meters_per_pixel) -> np.ndarray:
    # Получаем границы навигации сцены (оси x и z, высоту по оси y игнорируем)
    bounds = pathfinder.get_bounds()
    min_x, _, min_z = bounds[0]
    max_x, _, max_z = bounds[1]

    # Вычисляем размер карты в пикселях (x — ширина, z — высота)
    map_width = int(np.ceil((max_x - min_x) / meters_per_pixel))
    map_height = int(np.ceil((max_z - min_z) / meters_per_pixel))

    # Инициализируем карту: 0 = не проходимо, 1 = проходимо
    topdown_map = np.zeros((map_height, map_width), dtype=np.uint8)

    # Проходим каждый пиксель и определяем проходимость (оптимизация: векторные операции вместо двойного цикла, работает быстрее)
    x_coords = np.linspace(min_x, max_x, map_width, endpoint=False)
    z_coords = np.linspace(min_z, max_z, map_height, endpoint=False)
    x_grid, z_grid = np.meshgrid(x_coords, z_coords)
    # Строим массив координат (x, height, z)
    world_coords = np.stack([x_grid.ravel(), np.full_like(x_grid.ravel(), height), z_grid.ravel()], axis=1)
    # Пакетно определяем проходимость
    navigable = np.array([pathfinder.is_navigable(coord) for coord in world_coords])
    # Приводим к размеру карты
    topdown_map = navigable.reshape((map_height, map_width)).astype(np.uint8)

    # Вычисляем границы (необязательно, соответствует двум значениям старой версии)
    edges = ndimage.laplace(topdown_map) != 0
    topdown_map[edges] = 2

    return topdown_map

display = True #question???
test_scene = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
# test_scene = "./data_test/scene_datasets/habitat-test-scenes/apartment_1.glb"
rgb_sensor = True 
depth_sensor = True
semantic_sensor = True 
img_counter = 0

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}
# sim_settings["scene"] = "./data_test/scene_datasets/habitat-test-scenes/apartment_1.glb"

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
meters_per_pixel = 0.12
custom_height = False
height = 1


print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
if not custom_height:
    # get bounding box minumum elevation for automatic height
    height = sim.pathfinder.get_bounds()[0][1]

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

    if display:
        hablab_topdown_map = get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        hablab_topdown_map = recolor_map[hablab_topdown_map]
        display_map(sim_topdown_map)
        display_map(hablab_topdown_map)