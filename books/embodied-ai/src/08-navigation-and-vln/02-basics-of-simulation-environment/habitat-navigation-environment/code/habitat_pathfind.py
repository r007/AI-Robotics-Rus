import math
import os
import magnum as mn  # 3D-графика и линейная алгебра (зависимость Habitat-Sim)
import numpy as np  # Численные вычисления (массивы, операции с матрицами)
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt  # Библиотека построения графиков

from PIL import Image, ImageDraw  # PIL/Pillow: обработка изображений (чтение, запись, обрезка и др.)

import habitat_sim  # Основная библиотека Habitat-Sim (ядро симуляции)
from habitat_sim.utils import common as utils  # Общие служебные функции (например, преобразование координат и обработка форматов данных)
from habitat_sim.utils import viz_utils as vut  # Функции визуализации (например, отрисовка сцены и траектории

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
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
    
    image_name = f"pathfind2_{img_counter}.png"
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    img_counter +=1
    plt.show(block=False)
    plt.pause(3)  # Показываем 3 секунды (число секунд можно изменить)
    plt.close()

def get_topdown_map(pathfinder, height, meters_per_pixel) -> np.ndarray:
    """
    Реализует основную функцию старой версии maps.get_topdown_map:
    Возвращает двумерный массив: 0 (не проходимо), 1 (проходимо), 2 (граница)
    """
    # Получаем границы навигации сцены (оси x и z, высоту по оси y игнорируем)
    bounds = pathfinder.get_bounds()
    min_x, _, min_z = bounds[0]
    max_x, _, max_z = bounds[1]

    # Вычисляем размер карты в пикселях (x соответствует ширине, z — высоте)
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


def to_grid(z, x, grid_dimensions, pathfinder):
    # Получаем границы навигационной сетки (min_x, _, min_z) и (max_x, _, max_z)
    min_bounds = pathfinder.get_bounds()[0]
    max_bounds = pathfinder.get_bounds()[1]
    min_x, _, min_z = min_bounds
    max_x, _, max_z = max_bounds

    # Нормируем x к [0, grid_width-1], z к [0, grid_height-1]
    grid_height, grid_width = grid_dimensions
    px = (x - min_x) / (max_x - min_x) * (grid_width - 1)
    py = (z - min_z) / (max_z - min_z) * (grid_height - 1)

    # Ограничиваем координаты пределами карты, чтобы не выйти за границы
    px = np.clip(px, 0, grid_width - 1)
    py = np.clip(py, 0, grid_height - 1)

    # Переводим в целые пиксельные координаты (как в старой версии)
    return int(round(py)), int(round(px))

def draw_path(top_down_map, trajectory, color=(255, 0, 0), thickness=2):
    # Переводим в PIL Image для рисования отрезков (можно использовать и OpenCV)
    img = Image.fromarray(top_down_map)
    draw = ImageDraw.Draw(img)

    # Проходим точки траектории и рисуем отрезки
    for i in range(len(trajectory) - 1):
        # У PIL координаты (x, y) соответствуют (px, py) сетки
        start = (trajectory[i][1], trajectory[i][0])
        end = (trajectory[i+1][1], trajectory[i+1][0])
        draw.line([start, end], fill=color, width=thickness)

    # Возвращаем в numpy-массив
    top_down_map[:] = np.array(img)

def draw_agent(top_down_map, agent_pos, angle, agent_radius_px=8, agent_color=(0, 255, 0), arrow_color=(0, 0, 255)):
    # Переводим в PIL Image для рисования
    img = Image.fromarray(top_down_map)
    draw = ImageDraw.Draw(img)
    py, px = agent_pos  # Координаты сетки (строка, столбец)
    x, y = px, py  # Координаты PIL (x = столбец, y = строка)

    # 1. Рисуем круглый корпус агента
    # Для эллипса в PIL нужны координаты левого верхнего и правого нижнего углов
    bbox = (x - agent_radius_px, y - agent_radius_px, x + agent_radius_px, y + agent_radius_px)
    draw.ellipse(bbox, fill=agent_color, outline=(0, 0, 0), width=1)

    # 2. Рисуем стрелку направления (конец стрелки вычисляем по углу)
    arrow_length = agent_radius_px * 1.5
    # Преобразование угла: math.atan2 отсчитывает угол от оси x против часовой стрелки; здесь подгоняем под ориентацию карты
    end_x = x + arrow_length * math.cos(angle)
    end_y = y + arrow_length * math.sin(angle)
    draw.line([(x, y), (end_x, end_y)], fill=arrow_color, width=2)

    # Возвращаем в numpy-массив
    top_down_map[:] = np.array(img)

display = True
test_scene = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
# test_scene = "./data/scene_datasets/habitat-test-scenes/apartment_1.glb"
mp3d_scene_dataset = "./data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"
rgb_sensor = True 
depth_sensor = True  
semantic_sensor = True 
img_counter = 0

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "scene_dataset": mp3d_scene_dataset,
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# the navmesh can also be explicitly loaded
# sim.pathfinder.load_nav_mesh(
#     "./data_test/scene_datasets/habitat-test-scenes/apartment_1.navmesh"
# )
# sim.pathfinder.load_nav_mesh(
#     "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.navmesh"
# )

meters_per_pixel = 0.12
custom_height = False 
height = 1  

agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
agent.set_state(agent_state)


img_counter = 0
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    global img_counter

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    
    image_name = f"pathfind2_{img_counter}.png"
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    img_counter +=1
    
    plt.show(block=False)
    plt.pause(3)  # Показываем 3 секунды (число секунд можно изменить)
    plt.close()

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    seed = 4
    sim.pathfinder.seed(seed)

    sample1 = sim.pathfinder.get_random_navigable_point()
    print("sample1:",sample1)
    sample2 = sim.pathfinder.get_random_navigable_point()
    print("sample2:",sample2)

    path = habitat_sim.ShortestPath()
    path.requested_start = sample1
    path.requested_end = sample2
    found_path = sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    path_points = path.points

    print("found_path : " + str(found_path))
    print("geodesic_distance : " + str(geodesic_distance))
    print("path_points : " + str(path_points))

    if found_path:
        meters_per_pixel = 0.025
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        height = scene_bb.y().min
        if display:
            top_down_map = get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            top_down_map = recolor_map[top_down_map]
            grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
            # convert world trajectory points to maps module grid points
            trajectory = [
                to_grid(
                    path_point[2],
                    path_point[0],
                    grid_dimensions,
                    pathfinder=sim.pathfinder,
                )
                for path_point in path_points
            ]
            grid_tangent = mn.Vector2(
                trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
            )
            path_initial_tangent = grid_tangent / grid_tangent.length()
            initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
            # draw the agent and trajectory on the map
            draw_path(top_down_map, trajectory)
            draw_agent(
                top_down_map, trajectory[0], initial_angle, agent_radius_px=8
            )
            print("\nDisplay the map with agent and path overlay:")
            display_map(top_down_map)

        display_path_agent_renders = True
        if display_path_agent_renders:
            print("Rendering observations at path points:")
            tangent = path_points[1] - path_points[0]
            agent_state = habitat_sim.AgentState()
            for ix, point in enumerate(path_points):
                if ix < len(path_points) - 1:
                    tangent = path_points[ix + 1] - point
                    agent_state.position = point
                    tangent_orientation_matrix = mn.Matrix4.look_at(
                        point, point + tangent, np.array([0, 1.0, 0])
                    )
                    tangent_orientation_q = mn.Quaternion.from_matrix(
                        tangent_orientation_matrix.rotation()
                    )
                    agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    agent.set_state(agent_state)

                    observations = sim.get_sensor_observations()
                    rgb = observations["color_sensor"]
                    semantic = observations["semantic_sensor"]
                    depth = observations["depth_sensor"]

                    if display:
                        display_sample(rgb, semantic, depth)