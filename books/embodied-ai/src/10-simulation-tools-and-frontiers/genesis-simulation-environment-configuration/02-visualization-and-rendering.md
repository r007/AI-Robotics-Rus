# 📸 Визуализация и рендеринг

Визуализацией созданной сцены Genesis управляет `visualizer` сцены (то есть `scene.visualizer`). Сцену можно визуализировать двумя способами: 1) с помощью интерактивного просмотрщика, работающего в отдельном потоке; 2) вручную добавляя камеру в сцену и получая изображения через камеру.

## Просмотрщик

Если подключён монитор, для визуализации сцены можно использовать интерактивный просмотрщик. Genesis использует разные группы `options` для настройки разных компонентов сцены. Чтобы настроить просмотрщик, меняют параметры в `viewer_options` при создании сцены. Кроме того, `vis_options` задают свойства визуализации, которые используют и просмотрщик, и камеры (камеры мы добавим чуть позже).

Пример сцены с более подробными настройками просмотрщика и визуализации (выглядит сложнее, чем нужно, но показывает возможности):

```python
scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # Показать систему координат `world` в её начале
        world_frame_size = 1.0, # Длина системы координат мира (в метрах)
        show_link_frame  = False, # Не показывать системы координат звеньев сущностей
        show_cameras     = False, # Не показывать сетку и усечённый конус добавленных камер
        plane_reflection = True, # Включить отражение от плоскости
        ambient_light    = (0.1, 0.1, 0.1), # Настройка окружающего освещения
    ),
    renderer = gs.renderers.Rasterizer(), # Использовать растеризатор для рендеринга камер
)
```

Здесь можно задать позу камеры просмотрщика и угол обзора. Если `max_FPS` установлен в `None`, просмотрщик работает так быстро, как может. Если `res` установлен в None, Genesis автоматически создаст окно с соотношением сторон 4:3 и высотой, равной половине высоты монитора. Обратите внимание: в примере выше для рендеринга камер выбран растеризатор. Genesis предоставляет два бэкенда рендеринга: `gs.renderers.Rasterizer()` и `gs.renderers.RayTracer()`. Просмотрщик всегда использует растеризатор; по умолчанию растеризатор используют и камеры.

После создания сцены объект просмотрщика доступен через `scene.visualizer.viewer` или короткую запись `scene.viewer`. Можно запросить или задать позу камеры просмотрщика:

```python
cam_pose = scene.viewer.camera_pose()
scene.viewer.set_camera_pose(cam_pose)
```

## Камеры и безголовый рендеринг

Теперь добавим в сцену объект камеры вручную. Камера не связана с просмотрщиком или монитором: она возвращает отрендеренное изображение только тогда, когда это нужно. Поэтому камера работает и в безголовом режиме (headless).

```python
cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False
)
```

Если `GUI=True`, каждая камера создаёт окно OpenCV, которое динамически показывает отрендеренное изображение. Обратите внимание: это не то же самое, что GUI просмотрщика.

После построения сцены камера может рендерить изображения. Наши камеры умеют выдавать RGB-изображения, карты глубины, маски сегментации и нормали поверхностей. По умолчанию рендерится только RGB; остальные режимы включаются параметрами при вызове `camera.render()`:

```python
scene.build()
# Рендеринг RGB, карты глубины, маски сегментации и карты нормалей
rgb, depth, segmentation, normal = cam.render(depth=True, segmentation=True, normal=True)
```

Если использован `GUI=True` и подключён монитор, теперь должно быть видно четыре окна. (Иногда окна OpenCV отстают; если окно чёрное, можно вызвать дополнительный `cv2.waitKey(1)` или просто снова вызвать `render()`, чтобы обновить окно.)

**Запись видео с камеры**

Теперь отрендерим только RGB-изображения, переместим камеру и запишем видео. Для записи Genesis предоставляет удобный инструмент:

```python
# Начать запись с камеры. Все отрендеренные RGB-изображения будут записываться внутри
cam.start_recording()
import numpy as np
for i in range(120):
    scene.step()
    # Меняем положение камеры
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )

    cam.render()
# Остановить запись и сохранить видео. Если `filename` не задан, имя будет создано автоматически по имени вызывающего файла
cam.stop_recording(save_to_filename='video.mp4', fps=60)
```

Полный код, который сохраняет видео в `video.mp4`:

```python
import genesis as gs
gs.init(backend=gs.cpu)
scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)
scene.build()
# Рендеринг RGB, карты глубины, маски сегментации и карты нормалей
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
cam.start_recording()
import numpy as np
for i in range(120):
    scene.step()
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='video.mp4', fps=60)
```

В исходном документе здесь был встроенный видеоплеер с записью этого кода; в этой версии его нет, поскольку запись получается при запуске кода.

## Параллельная симуляция

```python
import torch
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer   = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build(n_envs=30000)

# Управляем всеми роботами сразу
franka.control_dofs_position(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (30000, 1)
    ),
)

for i in range(1000):
    scene.step()
```

## Тестовый скрипт с бутылкой: test_genesis_bottle.py

```py
import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # Размер батча
parser.add_argument("-v", action="store_true", default=False) # Визуализация
parser.add_argument("-r", action="store_true", default=False) # Случайные действия

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

obj = scene.add_entity(
    morph=gs.morphs.URDF(
        file="../assets/urdf/bottle/bottle.urdf",
        scale=0.09,
        pos=(0.65, 0.0, 0.036),
        euler=(0, 90, 0),
    ),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# Коэффициенты управления (официальные значения из MuJoCo XML Franka)
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

grasp_qpos = np.array([-0.9937,  1.4588,  1.3058, -1.6924, -1.4882,  1.8461,  1.4577,  0.04, 0.04])
lift_qpos = np.array([-1.0411,  1.2861,  1.5200, -1.7065, -1.2946,  1.6572,  1.4315,  0.04, 0.04])
franka.set_dofs_position(grasp_qpos)

# Захват
franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-4, -4]), fingers_dof)


for i in range(100):
    scene.step()

# Подъём
franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for i in range(50):
    scene.step()

ref_pos = torch.tile(torch.tensor(lift_qpos[:7]), [n_envs, 1]).cuda()

t0 = time.perf_counter()
for i in range(500):
    if args.r and i % 2 == 0: # Совпадение с частотой управления ManiSkill 50 Гц
        # Совпадение со случайными действиями ManiSkill
        franka.control_dofs_position(ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025, motors_dof)
    scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')
```

### 03test_genesis_cube.py

```py
import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # Размер батча
parser.add_argument("-v", action="store_true", default=False) # Визуализация
parser.add_argument("-r", action="store_true", default=False) # Случайные действия
parser.add_argument("-m", action="store_true", default=False) # Движение по траектории
parser.add_argument("--mjcf", action="store_true", default=False) # Использовать MJCF

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    ),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml") if args.mjcf else gs.morphs.URDF(file="../assets/urdf/franka_description/robots/franka_panda.urdf", fixed=True),
)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# Коэффициенты управления (официальные значения из MuJoCo XML Franka)
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

grasp_qpos = np.array([-1.0104,  1.5623,  1.3601, -1.6840, -1.5863,  1.7810,  1.4598,  0.0400, 0.0400])
lift_qpos = np.array([-1.0426,  1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510,  0., 0.])
franka.set_dofs_position(grasp_qpos)

# Захват
franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)


for i in range(100):
    scene.step()

# Подъём
franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for i in range(50):
    scene.step()

ref_pos = torch.tile(torch.tensor(lift_qpos[:7]), [n_envs, 1]).cuda()

t0 = time.perf_counter()
if args.m:
    init_qpos = torch.tile(torch.tensor([-1.0426,  1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510]), [n_envs, 1]).cuda()
    target_qpos = torch.tile(torch.tensor([1.0426,  -1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510]), [n_envs, 1]).cuda()
    for i in range(500):
        if i % 2 == 0: # Совпадение с частотой управления ManiSkill 50 Гц
            franka.control_dofs_position(init_qpos + (target_qpos - init_qpos) * i / 500, motors_dof)
        scene.step()
else:
    for i in range(500):
        if args.r and i % 2 == 0: # Совпадение с частотой управления ManiSkill 50 Гц
            # Совпадение со случайными действиями ManiSkill
            franka.control_dofs_position(ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025, motors_dof)
        scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')
```

## Обратная кинематика

```py

```
