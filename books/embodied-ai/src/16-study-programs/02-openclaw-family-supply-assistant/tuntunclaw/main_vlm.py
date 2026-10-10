import os
import sys
import cv2
import mujoco
import matplotlib.pyplot as plt 
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from vlm_process import segment_image
from grasp_process import run_grasp_inference, execute_grasp


# Глобальные переменные
global color_img, depth_img, env
color_img = None
depth_img = None
env = None


# Получаем данные цветного и глубинного изображения
def get_image(env):
    global color_img, depth_img
     # Получаем данные изображения из рендеринга среды
    imgs = env.render()

    # Извлекаем данные цветного и глубинного изображения
    color_img = imgs['img']   # Это данные изображения в формате RGB
    depth_img = imgs['depth'] # Это данные глубины

    # Преобразуем RGB-изображение в формат BGR, принятый в OpenCV
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    return color_img, depth_img

# Конструируем callback-функцию, которая вызывается непрерывно
def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()


def test_grasp():
    global color_img, depth_img, env

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # Часть обработки изображения
    masks = segment_image(color_img)  

    gg = run_grasp_inference(color_img, depth_img, masks)

    execute_grasp(env, gg)



if __name__ == '__main__':
    
    env = UR5GraspEnv()
    env.reset()
    
    while True:

        for i in range(500): # 1000
            env.step()

        color_img, depth_img = get_image(env)

        callback(color_img, depth_img)


    env.close()


    