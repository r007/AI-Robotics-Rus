#!/usr/bin/env python3
"""
Регистрация робота GEN72-EG2 в окружении ManiSkill

Этот скрипт регистрирует пользовательского робота GEN72-EG2, настраивает его физические свойства и параметры управления,
чтобы его можно было использовать в окружениях ManiSkill, особенно вместе со стабильной версией алгоритма PPO.
"""
import os
import sys
import numpy as np
# torch будет импортирован при реальном обучении; на этапе регистрации он необязателен
try:
    import torch
except ImportError:
    print("Предупреждение: torch не установлен, он нужен только для обучения PPO")
import sapien
from copy import deepcopy

# Убеждаемся, что модули ManiSkill импортируются
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

# Измените этот путь на фактический путь к файлу URDF
URDF_PATH = '/home/<пользователь>/17robo/ManiSkill/urdf_01/GEN72-EG2.urdf'

# При первой регистрации окружения может появиться ошибка об отсутствии звеньев вроде "4C2_baselink": физический движок может некорректно загрузить некоторые звенья URDF
# В таком случае может понадобиться исправить URDF или проверить определения звеньев и суставов в URDF

@register_agent()
class GEN72EG2Robot(BaseAgent):
    """
    Класс робота GEN72-EG2, интегрирующий 7-степенной манипулятор и захват EG2
    """
    uid = "gen72_eg2_robot"
    urdf_path = URDF_PATH
    
    # Задаём трение, чтобы можно было захватывать объекты
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            # Убедитесь, что эти имена звеньев совпадают с фактическими именами в URDF
            "Link7": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "4C2_Link2": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "4C2_Link3": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        },
    )
    
    # Имена суставов — фактические имена из файла URDF
    arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    # Имена суставов захвата, из файла URDF
    gripper_joint_names = ['4C2_Joint1', '4C2_Joint2', '4C2_Joint3', '4C2_Joint5']
    
    # Имена звеньев концевого эффектора
    ee_link_name = "Link7"
    tcp_link_name = "Link7"  # Имя TCP-звена; обычно совпадает с концевым эффектором
    
    # Параметры управления — подобраны для более стабильного управления
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100
    
    # Ключевые кадры начальной позы — оптимизированы для задачи толкания кубика
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0, -0.1, 0, -1.5, 0, 1.8, 0.8,  # манипулятор — более естественная поза, рука слегка опущена, подходит для толкания кубика
                0.04, 0.04, 0.04, 0.04  # суставы захвата, состояние «открыт»
            ]),
            # Поворачиваем на 180° и сдвигаем основание вперёд, чтобы манипулятор легко доставал до объектов на столе
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),  # поворот на 180° вокруг оси z и сдвиг вперёд на 0,2 м
        ),
        push_ready=Keyframe(
            qpos=np.array([
                0, 0.2, 0, -1.2, 0, 1.6, 0.0,  # манипулятор — поза подготовки к толканию
                0.04, 0.04, 0.04, 0.04  # суставы захвата, состояние «открыт»
            ]),
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),
        ),
        grasp_ready=Keyframe(
            qpos=np.array([
                0, 0.1, 0, -1.0, 0, 1.2, 0.0,  # манипулятор — поза подготовки к захвату
                0.04, 0.04, 0.04, 0.04  # суставы захвата, состояние «открыт»
            ]),
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),
        ),
        grasp_close=Keyframe(
            qpos=np.array([
                0, 0.1, 0, -1.0, 0, 1.2, 0.0,  # манипулятор — поза подготовки к захвату
                0.0, 0.0, 0.0, 0.0  # суставы захвата, состояние «закрыт»
            ]),
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),
        )
    )
    
    def initialize(self, engine, scene):
        """Инициализирует робота и сохраняет TCP-звено"""
        super().initialize(engine, scene)
        # Находим и сохраняем TCP-звено
        self._tcp_link = None
        for link in self.robot.get_links():
            if link.name == self.tcp_link_name:
                self._tcp_link = link
                break
        if self._tcp_link is None:
            raise ValueError(f"TCP link {self.tcp_link_name} not found in robot links")
    
    @property
    def tcp(self):
        """Возвращает Actor TCP-звена (Tool Center Point)"""
        if not hasattr(self, '_tcp_link') or self._tcp_link is None:
            for link in self.robot.get_links():
                if link.name == self.tcp_link_name:
                    self._tcp_link = link
                    break
        return self._tcp_link
    
    @property
    def _controller_configs(self):
        """Настраивает контроллеры робота"""
        # -------------------------------------------------------------------------- #
        # Контроллеры манипулятора
        # -------------------------------------------------------------------------- #
        # Управление положением суставов
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        
        # Управление приращением положения суставов — часто используется в RL
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,  # ограничиваем величину действия на каждом шаге для стабильности
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        
        # Управление приращением положения концевого эффектора
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.05,  # уменьшаем пространство действий для повышения стабильности
            pos_upper=0.05,
            stiffness=self.arm_stiffness*2,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit*3,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        
        # Управление приращением ориентации концевого эффектора
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.05,  # меньшее пространство действий
            pos_upper=0.05,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        
        # -------------------------------------------------------------------------- #
        # Контроллер захвата
        # -------------------------------------------------------------------------- #
        # Управление положением захвата
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,  # положение закрытия
            upper=0.04,  # положение открытия
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )
        
        # Возвращаем все конфигурации контроллеров
        controller_configs = dict(
            # Комбинированный контроллер — одновременно управляет манипулятором и захватом
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
        )
        
        # Глубокое копирование, чтобы пользователь случайно не изменил конфигурацию
        return deepcopy(controller_configs)
    
    def is_grasping(self, obj: Actor, min_force=0.5, max_angle=85):
        """
        Проверяет, удерживает ли захват объект
        
        Args:
            obj: Actor объекта, захват которого нужно проверить
            min_force: минимальная сила контакта
            max_angle: максимальный угол контакта (градусы)
            
        Returns:
            Булев тензор: успешен ли захват
        """
        # Упрощённая реализация: решение только по состоянию захвата, физический контакт не проверяется
        # Проверяем, достаточно ли закрыт захват
        q = self.robot.get_qpos()
        gripper_idx = [self.robot.get_active_joints().index(j) for j in self.robot.get_active_joints() 
                      if j.name in self.gripper_joint_names[:1]]  # проверяем только первый сустав
        
        if len(gripper_idx) == 0:
            return torch.zeros(self.count, dtype=torch.bool, device=self.device)
        
        gripper_pos = q[:, gripper_idx[0]]
        # Если степень закрытия захвата меньше половины максимального раскрытия, считаем захват успешным
        return gripper_pos < 0.02
    
    def is_static(self, threshold: float = 0.1):
        """Проверяет, неподвижен ли робот"""
        qvel = self.robot.get_qvel()
        
        # Вычисляем сумму квадратов скоростей суставов
        arm_vel = torch.zeros(self.count, device=self.device)
        for joint_name in self.arm_joint_names:
            idx = self.robot.get_qlimits().joint_map[joint_name]
            arm_vel += qvel[:, idx] ** 2
            
        is_static = arm_vel < threshold ** 2
        return is_static
    
    def get_state_names(self):
        """Возвращает список имён для использования в состоянии"""
        return ["qpos", "qvel"]


def register_to_envs():
    """Регистрирует робота в соответствующих окружениях"""
    # Импортируем модуль окружения
    from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    
    environments = [PushCubeEnv, PickCubeEnv]
    robot_uid = GEN72EG2Robot.uid
    
    for env_class in environments:
        if robot_uid not in env_class.SUPPORTED_ROBOTS:
            env_class.SUPPORTED_ROBOTS.append(robot_uid)
            print(f"{robot_uid} добавлен в список поддерживаемых роботов класса {env_class.__name__}")
    
    print(f"\nТеперь можно использовать '{robot_uid}' в качестве аргумента robot_uids в окружении")
    print(f"Пример команды: python ppo_my.py --env_id='PushCube-v1' --robot_uids='{robot_uid}' ...")


if __name__ == "__main__":
    # Регистрируем робота
    robot = GEN72EG2Robot
    print(f"Робот GEN72-EG2 зарегистрирован, UID: {robot.uid}")
    print(f"Путь к файлу URDF: {URDF_PATH}")
    
    # Регистрируем робота в окружении
    register_to_envs() 