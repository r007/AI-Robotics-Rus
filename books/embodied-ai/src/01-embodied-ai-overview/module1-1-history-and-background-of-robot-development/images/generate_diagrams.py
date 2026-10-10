#!/usr/bin/env python3
"""Генерация схем архитектуры и блок-схем для курса"""

from graphviz import Digraph
import os

# Выходной каталог
output_dir = os.path.expanduser("~/ros2_robot_simulation_ws/course/module1_1_robot_history/images/")
os.makedirs(output_dir, exist_ok=True)

# 1. Архитектура системы ROS
def create_ros_architecture():
    dot = Digraph(comment='ROS Architecture')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Узлы
    dot.node('User', 'Пользовательское приложение\n(User Node)', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.node('Master', 'ROS Master', shape='ellipse', style='filled', fillcolor='gold')
    dot.node('Talker', 'Издатель\n(Talker)', shape='box', style='rounded,filled', fillcolor='lightgreen')
    dot.node('Listener', 'Подписчик\n(Listener)', shape='box', style='rounded,filled', fillcolor='lightgreen')
    dot.node('Service', 'Узел сервиса\n(Service)', shape='box', style='rounded,filled', fillcolor='lightyellow')
    dot.node('Action', 'Узел действия\n(Action)', shape='box', style='rounded,filled', fillcolor='lightsalmon')
    
    # Рёбра
    dot.edge('User', 'Master', style='dashed')
    dot.edge('Talker', 'Master', label='регистрация')
    dot.edge('Listener', 'Master', label='регистрация')
    dot.edge('Talker', 'Listener', label='обмен темами', style='bold', color='blue')
    dot.edge('User', 'Service', label='вызов')
    dot.edge('User', 'Action', label='цель')
    
    dot.render(f'{output_dir}ros_architecture', format='jpg', cleanup=True)
    print("ros_architecture.jpg created")

# 2. Схема концепции воплощённого ИИ
def create_embodied_ai_concept():
    dot = Digraph(comment='Embodied AI Concept')
    dot.attr(rankdir='LR')
    
    dot.node('AI', 'Традиционный ИИ\n(большая языковая модель)', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.node('Body', 'Физическое тело\n(робот/устройство)', shape='box', style='rounded,filled', fillcolor='lightgreen')
    dot.node('Sensor', 'Датчики\n(зрение/осязание)', shape='ellipse', fillcolor='yellow')
    dot.node('Actuator', 'Исполнительные механизмы\n(моторы/манипулятор)', shape='ellipse', fillcolor='orange')
    dot.node('Env', 'Физический мир\nEnvironment', shape='ellipse', fillcolor='pink')
    dot.node('Learn', 'Обучение взаимодействием', shape='diamond', fillcolor='lightyellow')
    
    dot.edge('AI', 'Body', label='управление')
    dot.edge('Body', 'Sensor', label='восприятие')
    dot.edge('Sensor', 'AI', label='обратная связь')
    dot.edge('AI', 'Actuator', label='команда')
    dot.edge('Actuator', 'Env', label='воздействие')
    dot.edge('Env', 'Body', label='влияние')
    dot.edge('Env', 'Learn', label='взаимодействие')
    dot.edge('Learn', 'AI', label='обучение')
    
    dot.render(f'{output_dir}embodied_ai_concept', format='jpg', cleanup=True)
    print("embodied_ai_concept.jpg created")

# 3. Архитектура робота с LLM
def create_llm_robot():
    dot = Digraph(comment='LLM Robot Architecture')
    dot.attr(rankdir='TB')
    
    dot.node('User', 'Команда пользователя\n"положить чип в коробку"', shape='box', fillcolor='lightblue')
    dot.node('LLM', 'большая языковая модель\n(PaLM-E)', shape='box', fillcolor='gold')
    dot.node('Plan', 'Планирование задачи\n(Task Planning)', shape='box', fillcolor='lightgreen')
    dot.node('Vision', 'Визуальное восприятие\n(Vision)', shape='ellipse', fillcolor='yellow')
    dot.node('Control', 'Управление движением\n(Motion Control)', shape='ellipse', fillcolor='orange')
    dot.node('Robot', 'Робот', shape='box', fillcolor='pink', style='rounded')
    
    dot.edge('User', 'LLM')
    dot.edge('LLM', 'Plan')
    dot.edge('Vision', 'LLM', style='dashed')
    dot.edge('Plan', 'Control')
    dot.edge('Control', 'Robot')
    dot.edge('Vision', 'Robot', style='dashed')
    
    dot.render(f'{output_dir}llm_robot', format='jpg', cleanup=True)
    print("llm_robot.jpg created")

# 4. Архитектура робота с VLM
def create_vlm_robot():
    dot = Digraph(comment='VLM Robot Architecture')
    dot.attr(rankdir='TB')
    
    dot.node('Image', 'Изображение с камеры', shape='ellipse', fillcolor='lightblue')
    dot.node('Text', 'Текстовая команда', shape='ellipse', fillcolor='lightblue')
    dot.node('VLM', 'Визуально-языковая модель\n(RT-2/OK-Robot)', shape='box', fillcolor='gold')
    dot.node('Grasp', 'Планирование захвата', shape='box', fillcolor='lightgreen')
    dot.node('Robot', 'Манипулятор', shape='box', fillcolor='pink', style='rounded')
    
    dot.edge('Image', 'VLM')
    dot.edge('Text', 'VLM')
    dot.edge('VLM', 'Grasp')
    dot.edge('Grasp', 'Robot')
    
    dot.render(f'{output_dir}vlm_robot', format='jpg', cleanup=True)
    print("vlm_robot.jpg created")

# 5. Архитектура модели VLA
def create_vla_architecture():
    dot = Digraph(comment='VLA Architecture')
    dot.attr(rankdir='TB')
    
    dot.node('Input', 'Визуальный вход\n+ языковая команда', shape='box', fillcolor='lightblue')
    dot.node('Encoder', 'Визуальный/языковой\nэнкодер', shape='box', fillcolor='lightgreen')
    dot.node('Transformer', 'Transformer\nбэкбон', shape='ellipse', fillcolor='gold')
    dot.node('ActionHead', 'Вывод действий\nголова', shape='box', fillcolor='orange')
    dot.node('Output', 'Действие робота\n(положение/поворот/захват)', shape='box', fillcolor='pink')
    
    dot.edge('Input', 'Encoder')
    dot.edge('Encoder', 'Transformer')
    dot.edge('Transformer', 'ActionHead')
    dot.edge('ActionHead', 'Output')
    
    dot.render(f'{output_dir}vla_architecture', format='jpg', cleanup=True)
    print("vla_architecture.jpg created")

# 6. Блок-схема обучения с подкреплением
def create_rl_flow():
    dot = Digraph(comment='Reinforcement Learning')
    dot.attr(rankdir='LR')
    
    dot.node('Start', 'Агент\n(Agent)', shape='box', fillcolor='lightblue')
    dot.node('Env', 'Среда\n(Environment)', shape='ellipse', fillcolor='lightgreen')
    dot.node('State', 'Состояние s', shape='point')
    dot.node('Action', 'Действие a', shape='point')
    dot.node('Reward', 'Награда r', shape='diamond', fillcolor='yellow')
    dot.node('Policy', 'Обновление политики\n(Policy Update)', shape='box', fillcolor='orange')
    
    dot.edge('Start', 'State', label='наблюдение')
    dot.edge('State', 'Action', label='выбор действия')
    dot.edge('Action', 'Env', label='выполнение')
    dot.edge('Env', 'Reward', label='получение обратной связи')
    dot.edge('Reward', 'Policy', label='обучение')
    dot.edge('Policy', 'Start', label='обновление политики')
    dot.edge('Env', 'State', label='новое состояние')
    
    dot.render(f'{output_dir}rl_flow', format='jpg', cleanup=True)
    print("rl_flow.jpg created")

# 7. Схема имитационного обучения
def create_imitation_learning():
    dot = Digraph(comment='Imitation Learning')
    dot.attr(rankdir='TB')
    
    dot.node('Human', 'Демонстратор-человек', shape='box', fillcolor='lightblue')
    dot.node('Demo', 'Данные демонстрации\n(траектории/действия)', shape='box', fillcolor='lightgreen')
    dot.node('BC', 'Клонирование поведения\n(Behavior Clone)', shape='ellipse', fillcolor='gold')
    dot.node('Policy', 'Сеть политики\n(Policy)', shape='box', fillcolor='orange')
    dot.node('Robot', 'Робот', shape='box', fillcolor='pink', style='rounded')
    dot.node('Loss', 'Функция потерь', shape='diamond', fillcolor='yellow')
    
    dot.edge('Human', 'Demo', label='демонстрация')
    dot.edge('Demo', 'BC')
    dot.edge('BC', 'Policy')
    dot.edge('Policy', 'Robot', label='выполнение')
    dot.edge('Robot', 'Loss', label='оценка')
    dot.edge('Loss', 'BC', label='обратное распространение')
    
    dot.render(f'{output_dir}imitation_learning', format='jpg', cleanup=True)
    print("imitation_learning.jpg created")

# 8. Схема пространственного 3D-восприятия
def create_spatial_intelligence():
    dot = Digraph(comment='Spatial Intelligence')
    dot.attr(rankdir='TB')
    
    dot.node('Camera', 'Камера глубины', shape='ellipse', fillcolor='lightblue')
    dot.node('Lidar', 'Лидар', shape='ellipse', fillcolor='lightblue')
    dot.node('PointCloud', 'Облако точек\n(Point Cloud)', shape='box', fillcolor='lightgreen')
    dot.node('PointNet', 'PointNet\n3D-глубокое обучение', shape='ellipse', fillcolor='gold')
    dot.node('3D_understanding', '3D-детекция объектов\n/сегментация/оценка позы', shape='box', fillcolor='orange')
    dot.node('Robot', 'Навигация робота\n/захват', shape='box', fillcolor='pink', style='rounded')
    
    dot.edge('Camera', 'PointCloud')
    dot.edge('Lidar', 'PointCloud')
    dot.edge('PointCloud', 'PointNet')
    dot.edge('PointNet', '3D_understanding')
    dot.edge('3D_understanding', 'Robot')
    
    dot.render(f'{output_dir}spatial_intelligence', format='jpg', cleanup=True)
    print("spatial_intelligence.jpg created")

if __name__ == '__main__':
    print("Generating diagrams...")
    create_ros_architecture()
    create_embodied_ai_concept()
    create_llm_robot()
    create_vlm_robot()
    create_vla_architecture()
    create_rl_flow()
    create_imitation_learning()
    create_spatial_intelligence()
    print("All diagrams generated!")
