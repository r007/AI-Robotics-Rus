#!/usr/bin/env python3
"""Генерация схем демонстрационных проектов воплощённого ИИ"""

from graphviz import Digraph
import os

output_dir = os.path.expanduser("~/ros2_robot_simulation_ws/course/module1_1_robot_history/images/")
os.makedirs(output_dir, exist_ok=True)

# 1. Тенденции развития робототехники: четыре направления
def create_robot_trends():
    dot = Digraph(comment='Robot Technology Trends')
    dot.attr(rankdir='TB', dpi='300', size='12,10')
    
    # Центр
    dot.node('center', 'Робототехника\nТенденции развития', shape='doublecircle', 
             fillcolor='lightblue', style='filled', fontsize='24')
    
    # Четыре направления
    with dot.subgraph(name='cluster1') as s:
        s.attr(label='Интеллектуализация', style='filled', fillcolor='lightyellow')
        s.node('AI', 'ИИ-поддержка\n(большие модели/глубокое обучение)', shape='box', style='rounded,filled', fillcolor='gold')
        s.node('Auto', 'Автономное принятие решений', shape='ellipse', fillcolor='lightgreen')
    
    with dot.subgraph(name='cluster2') as s:
        s.attr(label='Кооперативность', style='filled', fillcolor='lightcyan')
        s.node('Cobot', 'Коллаборативный робот\n(Cobot)', shape='box', style='rounded,filled', fillcolor='lightblue')
        s.node('Human', 'Совместная работа человека и робота', shape='ellipse', fillcolor='lightgreen')
    
    with dot.subgraph(name='cluster3') as s:
        s.attr(label='Платформенность', style='filled', fillcolor='lightpink')
        s.node('ROS', 'ROS/ROS2\nоткрытый фреймворк', shape='box', style='rounded,filled', fillcolor='orange')
        s.node('Module', 'Модульный дизайн', shape='ellipse', fillcolor='lightgreen')
    
    with dot.subgraph(name='cluster4') as s:
        s.attr(label='Автономность', style='filled', fillcolor='lightgray')
        s.node('SLAM', 'SLAM\nАвтономная навигация', shape='box', style='rounded,filled', fillcolor='purple', fontcolor='white')
        s.node('L4', 'L4 автономное вождение', shape='ellipse', fillcolor='lightgreen')
    
    dot.edge('center', 'AI')
    dot.edge('center', 'Cobot')
    dot.edge('center', 'ROS')
    dot.edge('center', 'SLAM')
    
    dot.render(f'{output_dir}robot_trends', format='jpg', cleanup=True)
    print("robot_trends.jpg created")

# 2. Схема проекта PaLM-E
def create_palm_e_demo():
    dot = Digraph(comment='PaLM-E Demo')
    dot.attr(rankdir='LR', dpi='300')
    
    # Пользователь
    dot.node('User', 'Команда пользователя:\n"положить чип из ящика\nв коробку"', 
             shape='box', fillcolor='lightblue', style='rounded', fontsize='14')
    
    # PaLM-E
    dot.node('PaLM', 'PaLM-E\n(Embodied LM)', shape='box', fillcolor='gold', fontsize='18')
    dot.attr(label='Понимание большой языковой моделью и визуальное восприятие', fontcolor='gray')
    
    # Выход
    dot.node('Task', 'Декомпозиция задачи:\n1. открыть ящик\n2. найти чип\n3. захватить чип\n4. поместить в коробку', 
             shape='box', fillcolor='lightgreen', style='rounded')
    
    # Выполнение роботом
    dot.node('Robot', 'Выполнение роботом', shape='ellipse', fillcolor='orange', fontsize='14')
    
    dot.edge('User', 'PaLM')
    dot.edge('PaLM', 'Task')
    dot.edge('Task', 'Robot')
    
    dot.render(f'{output_dir}palm_e_demo', format='jpg', cleanup=True)
    print("palm_e_demo.jpg created")

# 3. Схема проекта RT-2
def create_rt2_demo():
    dot = Digraph(comment='RT-2 Demo')
    dot.attr(rankdir='LR', dpi='300')
    
    # Вход
    dot.node('Vision', 'Камера\nВизуальный вход', shape='ellipse', fillcolor='lightblue')
    dot.node('Cmd', 'Команда:\n"положить закуску\nв руки ближайшему человеку"', shape='box', fillcolor='lightyellow')
    
    # RT-2
    dot.node('RT2', 'RT-2\n(Vision-Language-Action)', shape='box', fillcolor='gold', fontsize='16')
    
    # Выход
    dot.node('Action', 'Вывод действий:\nдвижение руки → захват → размещение', shape='box', fillcolor='lightgreen')
    
    # Понимание и рассуждение
    dot.node('Reason', 'Семантическое рассуждение:\nпонимание понятия «человек»\nпонимание пространственных отношений', 
             shape='note', fillcolor='lightpink', fontsize='12')
    
    dot.edge('Vision', 'RT2')
    dot.edge('Cmd', 'RT2')
    dot.edge('RT2', 'Action')
    dot.edge('RT2', 'Reason', style='dashed')
    
    dot.render(f'{output_dir}rt2_demo', format='jpg', cleanup=True)
    print("rt2_demo.jpg created")

# 4. Схема NVIDIA GR00T
def create_gr00t_demo():
    dot = Digraph(comment='GR00T Demo')
    dot.attr(rankdir='TB', dpi='300')
    
    dot.node('Title', 'NVIDIA GR00T\nGeneralist Robot 00', shape='none', fontsize='20', fontcolor='green')
    
    with dot.subgraph(name='c1') as s:
        s.attr(label='Вход', style='filled', fillcolor='lightblue')
        s.node('Video', 'видеоданные из интернета', shape='box')
        s.node('Robot', 'данные о роботе', shape='box')
        s.node('Sim', 'данные симуляции', shape='box')
    
    with dot.subgraph(name='c2') as s:
        s.attr(label='GR00T Foundation Model', style='filled', fillcolor='gold')
        s.node('Model', 'универсальная модель робота', shape='ellipse', fillcolor='orange', fontsize='14')
    
    with dot.subgraph(name='c3') as s:
        s.attr(label='Выход', style='filled', fillcolor='lightgreen')
        s.node('Move', 'Управление движением', shape='box')
        s.node('Grasp', 'точный захват', shape='box')
        s.node('Nav', 'Автономная навигация', shape='box')
    
    dot.edge('Video', 'Model')
    dot.edge('Robot', 'Model')
    dot.edge('Sim', 'Model')
    dot.edge('Model', 'Move')
    dot.edge('Model', 'Grasp')
    dot.edge('Model', 'Nav')
    
    dot.render(f'{output_dir}gr00t_demo', format='jpg', cleanup=True)
    print("gr00t_demo.jpg created")

# 5. Схема Stanford ALOHA
def create_aloha_demo():
    dot = Digraph(comment='ALOHA Demo')
    dot.attr(rankdir='LR', dpi='300')
    
    # Сбор данных
    dot.node('Data', 'Этап сбора данных', shape='none', fontsize='16', fontcolor='blue')
    dot.node('Human', 'Человек надевает\nустройство захвата движения', shape='box', fillcolor='lightblue')
    dot.node('Teleop', 'телеоперация\n(Teleoperation)', shape='ellipse', fillcolor='yellow')
    dot.node('Dataset', 'Набор данных демонстраций\n(траектории/тактильные данные)', shape='box', fillcolor='lightgreen')
    
    # Обучение
    dot.node('Train', 'Этап обучения', shape='none', fontsize='16', fontcolor='blue')
    dot.node('BC', 'Клонирование поведения\n(Behavior Clone)', shape='ellipse', fillcolor='gold')
    dot.node('Policy', 'Политика робота', shape='box', fillcolor='orange')
    
    # Выполнение
    dot.node('Execute', 'Этап выполнения', shape='none', fontsize='16', fontcolor='blue')
    dot.node('Robot', 'Робот\nавтономное выполнение', shape='box', fillcolor='lightpink', style='rounded')
    
    dot.edge('Human', 'Teleop', label='демонстрация')
    dot.edge('Teleop', 'Dataset', label='запись')
    dot.edge('Dataset', 'BC', label='обучение')
    dot.edge('BC', 'Policy', label='обучение')
    dot.edge('Policy', 'Robot', label='выполнение')
    
    dot.render(f'{output_dir}aloha_demo', format='jpg', cleanup=True)
    print("aloha_demo.jpg created")

# 6. Панорама воплощённого ИИ
def create_embodied_overview():
    dot = Digraph(comment='Embodied AI Overview')
    dot.attr(rankdir='TB', dpi='300', size='14,12')
    
    dot.node('Title', 'Панорама технологий воплощённого ИИ', shape='none', fontsize='22', fontcolor='blue')
    
    # Базовые технологии
    with dot.subgraph(name='tech') as s:
        s.attr(label='Ключевые технологии', style='filled', fillcolor='lightyellow')
        s.node('LLM', 'LLM\nбольшая языковая модель', shape='box', fillcolor='gold')
        s.node('VLM', 'VLM\nвизуально-языковая модель', shape='box', fillcolor='gold')
        s.node('VLA', 'VLA\nвизуально-языковые действия', shape='box', fillcolor='gold')
        s.node('RL', 'RL\nобучение с подкреплением', shape='box', fillcolor='lightgreen')
        s.node('IL', 'IL\nимитационное обучение', shape='box', fillcolor='lightgreen')
    
    # Представительные проекты
    with dot.subgraph(name='projects') as s:
        s.attr(label='Представительные проекты', style='filled', fillcolor='lightcyan')
        s.node('PaLM_E', 'PaLM-E\n(Google)', shape='box', fillcolor='lightblue')
        s.node('RT2', 'RT-2\n(DeepMind)', shape='box', fillcolor='lightblue')
        s.node('GR00T', 'GR00T\n(NVIDIA)', shape='box', fillcolor='lightblue')
        s.node('ALOHA', 'ALOHA\n(Stanford)', shape='box', fillcolor='lightblue')
        s.node('Figure', 'Figure 01\n(Figure AI)', shape='box', fillcolor='lightblue')
    
    # Применение
    with dot.subgraph(name='apps') as s:
        s.attr(label='Сценарии применения', style='filled', fillcolor='lightpink')
        s.node('Home', 'Бытовые сервисы', shape='ellipse')
        s.node('Factory', 'Промышленное производство', shape='ellipse')
        s.node('Medical', 'Хирургия', shape='ellipse')
        s.node('Auto', 'Автономное вождение', shape='ellipse')
    
    dot.edge('LLM', 'PaLM_E')
    dot.edge('VLM', 'RT2')
    dot.edge('VLA', 'RT2')
    dot.edge('VLA', 'GR00T')
    dot.edge('IL', 'ALOHA')
    dot.edge('RL', 'GR00T')
    dot.edge('PaLM_E', 'Figure')
    dot.edge('RT2', 'Home')
    dot.edge('GR00T', 'Factory')
    dot.edge('ALOHA', 'Home')
    
    dot.render(f'{output_dir}embodied_overview', format='jpg', cleanup=True)
    print("embodied_overview.jpg created")

if __name__ == '__main__':
    print("Generating project demo diagrams...")
    create_robot_trends()
    create_palm_e_demo()
    create_rt2_demo()
    create_gr00t_demo()
    create_aloha_demo()
    create_embodied_overview()
    print("All project demos generated!")
