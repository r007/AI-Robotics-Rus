import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ============================================================
# Визуализация ограничения Corridor и сравнение распределения времени
# Показываем путевые точки, безопасный коридор, препятствия и разницу скоростей для двух стратегий распределения времени
# ============================================================

np.random.seed(42)

# --- Путевые точки и препятствия ---
waypoints = np.array([[0, 0], [3, 2], [6, 4], [9, 3], [12, 5]])
obstacles = [
    {"center": [1.5, 3.8], "w": 1.2, "h": 1.0},  # Препятствие 1: над коридором
    {"center": [4.5, 1.2], "w": 1.0, "h": 1.2},  # Препятствие 2: под коридором
    {"center": [7.5, 5.8], "w": 1.4, "h": 0.8},  # Препятствие 3: над коридором
    {"center": [10.5, 1.5], "w": 1.0, "h": 1.0}, # Препятствие 4: под коридором
]
corridor_half = 1.2  # Половина ширины коридора

# --- Генерируем гладкую кривую через коридор (имитация сплайновой интерполяции) ---
from numpy.polynomial import polynomial as P_mod
t_wp = np.linspace(0, 1, len(waypoints))
t_dense = np.linspace(0, 1, 300)
# Простая параметризованная интерполяция
degree = min(4, len(waypoints) - 1)
cx = np.polyfit(t_wp, waypoints[:, 0], degree)
cy = np.polyfit(t_wp, waypoints[:, 1], degree)
traj_x = np.polyval(cx, t_dense)
traj_y = np.polyval(cy, t_dense)

# ==================== Рисунок 1: визуализация Corridor ====================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax1 = axes[0]

# Рисуем прямоугольник коридора
for wp in waypoints:
    rect = Rectangle(
        (wp[0] - corridor_half, wp[1] - corridor_half),
        2 * corridor_half, 2 * corridor_half,
        linewidth=1.5, edgecolor='#2ca02c', facecolor='#2ca02c',
        alpha=0.15, linestyle='--', label='Безопасный коридор' if wp is waypoints[0] else ''
    )
    ax1.add_patch(rect)

# Рисуем препятствия
for i, obs in enumerate(obstacles):
    rect = Rectangle(
        (obs["center"][0] - obs["w"]/2, obs["center"][1] - obs["h"]/2),
        obs["w"], obs["h"],
        linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.35,
        label='Препятствие' if i == 0 else ''
    )
    ax1.add_patch(rect)

# Рисуем траекторию и путевые точки
ax1.plot(traj_x, traj_y, 'b-', linewidth=2, label='Запланированная траектория')
ax1.plot(waypoints[:, 0], waypoints[:, 1], 'ko', markersize=8, zorder=5, label='Путевые точки')
for i, wp in enumerate(waypoints):
    ax1.annotate(f'P{i}', wp, textcoords="offset points", xytext=(6, 8), fontsize=10)

ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
ax1.set_title('Схема ограничения Corridor — траектория обходит препятствия внутри коридора')
ax1.legend(loc='upper left', fontsize=9); ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal'); ax1.set_xlim(-2, 14); ax1.set_ylim(-2, 9)

# ==================== Рисунок 2: сравнение скоростей при двух распределениях времени ====================
ax2 = axes[1]
seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
T_total = 10.0  # Полное время полёта

# Способ 1: равномерное распределение (время каждого отрезка одинаково)
T_uniform = np.full(len(seg_lengths), T_total / len(seg_lengths))
# Способ 2: пропорционально расстоянию
T_prop = T_total * seg_lengths / seg_lengths.sum()

# Вычисляем среднюю скорость каждого отрезка
v_uniform = seg_lengths / T_uniform
v_prop = seg_lengths / T_prop

# Рисуем ступенчатую диаграмму скоростей
seg_labels = [f'Отрезок {i+1}\n({seg_lengths[i]:.1f} м)' for i in range(len(seg_lengths))]
x_pos = np.arange(len(seg_lengths))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, v_uniform, width, color='#ff7f0e', alpha=0.8, label='Равномерное распределение времени')
bars2 = ax2.bar(x_pos + width/2, v_prop, width, color='#1f77b4', alpha=0.8, label='Распределение пропорционально расстоянию')

# Подписываем значения
for bar, v in zip(bars1, v_uniform):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{v:.2f}', ha='center', fontsize=8)
for bar, v in zip(bars2, v_prop):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{v:.2f}', ha='center', fontsize=8)

ax2.set_xticks(x_pos); ax2.set_xticklabels(seg_labels, fontsize=9)
ax2.set_ylabel('Средняя скорость (м/с)'); ax2.set_title('Сравнение скоростей двух стратегий распределения времени')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

# Добавляем информацию о времени на график
info_uniform = '  '.join([f'{t:.1f}s' for t in T_uniform])
info_prop = '  '.join([f'{t:.1f}s' for t in T_prop])
ax2.text(0.02, 0.95, f'Время при равномерном распределении: [{info_uniform}]', transform=ax2.transAxes,
         fontsize=8, verticalalignment='top', color='#ff7f0e')
ax2.text(0.02, 0.88, f'Время при пропорциональном распределении: [{info_prop}]', transform=ax2.transAxes,
         fontsize=8, verticalalignment='top', color='#1f77b4')

plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/planning3.png', dpi=150, bbox_inches='tight')
plt.show()