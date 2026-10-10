import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата

# ========== Визуализация вращения «креста» дрона вокруг осей Z/Y/X ==========

# --- Построение формы дрона «+» (4 конца рычагов + центр) ---
L = 1.0  # Длина рычага
drone = np.array([
    [ L, 0, 0],  # Правый конец рычага
    [-L, 0, 0],  # Левый конец рычага
    [ 0, L, 0],  # Передний конец рычага
    [ 0,-L, 0],  # Задний конец рычага
    [ 0, 0, 0],  # Центр
]).T  # Матрица 3×5, каждый столбец — одна точка

# --- Матрицы трёхмерного вращения ---
def Rz(psi):
    """Вращение вокруг оси Z (угол рысканья ψ)"""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Ry(theta):
    """Вращение вокруг оси Y (угол тангажа θ)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rx(phi):
    """Вращение вокруг оси X (угол крена φ)"""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

angle = np.radians(45)  # Поворот на 45°

# --- Рисуем 4 подграфика: исходное положение + 3 вида вращения ---
titles = ['Исходная поза', f'Вращение вокруг Z на {int(np.degrees(angle))}° (рысканье)',
          f'Вращение вокруг Y на {int(np.degrees(angle))}° (тангаж)',
          f'Вращение вокруг X на {int(np.degrees(angle))}° (крен)']
rotations = [np.eye(3), Rz(angle), Ry(angle), Rx(angle)]

fig = plt.figure(figsize=(14, 10))
for i, (R, title) in enumerate(zip(rotations, titles)):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    pts = R @ drone  # Координаты точек после вращения

    # Рисуем два рычага (крест)
    ax.plot([pts[0,0], pts[0,1]], [pts[1,0], pts[1,1]],
            [pts[2,0], pts[2,1]], 'b-o', lw=3, label='Рычаг 1')
    ax.plot([pts[0,2], pts[0,3]], [pts[1,2], pts[1,3]],
            [pts[2,2], pts[2,3]], 'r-o', lw=3, label='Рычаг 2')
    # Подписываем центр
    ax.scatter(*pts[:, 4], color='k', s=80, zorder=5)

    # Рисуем оси повёрнутого корпуса
    axis_len = 0.6
    for j, (color, lbl) in enumerate(zip(['r','g','b'], ['Xb','Yb','Zb'])):
        axis_vec = R @ (axis_len * np.eye(3)[:, j])
        ax.quiver(0, 0, 0, *axis_vec, color=color, arrow_length_ratio=0.15)

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)

plt.suptitle('Трёхмерная демонстрация вращения «креста» дрона', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control1.png', dpi=150, bbox_inches='tight')
plt.show()