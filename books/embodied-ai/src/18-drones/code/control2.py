import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ========== Модель мотора квадрокоптера: расчёт и визуализация тяги и момента ==========

# --- Параметры мотора ---
kF = 6.11e-8   # Коэффициент тяги (Н/(рад/с)^2)
kM = 1.5e-9    # Коэффициент момента (Н·м/(рад/с)^2)
L  = 0.175     # Длина рычага (м)

# --- Скорости четырёх моторов (рад/с), моделируем асимметричный полёт ---
omega = np.array([4000, 4200, 3800, 4100])  # ω1, ω2, ω3, ω4

# --- Расчёт тяги и реактивного момента каждого мотора ---
F = kF * omega**2     # Тяга каждого мотора
M = kM * omega**2     # Реактивный момент каждого мотора

# --- Расчёт общей тяги и управляющих моментов ---
u1 = np.sum(F)                       # Общая тяга
u2 = L * (F[1] - F[3])              # Момент крена
u3 = L * (F[2] - F[0])              # Момент тангажа
u4 = M[0] - M[1] + M[2] - M[3]     # Момент рысканья

print(f"Тяга моторов: F = {F} Н")
print(f"Общая тяга u1 = {u1:.4f} Н")
print(f"Момент крена u2 = {u2:.6f} Н·м")
print(f"Момент тангажа u3 = {u3:.6f} Н·м")
print(f"Момент рысканья u4 = {u4:.8f} Н·м")

# --- Визуализация: вид сверху (положение моторов и тяга) и 3D-стрелки тяги ---
fig = plt.figure(figsize=(14, 5))

# Подграфик 1: вид сверху, положение четырёх моторов и величина тяги
ax1 = fig.add_subplot(1, 2, 1)
motor_pos = np.array([[0, L], [-L, 0], [0, -L], [L, 0]])  # Мотор 1 спереди, 2 слева, 3 сзади, 4 справа
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i in range(4):
    circle = plt.Circle(motor_pos[i], 0.04, color=colors[i], zorder=5)
    ax1.add_patch(circle)
    ax1.annotate(f'M{i+1}\nω={omega[i]}\nF={F[i]:.3f}N',
                 motor_pos[i], textcoords="offset points",
                 xytext=(15, 10), fontsize=8, color=colors[i])
# Рисуем корпус крестом
ax1.plot([motor_pos[1,0], motor_pos[3,0]],
         [motor_pos[1,1], motor_pos[3,1]], 'k-', lw=2)
ax1.plot([motor_pos[0,0], motor_pos[2,0]],
         [motor_pos[0,1], motor_pos[2,1]], 'k-', lw=2)
ax1.set_xlim(-0.35, 0.35); ax1.set_ylim(-0.35, 0.35)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
ax1.set_title('Вид сверху: расположение моторов и тяга'); ax1.set_xlabel('Y (m)'); ax1.set_ylabel('X (m)')

# Подграфик 2: 3D-вид со стрелками тяги
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i in range(4):
    x, y = motor_pos[i]
    # Стрелки тяги каждого мотора (вверх по оси Z)
    scale = F[i] / np.max(F) * 0.3  # Нормированное отображение
    ax2.quiver(x, y, 0, 0, 0, scale, color=colors[i],
               arrow_length_ratio=0.2, lw=2, label=f'M{i+1}: {F[i]:.3f}N')
    ax2.scatter(x, y, 0, color=colors[i], s=60)
# Стрелка общей тяги (от центра)
ax2.quiver(0, 0, 0, 0, 0, 0.4, color='k', arrow_length_ratio=0.15,
           lw=3, label=f'Общая тяга u1={u1:.3f} Н')
ax2.set_xlabel('Y'); ax2.set_ylabel('X'); ax2.set_zlabel('Z (направление тяги)')
ax2.set_title('3D-вид: вектор тяги'); ax2.legend(fontsize=7, loc='upper left')

plt.suptitle('Визуализация модели мотора квадрокоптера', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control2.png', dpi=150, bbox_inches='tight')
plt.show()