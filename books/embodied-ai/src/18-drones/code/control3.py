import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ========== Проверка дифференциальной плоскостности: углы ориентации и тяга по круговой траектории ==========

# --- Физические параметры ---
m = 0.5    # Масса дрона (кг)
g = 9.81   # Ускорение свободного падения (м/с^2)

# --- Определяем круговую траекторию (плоский выход σ = [x, y, z, ψ]) ---
T_total = 10.0              # Полное время (с)
dt = 0.01
t = np.arange(0, T_total, dt)
omega_c = 2 * np.pi / T_total  # Угловая частота круговой траектории

# Траектория положения: горизонтальный круг + медленный подъём + рысканье по направлению скорости
radius = 2.0
x = radius * np.cos(omega_c * t)
y = radius * np.sin(omega_c * t)
z = 0.5 * t                                # Винтовой подъём
psi = np.arctan2(np.gradient(y, dt), np.gradient(x, dt))  # Угол рысканья по направлению скорости

# --- Вычисляем производные ---
ddx = np.gradient(np.gradient(x, dt), dt)  # Вторая производная x (ускорение)
ddy = np.gradient(np.gradient(y, dt), dt)  # Вторая производная y
ddz = np.gradient(np.gradient(z, dt), dt)  # Вторая производная z

# --- Вывод общей тяги u1 из дифференциальной плоскостности ---
u1 = m * np.sqrt(ddx**2 + ddy**2 + (ddz + g)**2)

# --- Вывод углов крена φ (roll) и тангажа θ (pitch) из ускорения ---
# Направление оси z корпуса (единичный вектор тяги)
tx = ddx / (ddz + g + 1e-10)
ty = ddy / (ddz + g + 1e-10)

# Углы Эйлера через связь с матрицей вращения
phi   = np.arcsin(np.clip(tx * np.sin(psi) - ty * np.cos(psi), -1, 1))  # Угол крена
theta = np.arctan2(tx * np.cos(psi) + ty * np.sin(psi),
                   1 + ddz / g)  # Угол тангажа

# --- Построение результатов ---
fig = plt.figure(figsize=(15, 10))

# Подграфик 1: трёхмерная траектория
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(x, y, z, 'b-', lw=1.5)
# Через равные промежутки отмечаем положение и ориентацию дрона
step = len(t) // 8
for i in range(0, len(t), step):
    ax1.scatter(x[i], y[i], z[i], c='r', s=40, zorder=5)
    dx_arrow = 0.3 * np.cos(psi[i])
    dy_arrow = 0.3 * np.sin(psi[i])
    ax1.quiver(x[i], y[i], z[i], dx_arrow, dy_arrow, 0,
               color='r', arrow_length_ratio=0.3)
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.set_title('Винтовая траектория σ(t)')

# Подграфик 2: общая тяга u1
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t, u1, 'g-', lw=1.5)
ax2.axhline(y=m*g, color='k', ls='--', alpha=0.5, label=f'Тяга зависания mg={m*g:.2f} Н')
ax2.set_xlabel('Время (с)'); ax2.set_ylabel('u₁ (N)')
ax2.set_title('Общая тяга u₁ (из второй производной)'); ax2.legend(); ax2.grid(True, alpha=0.3)

# Подграфик 3: угол крена φ
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t, np.degrees(phi), 'r-', lw=1.5, label='φ (roll)')
ax3.set_xlabel('Время (с)'); ax3.set_ylabel('Угол (°)')
ax3.set_title('Угол крена φ (из σ̈)'); ax3.legend(); ax3.grid(True, alpha=0.3)

# Подграфик 4: угол тангажа θ
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t, np.degrees(theta), 'b-', lw=1.5, label='θ (pitch)')
ax4.set_xlabel('Время (с)'); ax4.set_ylabel('Угол (°)')
ax4.set_title('Угол тангажа θ (из σ̈)'); ax4.legend(); ax4.grid(True, alpha=0.3)

plt.suptitle('Проверка дифференциальной плоскостности: вывод ориентации и тяги из траектории [x,y,z,ψ]', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control3.png', dpi=150, bbox_inches='tight')
plt.show()
