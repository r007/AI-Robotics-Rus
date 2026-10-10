import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ========== Влияние параметров PID на ступенчатый отклик системы второго порядка ==========
# Объект управления: упрощённая модель высоты дрона  m * z̈ = u - mg
# Эквивалентная система второго порядка: ë + (Kd/m)*ė + (Kp/m)*e = -(Ki/m)*∫e dt

def simulate_pid(Kp, Kd, Ki, dt=0.001, T=5.0):
    """Моделирование ступенчатого отклика системы второго порядка с PID-управлением"""
    steps = int(T / dt)
    z = 0.0       # Текущее положение (высота)
    dz = 0.0      # Текущая скорость
    z_des = 1.0   # Желаемая высота (ступенчатый вход)
    m = 0.5       # Масса дрона
    g = 9.81      # Ускорение свободного падения
    integral = 0.0  # Интегральная составляющая

    z_hist = np.zeros(steps)
    for i in range(steps):
        e = z_des - z           # Ошибка положения
        de = -dz                # Ошибка скорости (желаемая скорость равна 0)
        integral += e * dt      # Интеграл ошибки

        # Закон PID-управления: u = mg + Kp*e + Kd*ė + Ki*∫e
        u = m * g + Kp * e + Kd * de + Ki * integral

        # Динамика: m * z̈ = u - mg
        ddz = (u - m * g) / m
        dz += ddz * dt
        z += dz * dt
        z_hist[i] = z
    return z_hist

dt = 0.001; T = 5.0
t = np.arange(0, T, dt)

# --- Подграфик 1: меняем Kp (Kd=2, Ki=0 фиксированы) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for Kp in [1, 5, 15, 30]:
    z_hist = simulate_pid(Kp=Kp, Kd=2.0, Ki=0.0, dt=dt, T=T)
    ax.plot(t, z_hist, label=f'Kp={Kp}')
ax.axhline(y=1.0, color='k', ls='--', alpha=0.4, label='Цель')
ax.set_title('Меняем Kp (Kd=2, Ki=0)')
ax.set_xlabel('Время (с)'); ax.set_ylabel('Высота z (м)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.8)

# --- Подграфик 2: меняем Kd (Kp=15, Ki=0 фиксированы) ---
ax = axes[1]
for Kd in [0, 1, 3, 8]:
    z_hist = simulate_pid(Kp=15.0, Kd=Kd, Ki=0.0, dt=dt, T=T)
    ax.plot(t, z_hist, label=f'Kd={Kd}')
ax.axhline(y=1.0, color='k', ls='--', alpha=0.4, label='Цель')
ax.set_title('Меняем Kd (Kp=15, Ki=0)')
ax.set_xlabel('Время (с)'); ax.set_ylabel('Высота z (м)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.8)

# --- Подграфик 3: меняем Ki (Kp=10, Kd=3 фиксированы) ---
ax = axes[2]
for Ki in [0, 2, 8, 20]:
    z_hist = simulate_pid(Kp=10.0, Kd=3.0, Ki=Ki, dt=dt, T=T)
    ax.plot(t, z_hist, label=f'Ki={Ki}')
ax.axhline(y=1.0, color='k', ls='--', alpha=0.4, label='Цель')
ax.set_title('Меняем Ki (Kp=10, Kd=3)')
ax.set_xlabel('Время (с)'); ax.set_ylabel('Высота z (м)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.8)

plt.suptitle('Влияние параметров PID на ступенчатый отклик высоты дрона', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control4.png', dpi=150, bbox_inches='tight')
plt.show()