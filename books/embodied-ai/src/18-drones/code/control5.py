import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ========== Моделирование 2D-дрона (плоскость Y-Z): прямая связь + PD-управление ==========

# --- Физические параметры ---
m = 0.5          # Масса (кг)
g = 9.81         # Ускорение свободного падения (м/с^2)
Ixx = 0.002      # Момент инерции вокруг оси x (кг·м^2)
dt = 0.002       # Шаг моделирования (с)
T_sim = 6.0      # Полное время моделирования
N = int(T_sim / dt)

# --- Коэффициенты PD ---
kp_z, kd_z = 15.0, 8.0       # PD по положению Z
kp_y, kd_y = 8.0, 6.0        # PD по положению Y
kp_phi, kd_phi = 150.0, 20.0 # PD по углу ориентации

# --- Желаемая траектория: зависание 2 с, затем движение к целевой позиции ---
y_des = np.where(np.arange(N) * dt < 2.0, 0.0, 2.0)
z_des = np.where(np.arange(N) * dt < 2.0, 1.0, 2.5)

# --- Инициализация состояния [y, z, phi, dy, dz, dphi] ---
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
hist_y, hist_z, hist_u1, hist_u2, hist_phi = [], [], [], [], []

for i in range(N):
    y, z, phi, dy, dz, dphi = state

    # Внешний контур: PD по Z + прямая компенсация силы тяжести
    ddz_c = kp_z * (z_des[i] - z) + kd_z * (0 - dz)
    u1 = m * (g + ddz_c) / (np.cos(phi) + 1e-6)  # Прямая связь: mg/cos(φ)
    u1 = np.clip(u1, 0, 3 * m * g)  # Ограничение тяги

    # Внешний контур: PD по Y → желаемый угол крена
    ddy_c = kp_y * (y_des[i] - y) + kd_y * (0 - dy)
    phi_c = -ddy_c / g  # Из линеаризации ÿ = -g·φ
    phi_c = np.clip(phi_c, -0.5, 0.5)  # Ограничение желаемого угла

    # Внутренний контур: PD по ориентации
    u2 = Ixx * (kp_phi * (phi_c - phi) + kd_phi * (0 - dphi))

    # Обновление нелинейной динамики
    ddy = -(u1 / m) * np.sin(phi)
    ddz = (u1 / m) * np.cos(phi) - g
    ddphi = u2 / Ixx

    state += np.array([dy, dz, dphi, ddy, ddz, ddphi]) * dt

    hist_y.append(y); hist_z.append(z)
    hist_u1.append(u1); hist_u2.append(u2); hist_phi.append(phi)

t_arr = np.arange(N) * dt

# --- Построение графиков: 2 строки, 2 столбца ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Траектория Y-Z
axes[0, 0].plot(hist_y, hist_z, 'b-', lw=1.5, label='Реальная траектория')
axes[0, 0].plot(y_des[0], z_des[0], 'go', ms=10, label='Начальная цель')
axes[0, 0].plot(y_des[-1], z_des[-1], 'r*', ms=14, label='Конечная цель')
axes[0, 0].set_xlabel('Y (m)'); axes[0, 0].set_ylabel('Z (m)')
axes[0, 0].set_title('Траектория полёта в плоскости Y-Z'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# Отклик по высоте и горизонтальному положению
axes[0, 1].plot(t_arr, hist_z, 'b-', label='z фактическое')
axes[0, 1].plot(t_arr, z_des, 'b--', alpha=0.5, label='z желаемое')
axes[0, 1].plot(t_arr, hist_y, 'r-', label='y фактическое')
axes[0, 1].plot(t_arr, y_des, 'r--', alpha=0.5, label='y желаемое')
axes[0, 1].set_xlabel('Время (с)'); axes[0, 1].set_ylabel('Положение (м)')
axes[0, 1].set_title('Отклик отслеживания положения'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# Управляющий вход
axes[1, 0].plot(t_arr, hist_u1, 'g-', label='u₁ (полная тяга)')
axes[1, 0].axhline(y=m * g, color='k', ls='--', alpha=0.4, label=f'Тяга зависания mg={m*g:.2f}')
axes[1, 0].set_xlabel('Время (с)'); axes[1, 0].set_ylabel('Тяга (Н)')
axes[1, 0].set_title('Полная тяга u₁'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# Угол крена
axes[1, 1].plot(t_arr, np.degrees(hist_phi), 'm-', label='φ (угол крена)')
axes[1, 1].set_xlabel('Время (с)'); axes[1, 1].set_ylabel('Угол (°)')
axes[1, 1].set_title('Отклик угла крена φ'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Глава 5: моделирование 2D-дрона с прямой связью + PD (зависание → движение)', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control5.png', dpi=150, bbox_inches='tight')
plt.show()