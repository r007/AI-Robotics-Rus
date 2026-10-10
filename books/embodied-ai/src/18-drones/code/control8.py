import numpy as np
import matplotlib.pyplot as plt

# ========== Упрощённое 1D MPC-управление высотой (скользящая оптимизация + ограничение тяги) ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение знака минуса в виде квадрата
# --- Системные параметры ---
m = 0.5; g = 9.81; dt_ctrl = 0.05  # Период управления 50 мс
N_horizon = 20                       # Горизонт прогноза
T_sim = 6.0; N_sim = int(T_sim / dt_ctrl)
z_target = 2.0                       # Целевая высота

# Ограничение тяги (физические пределы)
a_min = -g          # Минимальное ускорение (свободное падение, тяга = 0)
a_max = 2.0 * g     # Максимальное ускорение (верхний предел тяги)

# Веса функции стоимости MPC
Q_z, Q_v, R_a = 10.0, 5.0, 0.1  # Веса позиции, скорости и стоимости управления

def solve_mpc(z0, v0, z_ref, N_h):
    """
    Решение 1D MPC: вручную строим квадратичную задачу и решаем её с помощью numpy
    Состояние: [z, v], вход: a (ускорение), дискретная модель: z+=v*dt, v+=a*dt
    """
    # Построение матриц прогноза: X = Sx*x0 + Su*U
    Sx = np.zeros((2 * N_h, 2))  # Матрица распространения состояния
    Su = np.zeros((2 * N_h, N_h))  # Матрица влияния входа

    A = np.array([[1, dt_ctrl], [0, 1]])
    B = np.array([[0.5 * dt_ctrl**2], [dt_ctrl]])

    # Пошаговое построение матриц прогноза
    A_pow = np.eye(2)
    for k in range(N_h):
        A_pow = A_pow @ A
        Sx[2*k:2*k+2, :] = A_pow
        for j in range(k + 1):
            A_mid = np.linalg.matrix_power(A, k - j)
            Su[2*k:2*k+2, j] = (A_mid @ B).flatten()

    # Матрицы стоимости Q_bar, R_bar
    Q_bar = np.zeros((2 * N_h, 2 * N_h))
    for k in range(N_h):
        Q_bar[2*k, 2*k] = Q_z      # Вес позиции
        Q_bar[2*k+1, 2*k+1] = Q_v  # Вес скорости
    R_bar = R_a * np.eye(N_h)

    # Вектор опорной траектории
    X_ref = np.zeros(2 * N_h)
    for k in range(N_h):
        X_ref[2*k] = z_ref
        X_ref[2*k+1] = 0.0  # Желаемая скорость равна 0

    x0 = np.array([z0, v0])

    # Оптимальное решение без ограничений: U* = (Su'QSu + R)^{-1} Su'Q(Xref - Sx*x0)
    H = Su.T @ Q_bar @ Su + R_bar
    f = Su.T @ Q_bar @ (X_ref - Sx @ x0)
    U_star = np.linalg.solve(H, f)

    # Применяем ограничения (простое усечение)
    U_star = np.clip(U_star, a_min, a_max)
    return U_star

# --- Главный цикл симуляции ---
z, v = 0.0, 0.0  # Начальное состояние: покой на земле
hist_z, hist_v, hist_a = [], [], []
hist_pred_z = []  # Записываем предсказанную траекторию на каждом шаге (для визуализации)

for i in range(N_sim):
    U_opt = solve_mpc(z, v, z_target, N_horizon)
    a_cmd = U_opt[0]  # Выполняем только первый шаг (скользящая оптимизация)

    # Записываем предсказанную траекторию
    if i % 20 == 0:
        pred_z = [z]
        pz, pv = z, v
        for k in range(N_horizon):
            ak = np.clip(U_opt[k], a_min, a_max)
            pv += ak * dt_ctrl
            pz += pv * dt_ctrl
            pred_z.append(pz)
        hist_pred_z.append((i * dt_ctrl, pred_z))

    # Обновление состояния
    v += a_cmd * dt_ctrl
    z += v * dt_ctrl
    hist_z.append(z); hist_v.append(v); hist_a.append(a_cmd)

t_arr = np.arange(N_sim) * dt_ctrl

# --- Построение графиков ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Отклик высоты + предсказанная траектория
ax = axes[0, 0]
ax.plot(t_arr, hist_z, 'b-', lw=2, label='Фактическая высота')
ax.axhline(y=z_target, color='r', ls='--', lw=1.5, label=f'Цель z={z_target} м')
for (t0, pred) in hist_pred_z:
    t_pred = t0 + np.arange(len(pred)) * dt_ctrl
    ax.plot(t_pred, pred, 'g-', alpha=0.3, lw=0.8)
ax.plot([], [], 'g-', alpha=0.5, label='Предсказанная траектория (скользящее окно)')
ax.set_xlabel('Время (с)'); ax.set_ylabel('Высота (м)')
ax.set_title('Отклик высоты MPC'); ax.legend(); ax.grid(True, alpha=0.3)

# Скорость
ax = axes[0, 1]
ax.plot(t_arr, hist_v, 'c-', lw=1.5)
ax.set_xlabel('Время (с)'); ax.set_ylabel('Скорость (м/с)')
ax.set_title('Вертикальная скорость'); ax.grid(True, alpha=0.3)

# Входное управление (ускорение/тяга)
ax = axes[1, 0]
ax.plot(t_arr, hist_a, 'm-', lw=1.5, label='Выход MPC a')
ax.axhline(y=a_max, color='r', ls=':', label=f'Максимальное ускорение {a_max:.1f}')
ax.axhline(y=a_min, color='b', ls=':', label=f'Минимальное ускорение {a_min:.1f}')
ax.axhline(y=0, color='k', ls='-', alpha=0.2)
ax.set_xlabel('Время (с)'); ax.set_ylabel('Ускорение (м/с²)')
ax.set_title('Входное управление (с учётом усечения)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Соответствующая тяга
ax = axes[1, 1]
thrust = [m * (a + g) for a in hist_a]
ax.plot(t_arr, thrust, 'g-', lw=1.5, label='Тяга u₁ = m(a+g)')
ax.axhline(y=m * g, color='k', ls='--', alpha=0.4, label=f'Тяга зависания {m*g:.2f} Н')
ax.axhline(y=m * (a_max + g), color='r', ls=':', alpha=0.6, label='Верхний предел тяги')
ax.axhline(y=0, color='b', ls=':', alpha=0.6, label='Нижний предел тяги (0 Н)')
ax.set_xlabel('Время (с)'); ax.set_ylabel('Тяга (Н)')
ax.set_title('Визуализация ограничений тяги'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Глава 8: 1D MPC-управление высотой — скользящая оптимизация + ограничение тяги', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control8.png', dpi=150, bbox_inches='tight')
plt.show()