import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========== Сравнение: управление по углам Эйлера и SE(3) при больших поворотах ==========

def vee(M):
    """Антисимметричная матрица → вектор"""
    return np.array([M[2,1], M[0,2], M[1,0]])

def Ry(a):
    """Матрица поворота вокруг оси Y"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

# --- Параметры симуляции (единичная инерция для упрощения, акцент на различиях управления) ---
dt = 0.005; T_sim = 6.0; N = int(T_sim / dt)
t = np.arange(N) * dt
I_mat = np.diag([1.0, 1.0, 1.0])  # Единичная инерция

# Цель: разворот на 170° вокруг оси Y (близко к области шарнирного замка при pitch ±90°)
flip_angle = np.deg2rad(170)
R_des = Ry(flip_angle)

# --- Метод 1: PD-управление по углам Эйлера ---
euler = np.zeros(3)   # [phi, theta, psi]
deuler = np.zeros(3)
kp_e, kd_e = 4.0, 3.0

# Извлекаем целевые углы Эйлера из R_des (порядок ZYX)
theta_des = np.arctan2(-R_des[2,0], np.sqrt(R_des[0,0]**2 + R_des[1,0]**2))
phi_des = np.arctan2(R_des[2,1], R_des[2,2])
psi_des = np.arctan2(R_des[1,0], R_des[0,0])
euler_des = np.array([phi_des, theta_des, psi_des])

err_euler = np.zeros(N)
for i in range(N):
    e_ang = euler_des - euler
    err_euler[i] = np.linalg.norm(e_ang)
    tau = kp_e * e_ang + kd_e * (-deuler)
    deuler += tau * dt
    euler += deuler * dt

# --- Метод 2: управление SE(3) ---
R_cur = np.eye(3)
omega = np.zeros(3)
kR, kw = 4.0, 3.0
err_se3 = np.zeros(N)
for i in range(N):
    eR_mat = R_des.T @ R_cur - R_cur.T @ R_des
    eR = 0.5 * vee(eR_mat)
    err_se3[i] = np.linalg.norm(eR)
    tau_se3 = -kR * eR - kw * omega
    omega += tau_se3 * dt
    # Обновление матрицы поворота по формуле Родригеса
    wx = omega * dt
    ang = np.linalg.norm(wx)
    if ang > 1e-12:
        K = np.array([[0,-wx[2],wx[1]],[wx[2],0,-wx[0]],[-wx[1],wx[0],0]]) / ang
        dR = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    else:
        dR = np.eye(3)
    R_cur = R_cur @ dR

# --- Сравнительные графики ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(t, err_euler, 'r-', lw=1.5, label='Ошибка PD по углам Эйлера')
axes[0].plot(t, err_se3, 'b-', lw=1.5, label='Ошибка ориентации SE(3)')
axes[0].set_xlabel('Время (с)'); axes[0].set_ylabel('Норма ошибки ориентации')
axes[0].set_title('Сравнение ошибок ориентации (цель: разворот на 170° вокруг оси Y)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(t, np.clip(err_euler, 1e-10, None), 'r-', lw=1.5, label='PD по углам Эйлера')
axes[1].semilogy(t, np.clip(err_se3, 1e-10, None), 'b-', lw=1.5, label='SE(3)')
axes[1].set_xlabel('Время (с)'); axes[1].set_ylabel('Ошибка ориентации (логарифм)')
axes[1].set_title('Сравнение в логарифмическом масштабе'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Глава 7: углы Эйлера и SE(3) — манёвр разворота на большой угол', fontsize=14)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/control7.png', dpi=150, bbox_inches='tight')
plt.show()
