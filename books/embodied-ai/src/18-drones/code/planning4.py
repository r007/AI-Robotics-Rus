import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ============================================================
# Minimum Snap: замкнутое решение и решение через KKT/QP — проверка на 3 путевых точках (2 отрезка)
# Замкнутое решение: d_P = -R_PP^{-1} R_FP^T d_F
# Решение через KKT: задача QP с ограничениями равенства методом множителей Лагранжа
# ============================================================

N = 7   # Многочлен 7-й степени
M = 2   # 2 отрезка (3 путевые точки)
n_c = N + 1  # Число коэффициентов на отрезок = 8

# --- Определение путевых точек (одномерно, по оси x, упрощённая демонстрация) ---
waypoints = np.array([0.0, 5.0, 12.0])  # Начало, промежуточная точка, конец
T_seg = np.array([2.0, 3.0])            # Время каждого отрезка

def poly_coeff_vec(n, t, deriv=0):
    """Строка коэффициентов многочлена n-го порядка в t для производной порядка deriv"""
    c = np.zeros(n + 1)
    for i in range(deriv, n + 1):
        c[i] = np.prod(range(i, i - deriv, -1)) * t ** (i - deriv)
    return c

def build_Q_seg(n, T, kr=4):
    """Строим матрицу стоимости snap для одного отрезка"""
    Q = np.zeros((n + 1, n + 1))
    for i in range(kr, n + 1):
        for j in range(kr, n + 1):
            ci = np.prod(range(i, i - kr, -1))
            cj = np.prod(range(j, j - kr, -1))
            Q[i, j] = ci * cj * T ** (i + j - 2*kr + 1) / (i + j - 2*kr + 1)
    return Q

# ===================== Метод 1: решение KKT =====================
Q_all = block_diag(*[build_Q_seg(N, T) for T in T_seg])
dim = M * n_c

rows, rhs = [], []
def add_eq(row, val):
    rows.append(row); rhs.append(val)

# Ограничение начала: p=0, v=0, a=0
for d in range(3):
    r = np.zeros(dim); r[:n_c] = poly_coeff_vec(N, 0, d)
    add_eq(r, waypoints[0] if d == 0 else 0.0)
# Ограничение конца: p=12, v=0, a=0
for d in range(3):
    r = np.zeros(dim); r[n_c:] = poly_coeff_vec(N, T_seg[1], d)
    add_eq(r, waypoints[2] if d == 0 else 0.0)
# Ограничение положения промежуточной точки (конец отрезка 0)
r = np.zeros(dim); r[:n_c] = poly_coeff_vec(N, T_seg[0], 0)
add_eq(r, waypoints[1])
# Ограничения непрерывности (производные 0–3 порядка)
for d in range(4):
    r = np.zeros(dim)
    r[:n_c] = poly_coeff_vec(N, T_seg[0], d)
    r[n_c:] = -poly_coeff_vec(N, 0, d)
    add_eq(r, 0.0)

Aeq = np.array(rows); beq = np.array(rhs)
n_eq = len(beq)
KKT = np.zeros((dim + n_eq, dim + n_eq))
KKT[:dim, :dim] = 2 * Q_all
KKT[:dim, dim:] = Aeq.T
KKT[dim:, :dim] = Aeq
rhs_kkt = np.zeros(dim + n_eq); rhs_kkt[dim:] = beq
P_kkt = np.linalg.solve(KKT, rhs_kkt)[:dim]

# ===================== Метод 2: замкнутое решение =====================
# Строим матрицу отображения A_map: значения производных на концах отрезков -> коэффициенты многочлена
# На каждый отрезок 8 коэффициентов, на концах производные 0–3 порядка (4 в начале + 4 в конце = 8 уравнений)
def build_A_seg(n, T):
    """Строим матрицу ограничений концов отрезка (8x8)"""
    A = np.zeros((2*4, n+1))
    for d in range(4):
        A[d, :] = poly_coeff_vec(n, 0, d)       # Производная d-го порядка в начале
        A[4+d, :] = poly_coeff_vec(n, T, d)     # Производная d-го порядка в конце
    return A

A_map = block_diag(*[build_A_seg(N, T) for T in T_seg])  # 16x16
# d_alpha: значения производных на концах по отрезкам [seg0 начало (p,v,a,j), seg0 конец (...), seg1 начало (...), seg1 конец (...)]
# Матрица M объединяет непрерывность (конец seg0 = начало seg1) и отображает 16-мерный d_alpha в 12-мерный независимый вектор d
# Набор независимых концов: seg0 начало (4), промежуточная точка (4, общая), seg1 конец (4) = 12 переменных
M_mat = np.zeros((16, 12))
M_mat[0:4, 0:4] = np.eye(4)     # seg0 начало -> независимые переменные 0~3
M_mat[4:8, 4:8] = np.eye(4)     # seg0 конец -> переменные промежуточной точки 4~7
M_mat[8:12, 4:8] = np.eye(4)    # seg1 начало -> переменные промежуточной точки 4~7 (непрерывность)
M_mat[12:16, 8:12] = np.eye(4)  # seg1 конец -> независимые переменные 8~11

# Матрица C переупорядочивает независимые переменные в [Fixed; Free]
# Fixed: seg0 начало_p(0), seg0 начало_v(0), seg0 начало_a(0), промежуточная_p(5), seg1 конец_p(12), seg1 конец_v(0), seg1 конец_a(0) = 7 шт.
# Free: seg0 начало_j, промежуточная_v, промежуточная_a, промежуточная_j, seg1 конец_j = 5 шт.
fixed_idx = [0, 1, 2, 4, 8, 9, 10]  # Индексы в 12-мерных независимых переменных
free_idx  = [3, 5, 6, 7, 11]
d_F = np.array([waypoints[0], 0, 0, waypoints[1], waypoints[2], 0, 0])
n_fixed, n_free = len(fixed_idx), len(free_idx)

C_mat = np.zeros((12, 12))
for i, fi in enumerate(fixed_idx):
    C_mat[fi, i] = 1.0
for i, fi in enumerate(free_idx):
    C_mat[fi, n_fixed + i] = 1.0

# K = A_map^{-1} @ M @ C
K = np.linalg.solve(A_map, M_mat @ C_mat)

# R = K^T Q K, блоки R_FF, R_FP, R_PF, R_PP
R_full = K.T @ Q_all @ K
R_FF = R_full[:n_fixed, :n_fixed]
R_FP = R_full[:n_fixed, n_fixed:]
R_PP = R_full[n_fixed:, n_fixed:]

# Ключевая формула замкнутого решения
d_P = -np.linalg.solve(R_PP, R_FP.T @ d_F)
d_prime = np.concatenate([d_F, d_P])
P_closed = K @ d_prime

# ===================== Выборка и сравнительные графики =====================
def sample_traj(P_vec, T_seg, N):
    t_all, pos_kkt, vel_kkt = [], [], []
    t_offset = 0.0
    for k in range(len(T_seg)):
        ts = np.linspace(0, T_seg[k], 150)
        p = P_vec[k*n_c:(k+1)*n_c]
        pos_kkt.extend([poly_coeff_vec(N, t, 0) @ p for t in ts])
        vel_kkt.extend([poly_coeff_vec(N, t, 1) @ p for t in ts])
        t_all.extend(ts + t_offset)
        t_offset += T_seg[k]
    return np.array(t_all), np.array(pos_kkt), np.array(vel_kkt)

t1, pos1, vel1 = sample_traj(P_kkt, T_seg, N)
t2, pos2, vel2 = sample_traj(P_closed, T_seg, N)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
# Сравнение положений
axes[0].plot(t1, pos1, 'b-', linewidth=2.5, label='Решение KKT/QP')
axes[0].plot(t2, pos2, 'r--', linewidth=2, label='Замкнутое решение')
t_wp = [0, T_seg[0], T_seg[0]+T_seg[1]]
axes[0].plot(t_wp, waypoints, 'ko', markersize=8, zorder=5, label='Путевые точки')
axes[0].set_ylabel('Положение (м)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_title('Minimum Snap: замкнутое решение и KKT полностью совпадают — проверка')

# Сравнение скоростей
axes[1].plot(t1, vel1, 'b-', linewidth=2.5, label='Скорость KKT/QP')
axes[1].plot(t2, vel2, 'r--', linewidth=2, label='Скорость замкнутого решения')
axes[1].set_ylabel('Скорость (м/с)'); axes[1].set_xlabel('Время (с)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Выводим ошибку
err = np.max(np.abs(P_kkt - P_closed))
print(f"Максимальная абсолютная ошибка коэффициентов двух методов: {err:.2e}")
fig.text(0.5, 0.01, f'Максимальная абсолютная ошибка коэффициентов: {err:.2e} (проверка эквивалентности двух методов)',
         ha='center', fontsize=11, color='green')
plt.tight_layout(rect=[0, 0.03, 1, 1])
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/planning4.png', dpi=150, bbox_inches='tight')
plt.show()