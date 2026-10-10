import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ============================================================
# Генерация траектории Minimum Snap — дрон последовательно пролетает несколько путевых точек
# Замкнутое решение задачи QP с ограничениями-равенствами через систему KKT (метод множителей Лагранжа)
# ============================================================

# --- Определение путевых точек (двумерные: x, y) ---
waypoints = np.array([
    [0.0, 0.0],   # Старт
    [2.0, 2.5],   # Промежуточная точка 1
    [5.0, 4.0],   # Промежуточная точка 2
    [8.0, 3.0],   # Промежуточная точка 3
    [10.0, 5.0],  # Финиш
])
M = len(waypoints) - 1  # 4 отрезка
N = 7                    # Многочлен 7-й степени (8 коэффициентов, minimum snap)

# --- Распределение времени (пропорционально длине отрезков, общее время 8 с) ---
dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
T_total = 8.0
T_seg = T_total * dists / dists.sum()  # Время каждого отрезка

def build_Q_seg(n, T, kr=4):
    """Строим матрицу стоимости snap Q для одного отрезка (относительное время, t in [0, T])"""
    Q = np.zeros((n+1, n+1))
    for i in range(kr, n+1):
        for j in range(kr, n+1):
            ci = np.prod(range(i, i-kr, -1))   # i!/(i-kr)!
            cj = np.prod(range(j, j-kr, -1))
            Q[i, j] = ci * cj * T**(i+j-2*kr+1) / (i+j-2*kr+1)
    return Q

def poly_coeff_vec(n, t, deriv=0):
    """Возвращает строку коэффициентов для производной порядка deriv в момент t для многочлена n-й степени"""
    c = np.zeros(n+1)
    for i in range(deriv, n+1):
        c[i] = np.prod(range(i, i-deriv, -1)) * t**(i-deriv)
    return c

# --- Строим общую матрицу Q (блочно-диагональная) ---
dim_total = M * (N+1)
Q_all = np.zeros((dim_total, dim_total))
for k in range(M):
    idx = k * (N+1)
    Q_all[idx:idx+N+1, idx:idx+N+1] = build_Q_seg(N, T_seg[k])

# --- Строим матрицу ограничений равенства Aeq @ P = beq ---
constraints_rows = []
constraints_rhs_x = []
constraints_rhs_y = []

def add_constraint(row_vec, rhs_x, rhs_y):
    constraints_rows.append(row_vec)
    constraints_rhs_x.append(rhs_x)
    constraints_rhs_y.append(rhs_y)

# Ограничения начальной точки: положение, скорость, ускорение, рывок (отрезок 0, t=0)
for d in range(4):
    row = np.zeros(dim_total)
    row[0:N+1] = poly_coeff_vec(N, 0, d)
    val = waypoints[0] if d == 0 else np.array([0.0, 0.0])
    add_constraint(row, val[0], val[1])

# Ограничения конечной точки: положение, скорость, ускорение, рывок (последний отрезок, t=T_M)
for d in range(4):
    row = np.zeros(dim_total)
    idx = (M-1)*(N+1)
    row[idx:idx+N+1] = poly_coeff_vec(N, T_seg[-1], d)
    val = waypoints[-1] if d == 0 else np.array([0.0, 0.0])
    add_constraint(row, val[0], val[1])

# Ограничения положения промежуточных путевых точек и условия непрерывности
for k in range(M-1):
    # Конечное положение k-го отрезка = waypoints[k+1]
    row = np.zeros(dim_total)
    idx = k * (N+1)
    row[idx:idx+N+1] = poly_coeff_vec(N, T_seg[k], 0)
    add_constraint(row, waypoints[k+1, 0], waypoints[k+1, 1])
    # Непрерывность: конец k-го отрезка = начало (k+1)-го (производные 0–3 порядка)
    for d in range(4):
        row = np.zeros(dim_total)
        row[idx:idx+N+1] = poly_coeff_vec(N, T_seg[k], d)
        idx2 = (k+1) * (N+1)
        row[idx2:idx2+N+1] = -poly_coeff_vec(N, 0, d)
        add_constraint(row, 0.0, 0.0)

Aeq = np.array(constraints_rows)
beq_x = np.array(constraints_rhs_x)
beq_y = np.array(constraints_rhs_y)

# --- Решение системы KKT: [2Q, A^T; A, 0] [P; lam] = [0; b] ---
n_eq = Aeq.shape[0]
KKT = np.zeros((dim_total + n_eq, dim_total + n_eq))
KKT[:dim_total, :dim_total] = 2 * Q_all
KKT[:dim_total, dim_total:] = Aeq.T
KKT[dim_total:, :dim_total] = Aeq

def solve_axis(beq):
    rhs = np.zeros(dim_total + n_eq)
    rhs[dim_total:] = beq
    sol = np.linalg.solve(KKT, rhs)
    return sol[:dim_total]

Px = solve_axis(beq_x)
Py = solve_axis(beq_y)

# --- Дискретизация траектории и расчёт производных ---
t_all, x_all, y_all = [], [], []
derivs = {d: ([], []) for d in range(5)}  # Производные 0–4 порядка
t_cum = 0.0
for k in range(M):
    ts = np.linspace(0, T_seg[k], 100)
    idx = k * (N+1)
    px, py = Px[idx:idx+N+1], Py[idx:idx+N+1]
    for d in range(5):
        cx = np.array([poly_coeff_vec(N, t, d) @ px for t in ts])
        cy = np.array([poly_coeff_vec(N, t, d) @ py for t in ts])
        derivs[d][0].extend(cx)
        derivs[d][1].extend(cy)
    t_all.extend(ts + t_cum)
    t_cum += T_seg[k]

# --- Графики ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Левый график: двумерная траектория
ax = axes[0]
ax.plot(derivs[0][0], derivs[0][1], 'b-', linewidth=2, label='Траектория Minimum Snap')
ax.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', markersize=8, linewidth=1, label='Путевые точки')
for i, wp in enumerate(waypoints):
    ax.annotate(f'P{i}', wp, textcoords="offset points", xytext=(5, 8), fontsize=10)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('Двумерная траектория дрона Minimum Snap'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Правый график: изменение производных во времени (только компонента по оси x)
ax2 = axes[1]
labels = ['Положение (м)', 'Скорость (м/с)', 'Ускорение (м/с²)', 'Jerk', 'Snap']
for d in range(5):
    ax2.plot(t_all, derivs[d][0], label=labels[d], alpha=0.85)
ax2.set_xlabel('Время (с)'); ax2.set_ylabel('Компонента по оси X')
ax2.set_title('Кривые производных по оси X'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/planning2.png', dpi=150, bbox_inches='tight')
plt.show()