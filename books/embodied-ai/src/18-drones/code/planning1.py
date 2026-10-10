import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Встроенный в Windows шрифт SimHei
plt.rcParams['axes.unicode_minus'] = False    # Исправляет отображение минуса в виде квадрата
# ============================================================
# Интерполяция пятой степенью между двумя точками — дрон взлетает из состояния покоя в целевую точку
# По заданным позициям, скоростям и ускорениям начала и конца решается матричное уравнение 6x6 для коэффициентов многочлена
# ============================================================

# --- Граничные условия (имитируем вертикальный взлёт дрона) ---
T = 3.0          # Время полёта 3 секунды
# Момент t=0: покой на земле
pos0, vel0, acc0 = 0.0, 0.0, 0.0
# Момент t=T: достигнута высота 10 м, зависание (скорость и ускорение равны нулю)
posT, velT, accT = 10.0, 0.0, 0.0

# --- Строим матрицу ограничений 6x6 ---
# X(t) = p0 + p1*t + p2*t^2 + p3*t^3 + p4*t^4 + p5*t^5
A = np.array([
    [1, 0,   0,       0,         0,          0        ],   # X(0)   = pos0
    [0, 1,   0,       0,         0,          0        ],   # X'(0)  = vel0
    [0, 0,   2,       0,         0,          0        ],   # X''(0) = acc0
    [1, T,   T**2,    T**3,      T**4,       T**5     ],   # X(T)   = posT
    [0, 1,   2*T,     3*T**2,    4*T**3,     5*T**4   ],   # X'(T)  = velT
    [0, 0,   2,       6*T,       12*T**2,    20*T**3  ],   # X''(T) = accT
])
b = np.array([pos0, vel0, acc0, posT, velT, accT])

# --- Решаем коэффициенты многочлена ---
coeffs = np.linalg.solve(A, b)
print("Коэффициенты многочлена [p0, p1, p2, p3, p4, p5]:", np.round(coeffs, 4))

# --- Дискретизация по времени и расчёт положения, скорости и ускорения ---
t = np.linspace(0, T, 200)
pos = sum(coeffs[i] * t**i for i in range(6))
vel = sum(i * coeffs[i] * t**(i-1) for i in range(1, 6))
acc = sum(i * (i-1) * coeffs[i] * t**(i-2) for i in range(2, 6))

# --- Графики ---
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
titles = ["Положение (м)", "Скорость (м/с)", "Ускорение (м/с²)"]
data = [pos, vel, acc]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for ax, title, y, c in zip(axes, titles, data, colors):
    ax.plot(t, y, color=c, linewidth=2)
    ax.set_ylabel(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)

# Отмечаем начало и конец
axes[0].plot(0, pos0, 'ko', markersize=8, label='Точка взлёта')
axes[0].plot(T, posT, 'r*', markersize=12, label='Целевая точка зависания')
axes[0].legend(fontsize=10)

axes[-1].set_xlabel("Время (с)", fontsize=12)
fig.suptitle("Траектория пятой степени — взлёт с земли и зависание на высоте 10 м", fontsize=13)
plt.tight_layout()
import os; os.makedirs('assets', exist_ok=True)
plt.savefig('assets/planning1.png', dpi=150, bbox_inches='tight')
plt.show()