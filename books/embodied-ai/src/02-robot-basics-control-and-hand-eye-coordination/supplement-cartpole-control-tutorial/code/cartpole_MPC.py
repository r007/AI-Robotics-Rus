import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import random
import gym
import math
import time
from scipy import linalg
from collections import deque
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import *
from copy import deepcopy, copy
from matplotlib import animation

from cartpole_env import *

from matplotlib import font_manager
font_path = "./AiDianFengYaHei（ShangYongMianFei）-2.ttf"  # Заменить на правильный путь
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

DIRECT_MAG=True
RANDOM_NOISE=False

if DIRECT_MAG:
    env=CartPoleEnv()
else:
    env = gym.make('CartPole-v1')


L = env.length
M = env.masscart
m = env.masspole
del_t = env.tau
g = env.gravity
state = env.reset()
x_goal = np.array([0.0,
                    0.0,
                    0.0,
                    0.0])
Q = np.eye(4)
Q = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
R = 0.5*np.eye(1)

A = np.eye(4) + del_t * np.array([[0, 1, 0, 0],
                [0, 0, (-3*m*g)/(m+4*M), 0],
                [0, 0, 0, 1],
                [0, 0, (3*(M+m)*g)/(L*(m+4*M)), 0]])
B = del_t * np.array([[0],
                [(1/(M+m))+((3*m)/(m+4*M))],
                [0],
                [-(3)/(L*(m+4*M))]])

# Системные параметры
N = 50  # Горизонт прогноза
k_steps = 100

# Матрицы состояния A и входа B
n = A.shape[0]
p = B.shape[1]

print(n)
print(p)
# Матрицы Q, F, R
F = np.eye(4)

# Задаём количество шагов (step)
k_steps = 401

# Выделяем память под все состояния x и задаём начальное состояние
X_k = np.zeros((n, k_steps+1))
X_k[:,0] = state
# Выделяем память под все входные управления u
U_k = np.zeros((p, k_steps))

def save_frames_as_gif(frames, path):
    filename = 'MPC_'+ 'CartPole-v1' + '.gif'
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow')

# Вычисляем матрицы функции стоимости QP
def get_QPMatrix(A, B, Q, R, F, N):
    M = np.vstack([np.eye(n), np.zeros((N*n, n))])
    C = np.zeros(((N+1)*n, N*p))
    temp = np.eye(n)
    for i in range(1,N+1):
        rows = i * n + np.arange(n)
        C[rows,:] = np.hstack([temp @ B, C[rows-n, :-p]])
        temp = A @ temp
        M[rows,:] = temp

    Q_ = np.kron(np.eye(N), Q)
    rows_Q, cols_Q = Q_.shape
    rows_F, cols_F = F.shape
    Q_bar = np.zeros((rows_Q+rows_F, cols_Q+cols_F))
    Q_bar[:rows_Q, :cols_Q] = Q_
    Q_bar[rows_Q:, cols_Q:] = F
    R_bar = np.kron(np.eye(N), R)

    # G = M.T @ Q_bar @ M
    E = C.T @ Q_bar @ M
    H = C.T @ Q_bar @ C + R_bar
    return E, H


# Определяем задачу оптимизации MPC
def mpc_prediction(x_k, E, H, N, p):
    # Определяем оптимизируемую переменную
    U = ca.SX.sym('U', N * p)
    # Определяем целевую функцию
    objective = 0.5 * ca.mtimes([U.T, H, U]) + ca.mtimes([U.T, E, x_k])
    qp = {'x': U, 'f': objective}
    opts = {'print_time': False, 'ipopt': {'print_level': 0}}
    solver = ca.nlpsol('solver', 'ipopt', qp, opts)

    # Решаем задачу
    sol = solver()
    # Извлекаем оптимальное решение
    U_k = sol['x'].full().flatten()
    u_k = 10*U_k[:p]  # Берём первый результат

    return u_k

if __name__ == "__main__":
    # Get QP Matrix
    E, H = get_QPMatrix(A, B, Q, R, F, N)
    done = False
    i=0
    step_a = 0
    frames = []
    while abs(state[2]<2) & (step_a < k_steps):
        env.render()
        frames.append(env.render(mode='rgb_array'))
        if RANDOM_NOISE and random.random()>0.99:
            i=2

        if i>0:
            if DIRECT_MAG:
                action = 10
            else:
                action=1
            i-=1
        else:
            action = mpc_prediction(state, E, H, N, p)

        next_state, reward, done, _ = env.step(action)
        state = next_state
        X_k[:,step_a+1] = state
        U_k[:,step_a] = action
        step_a += 1
        print(step_a)
    env.close()
    save_frames_as_gif(frames, path = './')

    # Рисуем результаты
    plt.subplot(2, 1, 1)
    plt.plot(X_k[0, :], label=f"x")
    plt.plot(X_k[1, :], label=f"x_dot")
    plt.plot(X_k[2, :], label=f"theta")
    plt.plot(X_k[3, :], label=f"theta_dot")
    plt.legend()
    plt.title("Переменные состояния MPC")
    plt.xlabel("Шаг по времени")
    plt.ylabel("Значения состояния")
    # Второй подграфик: управляющее воздействие
    plt.subplot(2, 1, 2)
    for i in range(U_k.shape[0]):
        plt.plot(U_k[i, :], label=f"u{i+1}")
    plt.legend()
    plt.title("Управляющее воздействие MPC")
    plt.xlabel("Шаг по времени")
    plt.ylabel("Значения управляющего воздействия")
    
    # Настраиваем компоновку и показываем
    plt.tight_layout()
    plt.savefig('MPC', dpi=1000)
    plt.show()
