import numpy as np
import matplotlib.pyplot as plt

def sample_circle_ddpm_style(
    n=1024,
    R=5.0,
    steps=50,
    noise_scale=0.15,
    eps=1e-8,
    return_trajectory=False
):
    """
    Имитация выборки DDPM по окружности.

    Параметры:
        n: число точек выборки
        R: радиус окружности
        steps: число шагов шумоподавления
        noise_scale: интенсивность шума в обратном процессе
        eps: защита от деления на ноль
        return_trajectory: возвращать ли всю траекторию выборки
    """
    x = np.random.randn(n, 2).astype(np.float32)

    trajectories = [x.copy()]
    for i in range(steps):
        progress = (i + 1) / steps
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x0_hat = R * x / (norm + eps)
        gamma = 0.05 + 0.95 * progress
        sigma = noise_scale * (1.0 - progress)
        noise = np.random.randn(n, 2).astype(np.float32)
        x = (1.0 - gamma) * x + gamma * x0_hat + sigma * noise

        trajectories.append(x.copy())

    if return_trajectory:
        return x.astype(np.float32), np.array(trajectories)

    return x.astype(np.float32)

x, traj = sample_circle_ddpm_style(
    n=3000,
    R=5.0,
    steps=50,
    noise_scale=0.15,
    return_trajectory=True
)

show_steps = [0, 1, 3, 5, 10, 20, 35, 50]

plt.figure(figsize=(16, 8))

for idx, s in enumerate(show_steps):
    plt.subplot(2, 4, idx + 1)
    xs = traj[s]
    plt.scatter(xs[:, 0], xs[:, 1], s=3, alpha=0.4)
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.axis("equal")
    plt.grid()
    plt.title(f"step {s}")

plt.tight_layout()

plt.savefig('ddpm_imitat.png', dpi=600)
plt.show()

