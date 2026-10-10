import numpy as np
import matplotlib.pyplot as plt

def sample_circle_vae_style(n=1024, R=5.0, eps=1e-8):
    """
    Имитация выборки VAE по окружности.
    Параметры:
        n: число точек выборки
        R: радиус окружности
        eps: малая константа для защиты от деления на ноль
    """
    z = np.random.randn(n, 2).astype(np.float32)

    norm = np.linalg.norm(z, axis=1, keepdims=True)

    x = R * z / (norm + eps)

    return x.astype(np.float32), z.astype(np.float32)

x, z = sample_circle_vae_style(n=3000, R=5.0)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(z[:, 0], z[:, 1], s=5, alpha=0.4)
plt.axis("equal")
plt.grid()
plt.title("Latent Space: z ~ N(0, I)")

plt.subplot(1, 2, 2)
plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.4)
plt.axis("equal")
plt.grid()
plt.title("Generated Circle: x = R z / ||z||")

plt.savefig('vae_imitat.png', dpi=600)
plt.show()


