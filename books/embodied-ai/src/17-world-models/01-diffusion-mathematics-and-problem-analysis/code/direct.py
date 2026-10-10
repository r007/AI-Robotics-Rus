import numpy as np
import matplotlib.pyplot as plt

def sample_circle_direct(n=1024, R=5.0):
    """
    Непосредственная выборка по окружности.
    Параметры:
        n: число точек выборки
        R: радиус окружности
    """
    theta = 2 * np.pi * np.random.rand(n)

    x1 = R * np.cos(theta)
    x2 = R * np.sin(theta)

    data = np.stack([x1, x2], axis=1).astype(np.float32)
    return data
data = sample_circle_direct(n=3000, R=5.0)

plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
plt.axis("equal")
plt.grid()
plt.title("Direct Sampling on Circle")
plt.savefig('direct.png', dpi=600)
plt.show()

