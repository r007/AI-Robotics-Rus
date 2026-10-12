# 11.8 Алгоритм RMSProp

Одна из ключевых проблем алгоритма AdaGrad из раздела 11.7 состоит в том, что скорость обучения уменьшается по заранее заданному графику $\mathcal{O}(t^{-\frac{1}{2}})$.
Хотя так обычно и делают для выпуклых задач, для невыпуклых задач, с которыми мы сталкиваемся в глубоком обучении, такое поведение может быть не идеальным.
Однако как предобусловливатель покоординатная адаптивность AdaGrad очень желательна.

Тилеман и Хинтон (Tieleman and Hinton, 2012) предложили алгоритм RMSProp как простое исправление, которое отделяет расписание скорости обучения от покоординатной адаптации скорости обучения.
Проблема в том, что AdaGrad суммирует квадраты градиентов $\mathbf{g}_t$ в векторе состояния $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$.
Поэтому из-за отсутствия нормировки $\mathbf{s}_t$ продолжает расти, примерно линейно, и это происходит и тогда, когда алгоритм уже сходится.

Один из способов решить эту проблему — использовать $\mathbf{s}_t / t$.
Для разумных распределений $\mathbf{g}_t$ это значение будет сходиться.
К сожалению, эффект ограничения может проявиться не скоро, поскольку эта процедура помнит всю траекторию значений.
Другой способ — использовать экспоненциально затухающее среднее, как в методе импульса: $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$, где параметр $\gamma > 0$.
Если сохранить всё остальное без изменений, получится алгоритм RMSProp.

## 11.8.1 Алгоритм

Запишем эти уравнения подробно.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Константа $\epsilon > 0$ обычно выбирается равной $10^{-6}$, чтобы не допустить деления на ноль или слишком большого шага.
С учётом этого расширения мы теперь можем свободно управлять скоростью обучения $\eta$, не обращая внимания на масштабирование, применяемое к каждой координате.
Что касается экспоненциально затухающего среднего, можно использовать те же рассуждения, что и для метода импульса.
Раскроем определение $\mathbf{s}_t$:

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Как и в разделе 11.6.1.3 о методе импульса, мы используем $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$.
Поэтому сумма весов нормирована к $1$, а период полураспада наблюдения равен $\gamma^{-1}$.
Изобразим веса для разных значений $\gamma$ за последние 40 шагов по времени.

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
import math
```

**TensorFlow:**

```python
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import math
```

```python
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'γ = {gamma:.2f}')
d2l.plt.xlabel('время');
```

## 11.8.2 Реализация с нуля

Как и раньше, мы используем квадратичную функцию $f(\mathbf{x})=0.1x_1^2+2x_2^2$, чтобы наблюдать за траекторией алгоритма RMSProp.
Вспомним, что в разделе 11.7 при использовании AdaGrad с коэффициентом скорости обучения 0.4 переменная на поздних стадиях алгоритма двигалась очень медленно, поскольку скорость обучения быстро убывала.
В RMSProp этого не происходит, потому что $\eta$ задаётся отдельно.

```python
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Далее реализуем алгоритм RMSProp для глубокой сети.

**MXNet, PyTorch:**

```python
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

**PaddlePaddle:**

```python
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros([1])
    return (s_w, s_b)
```

**TensorFlow:**

```python
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

**MXNet:**

```python
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

**PyTorch:**

```python
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

**TensorFlow:**

```python
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

**PaddlePaddle:**

```python
def rmsprop(params, states, hyperparams):
    a = []
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with paddle.no_grad():
            s[:] = gamma * s + (1 - gamma) * paddle.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / paddle.sqrt(s + eps)
        p.grad.zero_()
        a.append(p)
    return a
```

Начальную скорость обучения установим равной 0.01, а весовой коэффициент $\gamma$ — 0.9.
То есть $\mathbf{s}$ накапливает среднее значение последних $1/(1-\gamma) = 10$ наблюдений квадратов градиентов.

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 11.8.3 Краткая реализация

Мы можем использовать алгоритм RMSProp, уже реализованный во фреймворке глубокого обучения, для обучения модели.

```python
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

**PyTorch:**

```python
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

**TensorFlow:**

```python
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

**PaddlePaddle:**

```python
trainer = paddle.optimizer.RMSProp
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Резюме

* Алгоритм RMSProp очень похож на AdaGrad, поскольку оба используют квадраты градиентов для масштабирования коэффициентов.
* RMSProp и метод импульса используют экспоненциально затухающее среднее. Однако RMSProp применяет этот приём, чтобы настроить покоординатный предобусловливатель.
* В экспериментах скорость обучения должна подбираться экспериментатором.
* Коэффициент $\gamma$ определяет, насколько долго учитывается история при настройке масштаба для каждой координаты.

## Упражнения

1. Что произойдёт в эксперименте, если задать $\gamma = 1$? Почему?
1. Поверните задачу оптимизации так, чтобы минимизировать $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Что произойдёт со сходимостью?
1. Попробуйте применить RMSProp к настоящей задаче машинного обучения, например, к обучению на Fashion-MNIST. Подберите скорость обучения, перебирая разные значения.
1. Нужно ли корректировать $\gamma$ по ходу оптимизации? Насколько чувствителен RMSProp к этому параметру?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/4321)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/4322)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/4323)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11853)

## Литература

* Tieleman, T., & Hinton, G. (2012). Lecture 6.5-RMSProp: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*, 4, 26–31.
