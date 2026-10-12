# 11.9 Adadelta

Adadelta — это ещё один вариант AdaGrad (раздел 11.7).
Основное отличие в том, что он уменьшает объём адаптации скорости обучения к координатам.
Кроме того, в широком смысле говорят, что у Adadelta нет скорости обучения, поскольку он использует сами величины изменений в качестве калибровки для будущих изменений.
Алгоритм Adadelta предложен в работе (Zeiler, 2012).

## 11.9.1 Алгоритм Adadelta

Коротко говоря, Adadelta использует две переменные состояния: $\mathbf{s}_t$ хранит экспоненциально затухающее среднее квадратов градиентов, а $\Delta\mathbf{x}_t$ хранит экспоненциально затухающее среднее квадратов изменений самих параметров модели. Обратите внимание, что для совместимости с другими публикациями и реализациями мы используем оригинальные обозначения и названия авторов. (Нет иной веской причины, по которой в методе импульса, AdaGrad, RMSProp и Adadelta для одного и того же назначения следовало бы использовать разные греческие буквы.)

Ниже приведены технические детали Adadelta. Обозначив параметр через $\rho$, получаем следующее обновление с затуханием, аналогичное разделу 11.8:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Отличие от раздела 11.8 состоит в том, что обновление выполняется с помощью перемасштабированного градиента $\mathbf{g}_t'$:

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Что же представляет собой скорректированный градиент $\mathbf{g}_t'$? Его можно вычислить так:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

где $\Delta \mathbf{x}_{t-1}$ — экспоненциально затухающее среднее квадратов перемасштабированных градиентов $\mathbf{g}_t'$.
Мы инициализируем $\Delta \mathbf{x}_{0}$ нулём, а затем на каждом шаге обновляем его с помощью $\mathbf{g}_t'$:

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

а $\epsilon$ (малая величина, например $10^{-5}$) добавляется для численной устойчивости.

## 11.9.2 Реализация кода

Adadelta требует хранить две переменные состояния для каждой переменной: $\mathbf{s}_t$ и $\Delta\mathbf{x}_t$. Это приводит к следующей реализации.

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # Обновления на месте через [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

**PyTorch:**

```python
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # Обновления на месте через [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

**TensorFlow:**

```python
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

**PaddlePaddle:**

```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros(shape=(feature_dim, 1)), d2l.zeros(shape=(1, )) 
    delta_w, delta_b = d2l.zeros(shape=(feature_dim, 1)), d2l.zeros(shape=(1, )) 
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    a = []
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with paddle.no_grad():
            # Обновления на месте через [:]
            s[:] = rho * s + (1 - rho) * paddle.square(p.grad)
            g = (paddle.sqrt(delta + eps) / paddle.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.zero_()
        a.append(p)
    return a
```

Для каждого обновления параметров выбор $\rho = 0.9$ соответствует 10 периодам полураспада. В результате получаем:

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Для краткой реализации достаточно использовать алгоритм Adadelta из высокоуровневого API.

```python
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

**PyTorch:**

```python
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

**TensorFlow:**

```python
# Adadelta не сходится при скорости обучения по умолчанию,
# но сходится при lr=5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

**PaddlePaddle:**

```python
trainer = paddle.optimizer.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

## Резюме

* У Adadelta нет параметра скорости обучения. Вместо этого он использует скорость изменения самих параметров, чтобы корректировать скорость обучения.
* Adadelta требует двух переменных состояния для хранения второго момента градиента и изменений параметров.
* Adadelta использует экспоненциально затухающие средние, чтобы поддерживать скользящие оценки соответствующих статистик.

## Упражнения

1. Что произойдёт, если менять значение $\rho$?
1. Покажите, как реализовать алгоритм без использования $\mathbf{g}_t'$. Почему это может быть хорошей идеей?
1. Действительно ли у Adadelta скорость обучения равна нулю? Можно ли найти задачу оптимизации, которую Adadelta не может решить?
1. Сравните поведение сходимости Adadelta с поведением AdaGrad и RMSProp.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5771)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5772)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/5773)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11854)

## Литература

* Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method. *arXiv preprint arXiv:1212.5701*.
