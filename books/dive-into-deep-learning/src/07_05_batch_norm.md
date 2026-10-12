# 7.5 Пакетная нормализация

Обучение глубоких нейронных сетей очень трудно, особенно когда нужно добиться их сходимости за короткое время.
В этом разделе мы рассмотрим *пакетную нормализацию* (batch normalization) (Ioffe, Szegedy, 2015) — популярный и эффективный приём, который ускоряет сходимость глубоких сетей. В сочетании с остаточными блоками, о которых пойдёт речь в разделе 7.6, пакетная нормализация позволяет обучать сети в более чем 100 слоёв.

## 7.5.1 Обучение глубоких сетей

Зачем нужны слои пакетной нормализации? Вспомним некоторые практические трудности, возникающие при обучении нейронных сетей.

Во-первых, способ предобработки данных часто сильно влияет на итоговый результат. Вспомним пример с применением многослойного перцептрона для предсказания цен на жильё (см. раздел 4.10). Работая с реальными данными, первым шагом мы стандартизуем входные признаки, так чтобы их среднее значение было равно 0, а дисперсия — 1. Интуитивно такая стандартизация хорошо сочетается с оптимизатором, поскольку выравнивает масштаб параметров.

Во-вторых, в типичных многослойных перцептронах и свёрточных сетях во время обучения промежуточные переменные (например, выходы аффинного преобразования в многослойном перцептроне) могут иметь очень широкий диапазон значений. Это касается как изменений от слоя к слою, так и разброса внутри одного слоя; кроме того, параметры модели меняются по мере обновлений в ходе обучения. Создатели пакетной нормализации неформально предположили, что такой сдвиг распределений этих переменных может мешать сходимости сети. Интуитивно можно предположить, что если значения переменных одного слоя в 100 раз больше, чем у другого, то для компенсации может понадобиться корректировка скорости обучения.

В-третьих, глубокие сети сложны и склонны к переобучению. Поэтому регуляризация становится ещё важнее.

Пакетная нормализация применяется к отдельному опциональному слою (а можно и ко всем слоям). Её принцип таков: на каждой итерации обучения мы сначала нормализуем входы, вычитая из них среднее и деля на стандартное отклонение, причём обе статистики вычисляются по текущему мини-батчу. Затем применяем коэффициент масштаба и смещение.
Именно из-за этой *нормализации*, основанной на статистиках *батча*, метод и получил название *пакетная нормализация*.

Заметим, что если применять пакетную нормализацию к мини-батчу размера 1, мы ничему не научимся: после вычитания среднего каждый скрытый элемент станет равен 0. Поэтому метод эффективен и устойчив только при достаточно большом мини-батче. Кроме того, при использовании пакетной нормализации выбор размера батча может быть важнее, чем без неё.

Формально пусть $\mathbf{x} \in \mathcal{B}$ — вход из мини-батча $\mathcal{B}$. Тогда пакетная нормализация $\mathrm{BN}$ преобразует $\mathbf{x}$ по формуле:

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
(7.5.1)

В формуле (7.5.1) $\hat{\boldsymbol{\mu}}_\mathcal{B}$ — выборочное среднее мини-батча $\mathcal{B}$, а $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ — выборочное стандартное отклонение мини-батча $\mathcal{B}$.
После стандартизации получаемый мини-батч имеет среднее 0 и единичную дисперсию. Поскольку единичная дисперсия (как и некоторые другие «магические» числа) — субъективный выбор, обычно вводят *параметр растяжения* (scale) $\boldsymbol{\gamma}$ и *параметр сдвига* (shift) $\boldsymbol{\beta}$, имеющие ту же форму, что и $\mathbf{x}$. Заметим, что $\boldsymbol{\gamma}$ и $\boldsymbol{\beta}$ — параметры, которые обучаются вместе с остальными параметрами модели.

Поскольку во время обучения амплитуда изменений промежуточных слоёв не должна быть слишком резкой, пакетная нормализация центрирует каждый слой и приводит его к заданным среднему и масштабу (через $\hat{\boldsymbol{\mu}}_\mathcal{B}$ и $\hat{\boldsymbol{\sigma}}_\mathcal{B}$).

Формально $\hat{\boldsymbol{\mu}}_\mathcal{B}$ и $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ из формулы (7.5.1) вычисляются так:

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

Заметим, что к оценке дисперсии добавляется малая константа $\epsilon > 0$, чтобы мы никогда не делили на ноль, даже когда эмпирическая оценка дисперсии обращается в нуль. Оценки $\hat{\boldsymbol{\mu}}_\mathcal{B}$ и $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ с помощью шумных оценок среднего и дисперсии компенсируют проблему масштабирования. На первый взгляд шум кажется проблемой, но на самом деле он полезен.

Это известная в глубоком обучении тема. По причинам, которые пока не объяснены теоретически, различные источники шума при оптимизации обычно приводят к более быстрому обучению и меньшему переобучению: такое изменение действует как форма регуляризации.
В ряде предварительных работ (Teye et al., 2018) и (Luo et al., 2018) свойства пакетной нормализации связываются с байесовскими априорными распределениями. Эти теории объясняют, почему пакетная нормализация лучше всего работает со средними размерами батча в диапазоне $50 \sim 100$.

Кроме того, слой пакетной нормализации работает по-разному в «режиме обучения» (нормализация по статистикам мини-батча) и в «режиме предсказания» (нормализация по статистикам всего набора данных).
Во время обучения нельзя оценить среднее и дисперсию по всему набору данных, поэтому модель обучается, опираясь на среднее и дисперсию каждого мини-батча.
В режиме предсказания среднее и дисперсию пакетной нормализации можно точно вычислить по всему набору данных.

Теперь посмотрим, как пакетная нормализация работает на практике.

## 7.5.2 Слои пакетной нормализации

Напомним, что ключевое отличие пакетной нормализации от других слоёв в том, что она работает на полном мини-батче, поэтому мы не можем, как раньше, игнорировать размер батча. Рассмотрим два случая: полносвязные слои и свёрточные слои. Их реализация пакетной нормализации немного различается.

### Полносвязные слои

Обычно слой пакетной нормализации помещают в полносвязном слое между аффинным преобразованием и функцией активации.
Пусть вход полносвязного слоя — $\mathbf{x}$, весовой параметр и параметр смещения — $\mathbf{W}$ и $\mathbf{b}$, функция активации — $\phi$, а оператор пакетной нормализации — $\mathrm{BN}$.
Тогда выход полносвязного слоя с пакетной нормализацией вычисляется так:

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Напомним, что среднее и дисперсия вычисляются по тому же мини-батчу, к которому применяется преобразование.

### Свёрточные слои

Аналогично для свёрточных слоёв пакетную нормализацию можно применять после свёрточного слоя и перед нелинейной функцией активации.
Когда у свёртки несколько выходных каналов, пакетную нормализацию нужно выполнять для **каждого** выходного канала: у каждого канала свои параметры растяжения и смещения, и оба они скалярные.
Пусть мини-батч содержит $m$ примеров, а выход свёртки для каждого канала имеет высоту $p$ и ширину $q$. Тогда для свёрточного слоя мы одновременно выполняем пакетную нормализацию для каждого выходного канала по всем $m \cdot p \cdot q$ элементам.
Поэтому при вычислении среднего и дисперсии мы собираем значения по всем пространственным позициям, а затем применяем одни и те же среднее и дисперсию в пределах канала, нормализуя значения в каждой пространственной позиции.

### Пакетная нормализация в режиме предсказания

Как уже говорилось, пакетная нормализация ведёт себя по-разному в режимах обучения и предсказания.
Во-первых, при использовании обученной модели для предсказания нам больше не нужны шумовые оценки выборочного среднего и не нужно оценивать дисперсию по каждому мини-батчу.
Во-вторых, модель может понадобиться для предсказания по одному примеру за раз.
Обычно среднее и дисперсию по всему обучающему набору оценивают скользящим средним и используют их при предсказании, чтобы получать детерминированный результат.
Таким образом, как и исключение (dropout), слой пакетной нормализации даёт разные результаты в режимах обучения и предсказания.

## 7.5.3 Реализация с нуля

Ниже мы с нуля реализуем слой пакетной нормализации на тензорах.

**MXNet:**
```python
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Через autograd определяем, находимся ли мы в режиме обучения или предсказания
    if not autograd.is_training():
        # В режиме предсказания используем переданные скользящие среднее и дисперсию
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Для полносвязного слоя вычисляем среднее и дисперсию по измерению признаков
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # Для двумерного свёрточного слоя вычисляем среднее и дисперсию по измерению каналов (axis=1).
            # Здесь сохраняем форму X, чтобы потом можно было выполнять broadcasting
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # В режиме обучения стандартизуем с текущими средним и дисперсией
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Обновляем скользящие среднее и дисперсию
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Масштабирование и сдвиг
    return Y, moving_mean, moving_var
```

**PyTorch:**
```python
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Через is_grad_enabled определяем, находимся ли мы в режиме обучения или предсказания
    if not torch.is_grad_enabled():
        # В режиме предсказания используем переданные скользящие среднее и дисперсию
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Для полносвязного слоя вычисляем среднее и дисперсию по измерению признаков
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # Для двумерного свёрточного слоя вычисляем среднее и дисперсию по измерению каналов (axis=1).
            # Здесь сохраняем форму X, чтобы потом можно было выполнять broadcasting
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # В режиме обучения стандартизуем с текущими средним и дисперсией
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Обновляем скользящие среднее и дисперсию
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Масштабирование и сдвиг
    return Y, moving_mean.data, moving_var.data
```

**TensorFlow:**
```python
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Вычисляем обратный квадратный корень из скользящей дисперсии
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Масштабирование и сдвиг
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

**PaddlePaddle:**
```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum, is_training=True):
    # Обработка BN в режиме обучения отличается от обработки в режиме предсказания
    if not is_training:
        # В режиме предсказания используем переданные скользящие среднее и дисперсию
        X_hat = (X - moving_mean) / (moving_var + eps) ** 0.5
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Для полносвязного слоя вычисляем среднее и дисперсию по измерению признаков
            mean = paddle.mean(X)
            var = paddle.mean(((X - mean) ** 2))
        else:
            # Для двумерного свёрточного слоя вычисляем среднее и дисперсию по измерению каналов (axis=1).
            # Здесь сохраняем форму X, чтобы потом можно было выполнять broadcasting
            mean = paddle.mean(X, axis=(0, 2, 3), keepdim=True)
            var = paddle.mean(((X - mean) ** 2), axis=(0, 2, 3), keepdim=True)
        # В режиме обучения стандартизуем с текущими средним и дисперсией
        X_hat = (X - mean) / (var + eps) ** 0.5
        # Обновляем скользящие среднее и дисперсию
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Масштабирование и сдвиг
    return Y, moving_mean, moving_var
```

Теперь мы можем **создать корректный слой `BatchNorm`**. Этот слой хранит параметры: растяжение `gamma` и смещение `beta`, которые обновляются во время обучения. Кроме того, слой хранит скользящие средние значения среднего и дисперсии, чтобы использовать их при предсказании.

Отвлекаясь от деталей алгоритма, обратим внимание на базовый шаблон реализации слоя. Обычно математику задают отдельной функцией, например `batch_norm`. Затем эту функцию встраивают в пользовательский слой, код которого отвечает за перенос данных на устройство (например, GPU), выделение и инициализацию необходимых переменных, отслеживание скользящих значений (здесь — среднего и дисперсии) и т. д.
Для простоты мы не будем автоматически выводить форму входа, поэтому нужно указать полное число признаков. Не беспокойтесь: API пакетной нормализации в фреймворках решает эти задачи за нас, и мы покажем это позже.

**MXNet:**
```python
class BatchNorm(nn.Block):
    # num_features: число выходов полносвязного слоя или число выходных каналов свёрточного слоя.
    # num_dims: 2 — полносвязный слой, 4 — свёрточный слой
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Параметры растяжения и смещения, участвующие в вычислении градиентов и обновлении, инициализируем единицами и нулями соответственно
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # Переменные, которые не являются параметрами модели, инициализируем нулями и единицами
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # Если X не находится в памяти, копируем moving_mean и moving_var
        # в видеопамять, где находится X
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Сохраняем обновлённые moving_mean и moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

**PyTorch:**
```python
class BatchNorm(nn.Module):
    # num_features: число выходов полносвязного слоя или число выходных каналов свёрточного слоя.
    # num_dims: 2 — полносвязный слой, 4 — свёрточный слой
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Параметры растяжения и смещения, участвующие в вычислении градиентов и обновлении, инициализируем единицами и нулями соответственно
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # Переменные, которые не являются параметрами модели, инициализируем нулями и единицами
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # Если X не находится в памяти, копируем moving_mean и moving_var
        # в видеопамять, где находится X
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Сохраняем обновлённые moving_mean и moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

**TensorFlow:**
```python
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # Параметры растяжения и смещения, участвующие в вычислении градиентов и обновлении, инициализируем единицами и нулями соответственно
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # Переменные, которые не являются параметрами модели, инициализируем нулями и единицами
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

**PaddlePaddle:**
```python
class BatchNorm(nn.Layer):
    def __init__(self, num_features, num_dims=4):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Параметры растяжения и смещения, участвующие в вычислении градиентов и обновлении, инициализируем единицами и нулями соответственно
        self.gamma = self.create_parameter(
            attr=None,
            shape=shape,
            dtype='float32',
            is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.ones(shape=shape, dtype='float32')))
        self.beta = self.create_parameter(
            attr=None,
            shape=shape,
            dtype='float32',
            is_bias=False,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=shape, dtype='float32')))
        self.moving_mean = paddle.zeros(shape=shape, dtype='float32')
        self.moving_var = paddle.zeros(shape=shape, dtype='float32')

    def forward(self, X):
        # Сохраняем обновлённые moving_mean и moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9, is_training=self.training)
        return Y
```

## 7.5.4 Применение пакетной нормализации в LeNet

Чтобы лучше понять, как **применять `BatchNorm`**, используем её **в модели LeNet** (см. раздел 6.6). Напомним, что пакетная нормализация применяется после свёрточного или полносвязного слоя и перед соответствующей функцией активации.

**MXNet:**
```python
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

**PyTorch:**
```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

**TensorFlow:**
```python
# Напомним: эту функцию нужно передать в d2l.train_ch6.
# Иначе говоря, чтобы использовать имеющиеся CPU/GPU, модель нужно создавать внутри strategy.scope()
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

**PaddlePaddle:**
```python
net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(), nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(), 
    nn.Linear(84, 10))
```

Как и раньше, **обучим сеть на наборе Fashion-MNIST**. Этот код почти полностью совпадает с кодом, которым мы впервые обучали LeNet (см. раздел 6.6); главное отличие в том, что скорость обучения здесь намного больше.

**MXNet, PyTorch, PaddlePaddle:**
```python
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

**TensorFlow:**
```python
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

Посмотрим, какие [**параметры растяжения `gamma` и смещения `beta`**] были выучены в первом слое пакетной нормализации.

**MXNet:**
```python
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

**PyTorch:**
```python
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

**TensorFlow:**
```python
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

**PaddlePaddle:**
```python
param = net.parameters()
print('gamma:', param[2].numpy().reshape(-1))
print('beta:', param[3].numpy().reshape(-1))
```

## 7.5.5 Краткая реализация

Помимо только что определённого `BatchNorm`, можно напрямую использовать `BatchNorm`, определённую во фреймворке глубокого обучения. Код получается почти таким же, как выше.

**MXNet:**
```python
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

**PyTorch:**
```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

**TensorFlow:**
```python
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

**PaddlePaddle:**
```python
net = nn.Sequential(
    nn.Conv2D(1, 6, kernel_size=5), nn.BatchNorm2D(6, momentum=0.1), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Conv2D(6, 16, kernel_size=5), nn.BatchNorm2D(16, momentum=0.1), nn.Sigmoid(), 
    nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(), 
    nn.Linear(256, 120), nn.BatchNorm1D(120, momentum=0.1), nn.Sigmoid(), 
    nn.Linear(120, 84), nn.BatchNorm1D(84, momentum=0.1), nn.Sigmoid(), 
    nn.Linear(84, 10))
```

Теперь [**обучим модель с теми же гиперпараметрами**]. Заметим, что высокоуровневые API обычно работают значительно быстрее, поскольку их код скомпилирован в C++ или CUDA, тогда как наш пользовательский код написан на Python.

```python
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 7.5.6 Спорные моменты

Интуитивно пакетная нормализация считается средством, которое делает оптимизацию более плавной. Однако нужно тщательно различать интуицию и реальное объяснение наблюдаемых явлений. Напомним, что мы даже не знаем, почему простые нейронные сети (многослойные перцептроны и традиционные свёрточные сети) работают так эффективно. Даже с исключением и затуханием весов они остаются очень гибкими, поэтому общие результаты теории обучения не позволяют объяснить, будут ли они обобщаться на невиданные данные.

В статье о пакетной нормализации авторы, помимо описания её применения, объяснили её принцип: уменьшением *внутреннего ковариационного сдвига* (internal covariate shift).
Предполагается, что упомянутый авторами *внутренний ковариационный сдвиг* похож на описанную выше интуицию о том, что распределения значений переменных меняются в ходе обучения.
Однако у этого объяснения есть две проблемы:

1. такой сдвиг сильно отличается от строго определённого *ковариационного сдвига* (covariate shift), поэтому название употреблено некорректно;
2. это объяснение даёт лишь расплывчатую интуицию, но оставляет открытым вопрос: почему эта техника так эффективна?

Книга призвана передать интуицию, которой практики пользуются при разработке глубоких нейронных сетей. Однако важно отделять эти руководящие интуиции от установленных научных фактов. В конце концов, когда вы освоите эти методы и начнёте писать собственные научные статьи, вам понадобится ясно различать технику и интуицию.

По мере распространения пакетной нормализации объяснение через *внутренний ковариационный сдвиг* регулярно возникало в технической литературе, в частности в широкой дискуссии о том, как следует представлять результаты исследований в машинном обучении. Али Рахими (Ali Rahimi), получая премию за проверку временем (Test of Time Award) на конференции NeurIPS 2017, произнёс запоминающуюся речь. В центре его выступления был «внутренний ковариационный сдвиг», а современную практику глубокого обучения он сравнил с алхимией.
Он подробно разобрал этот пример (Lipton, Steinhardt, 2018), обозначив тревожные тенденции в машинном обучении.
Кроме того, некоторые авторы предложили другое объяснение успеха пакетной нормализации: по некоторым показателям её поведение противоположно тому, что утверждалось в исходной статье (Santurkar et al., 2018).

Однако по сравнению с тысячами похожих расплывчатых утверждений в литературе по машинному обучению внутренний ковариационный сдвиг не заслуживает особой критики. Скорее всего, он стал предметом дискуссии благодаря широкому признанию у целевой аудитории. Пакетная нормализация уже зарекомендовала себя как незаменимый метод. Она применяется почти во всех классификаторах изображений, и на неё ссылаются в научной литературе десятки тысяч раз.

## Резюме

* В процессе обучения модели пакетная нормализация использует среднее и стандартное отклонение мини-батча, постоянно корректируя промежуточные выходы нейронной сети, так что значения промежуточных выходов всех слоёв становятся более стабильными.
* Пакетная нормализация в полносвязных и свёрточных слоях используется немного по-разному.
* Слой пакетной нормализации, как и слой исключения, по-разному работает в режимах обучения и предсказания.
* У пакетной нормализации много полезных побочных эффектов, главным образом регуляризация. С другой стороны, исходная мотивация — «уменьшение внутреннего ковариационного сдвига» — не выглядит убедительным объяснением.

## Упражнения

1. Можно ли удалить параметры смещения из полносвязного или свёрточного слоя до применения пакетной нормализации? Почему?
1. Сравните скорость обучения LeNet с пакетной нормализацией и без неё.
    1. Постройте график роста точности на обучающих и тестовых данных.
    1. Насколько большой может быть скорость обучения?
1. Нужно ли выполнять пакетную нормализацию в каждом слое? Попробуйте.
1. Можно ли заменить исключение (dropout) пакетной нормализацией? Как изменится поведение модели?
1. Определите параметры `beta` и `gamma`, наблюдайте и проанализируйте результаты.
1. Посмотрите онлайн-документацию по `BatchNorm` в высокоуровневом API, чтобы узнать о других применениях пакетной нормализации.
1. Исследовательская идея: какие ещё «нормализующие» преобразования можно применить? Подойдёт ли для этого интегральное преобразование вероятностей? А оценка полноранговой ковариационной матрицы?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/1876)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/1874)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/1875)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11792)

## Литература

* Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*.
* Lipton, Z. C., & Steinhardt, J. (2018). Troubling trends in machine learning scholarship. *arXiv preprint arXiv:1807.03341*.
* Luo, P., Wang, X., Shao, W., & Peng, Z. (2018). Towards understanding regularization in batch normalization. *arXiv preprint*.
* Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How does batch normalization help optimization? *Advances in Neural Information Processing Systems*, 2483–2493.
* Teye, M., Azizpour, H., & Smith, K. (2018). Bayesian uncertainty estimation for batch normalized deep networks. *arXiv preprint arXiv:1802.06455*.
