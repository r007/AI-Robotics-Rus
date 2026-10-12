# 11.11 Планировщики скорости обучения

До сих пор мы в основном рассматривали алгоритмы оптимизации, которые определяют, как обновлять весовой вектор, но не то, с какой скоростью это делать.
Однако настройка скорости обучения часто не менее важна, чем сам алгоритм. Нужно учитывать следующие аспекты:

* Во-первых, важна величина скорости обучения. Если она слишком велика, оптимизация расходится; если слишком мала, обучение занимает слишком много времени или мы в итоге получаем лишь субоптимальный результат. Ранее мы видели, что число обусловленности задачи важно (подробности см. в разделе 11.6). Интуитивно это отношение наименьшего изменения к наибольшему по чувствительности направлениям.
* Во-вторых, важна и скорость убывания. Если скорость обучения остаётся слишком высокой, мы можем в конце концов «прыгать» вокруг минимума и не достигнуть оптимального решения. Этот вопрос подробно обсуждается в разделе 11.5, а в разделе 11.4 мы анализировали гарантии производительности. Коротко говоря, мы хотим, чтобы скорость убывала медленнее, чем $\mathcal{O}(t^{-\frac{1}{2}})$; это хороший выбор для выпуклых задач.
* Ещё один не менее важный аспект — инициализация. Это касается как первоначальной установки параметров (подробности см. в разделе 4.8), так и того, как они меняются в начале. Это шутливо называют *разогревом* (warmup): насколько быстро мы начинаем двигаться к решению. Большие шаги в начале могут быть бесполезны, особенно потому, что начальный набор параметров случаен. Направление первых обновлений тоже может быть бессмысленным.
* Наконец, существует множество вариантов оптимизации, которые выполняют периодическую корректировку скорости обучения. Это выходит за рамки этой главы; за подробностями мы рекомендуем обратиться к работе (Izmailov et al., 2018). Например, там показано, как получить лучшее решение, усредняя параметры по всему пути оптимизации.

Поскольку управление скоростью обучения требует много деталей, большинство фреймворков глубокого обучения имеют инструменты для автоматического решения этой задачи.
В этой главе мы рассмотрим, как разные стратегии расписания влияют на точность, и покажем, как эффективно управлять ими с помощью *планировщика скорости обучения* (learning rate scheduler).

## Простой пример

Начнём с простой задачи, которую легко посчитать, но которая достаточно хорошо иллюстрирует суть.
Возьмём несколько модернизированную версию LeNet (функция активации `relu` вместо `sigmoid`, а вместо average pooling — max pooling) и применим её к набору данных Fashion-MNIST.
Кроме того, мы гибридизируем сеть для повышения производительности.
Поскольку большая часть кода стандартна, мы поясним только основы и не будем подробно обсуждать детали. При необходимости обратитесь к главе 6 за повторением материала.

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Код почти такой же, как d2l.train_ch6 из раздела о LeNet в главе о свёрточных нейронных сетях
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='эпоха', xlim=[0, num_epochs],
                            legend=['потери на обучении', 'точность на обучении',
                                    'точность на тесте'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # Потери на обучении, точность на обучении, число примеров
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'потери на обучении {train_loss:.3f}, точность на обучении {train_acc:.3f}, '
          f'точность на тесте {test_acc:.3f}')
```

**PyTorch:**

```python
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Код почти такой же, как d2l.train_ch6 из раздела о LeNet в главе о свёрточных нейронных сетях
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='эпоха', xlim=[0, num_epochs],
                            legend=['потери на обучении', 'точность на обучении',
                                    'точность на тесте'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # Потери на обучении, точность на обучении, число примеров
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Встроенный планировщик PyTorch
                scheduler.step()
            else:
                # Пользовательский планировщик
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'потери на обучении {train_loss:.3f}, точность на обучении {train_acc:.3f}, '
          f'точность на тесте {test_acc:.3f}')
```

**TensorFlow:**

```python
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Код почти такой же, как d2l.train_ch6 из раздела о LeNet в главе о свёрточных нейронных сетях
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

**PaddlePaddle:**

```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn
from paddle.optimizer import lr as lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2D(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Conv2D(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2D(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Код почти такой же, как d2l.train_ch6 из раздела о LeNet в главе о свёрточных нейронных сетях
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    animator = d2l.Animator(xlabel='эпоха', xlim=[0, num_epochs],
                            legend=['потери на обучении', 'точность на обучении',
                                    'точность на тесте'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # Потери на обучении, точность на обучении, число примеров
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.clear_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with paddle.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat,y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Встроенный планировщик PaddlePaddle
                scheduler.step()
            else:
                # Пользовательский планировщик
                trainer.set_lr(scheduler(epoch))
    print(f'потери на обучении {train_loss:.3f}, точность на обучении {train_acc:.3f}, 'f'точность на тесте {test_acc:.3f}')
```

Посмотрим, что произойдёт, если использовать настройки по умолчанию.
Например, установим скорость обучения равной $0.3$ и обучим модель за $30$ итераций.
Обратите внимание, что после определённого момента прогресс точности на тесте останавливается, в то время как точность на обучении продолжает расти.
Разрыв между двумя кривыми указывает на переобучение.

**MXNet:**

```python
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

**PyTorch:**

```python
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

**TensorFlow:**

```python
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

**PaddlePaddle:**

```python
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=lr, parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

## Планировщики скорости обучения

Мы можем понижать скорость обучения после каждой эпохи (или даже после каждого минибатча).
Например, это можно делать динамически, реагируя на ход оптимизации.

**MXNet:**

```python
trainer.set_learning_rate(0.1)
print(f'скорость обучения теперь {trainer.learning_rate:.2f}')
```

**PyTorch:**

```python
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'скорость обучения теперь {trainer.param_groups[0]["lr"]:.2f}')
```

**TensorFlow:**

```python
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'скорость обучения теперь ,', dummy_model.optimizer.lr.numpy())
```

**PaddlePaddle:**

```python
lr = 0.1
trainer.set_lr(lr)
print(f'скорость обучения теперь {trainer.get_lr():.2f}')
```

В более общем случае мы должны определить планировщик.
При вызове с номером обновления он возвращает подходящее значение скорости обучения.
Определим простой метод, который задаёт скорость обучения $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```python
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Построим график его поведения на наборе значений.

```python
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Теперь посмотрим, как это влияет на обучение на наборе Fashion-MNIST.
Мы просто передаём планировщик как дополнительный параметр алгоритма обучения.

**MXNet:**

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

**PyTorch:**

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

**TensorFlow:**

```python
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

**PaddlePaddle:**

```python
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=lr , parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

Это немного лучше, чем раньше: кривые стали более гладкими, а переобучение меньше.
К сожалению, есть взгляды, объясняющие, почему некоторые стратегии в теории приводят к более слабому переобучению: меньший шаг приводит к тому, что параметры ближе к нулю и, следовательно, проще.
Однако это не полностью объясняет явление, поскольку мы не останавливали обучение раньше времени, а лишь плавно снижали скорость обучения.

## Стратегии

Хотя мы не можем рассмотреть все типы планировщиков скорости обучения, попробуем кратко описать часто используемые стратегии: полиномиальное затухание и кусочно-постоянный график.
Кроме того, косинусный планировщик хорошо работает на некоторых задачах.
А на некоторых задачах полезно прогреть оптимизатор перед использованием высокой скорости обучения.

### Планировщик с мультипликативным затуханием

Альтернативой полиномиальному затуханию является мультипликативное затухание, то есть $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$, где $\alpha \in (0, 1)$.
Чтобы скорость обучения не опускалась ниже разумного нижнего предела, уравнение обновления часто модифицируют так: $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```python
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Далее мы будем использовать встроенные планировщики, но здесь мы лишь поясняем, как они работают.

### Многоступенчатый планировщик

Одна из распространённых стратегий обучения глубоких сетей — поддерживать скорость обучения на уровне кусочно-постоянных значений и время от времени умножать её на заданный коэффициент.
В частности, задав набор моментов, в которые скорость обучения снижается, например $s = \{5, 10, 20\}$, при $t \in s$ мы уменьшаем скорость по правилу $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$.
Если предположить, что на каждом шаге значение уменьшается вдвое, то реализовать это можно так.

**MXNet:**

```python
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

**PyTorch:**

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

**TensorFlow:**

```python
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

**PaddlePaddle:**

```python
net = net_fn()
scheduler =paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[15,30], gamma=0.5)
trainer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=net.parameters())
def get_lr(trainer, scheduler):
    lr=trainer.state_dict()['LR_Scheduler']['last_lr']
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(paddle.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

Интуиция этого кусочно-постоянного расписания такова: пусть оптимизация продолжается, пока распределение весового вектора не достигнет стационарной точки.
И только тогда мы снижаем скорость обучения, чтобы получить более качественное приближение к хорошему локальному минимуму.
Пример ниже показывает, как этот подход может дать лучшее решение.

**MXNet:**

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

**PyTorch:**

```python
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

**TensorFlow:**

```python
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

**PaddlePaddle:**

```python
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

### Косинусный планировщик

Косинусный планировщик — эвристический алгоритм, предложенный в работе (Loshchilov & Hutter, 2016).
Он основан на идее, что мы, возможно, не хотим слишком сильно снижать скорость обучения в самом начале, а в конце хотим «улучшить» решение очень маленькой скоростью обучения.
В результате получается расписание, похожее на косинус; его функциональная форма показана ниже, при этом значения скорости обучения лежат на отрезке $t \in [0, T]$.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

Здесь $\eta_0$ — начальная скорость обучения, а $\eta_T$ — целевая скорость обучения при $T$.
Кроме того, при $t > T$ мы просто фиксируем значение на уровне $\eta_T$ и больше его не увеличиваем.
В следующем примере задаём максимальное число шагов обновления $T = 20$.

```python
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```python
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

В контексте компьютерного зрения такой график может дать улучшенные результаты.
Однако, как показано ниже, это улучшение не обязательно будет иметь место.

**MXNet:**

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

**PyTorch:**

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

**TensorFlow:**

```python
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

**PaddlePaddle:**

```python
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=0.3, parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

### Разогрев

В некоторых случаях инициализации параметров недостаточно для получения хорошего решения.
Это особенно проблематично для некоторых сложных архитектур сетей и может приводить к нестабильным результатам оптимизации.
С одной стороны, можно выбрать достаточно малую скорость обучения, чтобы предотвратить расходимость в самом начале; однако тогда прогресс слишком медленный.
С другой стороны, высокая скорость обучения с самого начала приводит к расходимости.

Довольно простой способ выйти из этой дилеммы — использовать период разогрева, в течение которого скорость обучения возрастает до своего начального максимума, а затем снижается до конца оптимизации.
Для простоты обычно используют линейный рост.
Так получается расписание, показанное в таблице ниже.

**MXNet:**

```python
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

**PyTorch, TensorFlow, PaddlePaddle:**

```python
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Обратите внимание: на первых 5 эпохах сеть сходится лучше.

**MXNet:**

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

**PyTorch:**

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

**TensorFlow:**

```python
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

**PaddlePaddle:**

```python
net = net_fn()
trainer = paddle.optimizer.SGD(learning_rate=0.3, parameters=net.parameters())
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

Разогрев можно применять к любому планировщику, а не только к косинусному.
Более подробные эксперименты и обсуждение расписаний скорости обучения см. в работе (Gotmare et al., 2018).
Там, в частности, показано ключевое наблюдение: фаза разогрева ограничивает степень расходимости параметров в очень глубоких сетях.
Это интуитивно понятно: в тех частях сети, которые в начале медленнее всего прогрессируют, случайная инициализация порождает сильную расходимость.

## Резюме

* Постепенное снижение скорости обучения во время обучения повышает точность и уменьшает переобучение модели.
* В экспериментах очень эффективно снижать скорость обучения, когда прогресс выходит на плато. По сути, это гарантирует, что мы эффективно сходимся к подходящему решению, и только тогда уменьшение скорости обучения позволяет снизить внутренний разброс параметров.
* Косинусный планировщик популярен для некоторых задач компьютерного зрения.
* Период разогрева перед оптимизацией может предотвратить расходимость.
* Оптимизация имеет много применений в глубоком обучении. Для одной и той же ошибки на обучении выбор разных алгоритмов оптимизации и расписаний скорости обучения может давать разную степень обобщения и переобучения на тестовом наборе, помимо минимизации времени обучения.

## Упражнения

1. Исследуйте поведение оптимизации при фиксированной скорости обучения. Какую лучшую модель можно получить в этом случае?
1. Как изменится сходимость, если менять показатель степени в убывании скорости обучения? В экспериментах для удобства используйте `PolyScheduler`.
1. Примените косинусный планировщик к большой задаче компьютерного зрения, например, к обучению на наборе ImageNet. Как он влияет на производительность по сравнению с другими планировщиками?
1. Как долго должен длиться период разогрева?
1. Можно ли связать оптимизацию с сэмплированием? В качестве первого шага используйте результаты (Welling & Teh, 2011) для стохастической градиентной динамики Ланжевена.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/4333)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/4334)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/4335)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11856)

## Литература

* Gotmare, A., Keskar, N. S., Xiong, C., & Socher, R. (2018). A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation. *arXiv preprint arXiv:1810.13243*.
* Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *arXiv preprint arXiv:1803.05407*.
* Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*.
* Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *Proceedings of the 28th International Conference on Machine Learning (ICML-11)*, 681–688.
