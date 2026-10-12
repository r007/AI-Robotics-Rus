# 12.6 Краткая реализация обучения на нескольких GPU

Реализовывать параллельные вычисления для каждой новой модели с нуля — неинтересно. Кроме того, полезно оптимизировать инструменты синхронизации, чтобы добиться высокой производительности. Ниже мы покажем, как сделать это с помощью высокоуровневого API фреймворков глубокого обучения. Математика и алгоритмы такие же, как в разделе 12.5. Код этого раздела запускается как минимум на двух GPU.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
from torch import nn
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## Простая сеть

Возьмём сеть, более содержательную, чем LeNet из раздела 12.5, которую при этом можно обучать быстро и легко. Мы выбираем ResNet-18 из работы :cite:`He.Zhang.Ren.ea.2016`. Поскольку входные изображения очень маленькие, мы немного её изменим: по сравнению с разделом 7.6 в начале сети используются меньшие ядра свёртки, шаги и паддинг, а слой максимального пулинга удалён.

**MXNet:**

```python
#@save
def resnet18(num_classes):
    """ResNet-18 с небольшими изменениями"""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # В этой модели использованы меньшие ядра свёртки, шаги и паддинг, а также удалён слой максимального пулинга
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

**PyTorch:**

```python
#@save
def resnet18(num_classes, in_channels=1):
    """ResNet-18 с небольшими изменениями"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # В этой модели использованы меньшие ядра свёртки, шаги и паддинг, а также удалён слой максимального пулинга
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

**PaddlePaddle:**

```python
#@save
def resnet18(num_classes, in_channels=1):
    """ResNet-18 с небольшими изменениями"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # В этой модели использованы меньшие ядра свёртки, шаги и паддинг, а также удалён слой максимального пулинга
    net = nn.Sequential(
        nn.Conv2D(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2D(64),
        nn.ReLU())
    net.add_sublayer("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_sublayer("resnet_block2", resnet_block(64, 128, 2))
    net.add_sublayer("resnet_block3", resnet_block(128, 256, 2))
    net.add_sublayer("resnet_block4", resnet_block(256, 512, 2))
    net.add_sublayer("global_avg_pool", nn.AdaptiveAvgPool2D((1, 1)))
    net.add_sublayer("fc", nn.Sequential(nn.Flatten(),
                                         nn.Linear(512, num_classes)))
    return net
```

## Инициализация сети

**MXNet:** Функция `initialize` позволяет инициализировать параметры на выбранных устройствах. Методы инициализации мы повторяли в разделе 4.8. Эта функция особенно удобна, когда сеть инициализируется сразу на нескольких устройствах. Посмотрим, как она работает на практике.

**PyTorch:** Сеть мы инициализируем в цикле обучения. Методы инициализации см. в разделе 4.8.

**MXNet:**

```python
net = resnet18(10)
# Получаем список GPU
devices = d2l.try_all_gpus()
# Инициализируем все параметры сети
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

**PyTorch:**

```python
net = resnet18(10)
# Получаем список GPU
devices = d2l.try_all_gpus()
# Сеть инициализируем в коде обучения
```

**PaddlePaddle:**

```python
net = resnet18(10)
# Получаем список GPU
devices = d2l.try_all_gpus()
# Сеть инициализируем в коде обучения
```

**MXNet:** Функция `split_and_load` из раздела 12.5 разбивает мини-батч данных и копирует части на устройства из списка `devices`. Сеть автоматически использует подходящий GPU, чтобы вычислить прямое распространение. Ниже мы сгенерируем $4$ наблюдения и разобьём их между GPU.

```python
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

**MXNet:** Как только данные проходят через сеть, соответствующие параметры сети инициализируются *на устройствах, через которые прошли данные*. Это означает, что инициализация выполняется отдельно на каждом устройстве. Поскольку мы выбрали GPU0 и GPU1, сеть инициализируется только на этих двух GPU, а не на CPU. Действительно, на CPU этих параметров даже нет. Проверим это, выведя параметры и посмотрев, не возникнет ли ошибка.

```python
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

**MXNet:** Далее заменим код [**оценки модели**] кодом, который [**работает параллельно на нескольких устройствах**]. Это замена функции `evaluate_accuracy_gpu` из раздела 6.6. Основное отличие в том, что перед вызовом сети мини-батч разбивается, в остальном код по сути тот же.

```python
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Вычисляет точность модели на наборе данных с помощью нескольких GPU"""
    # Получаем список устройств
    devices = list(net.collect_params().values())[0].list_ctx()
    # Число верных предсказаний и общее число предсказаний
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Параллельный запуск
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Обучение**]

Как и раньше, код обучения должен выполнять несколько базовых функций, чтобы параллелизм был эффективным:

* нужно инициализировать параметры сети на всех устройствах;
* при итерации по набору данных мини-батчи нужно распределять по всем устройствам;
* нужно параллельно вычислять потери и их градиенты на всех устройствах;
* нужно агрегировать градиенты и соответствующим образом обновлять параметры.

Наконец, точность вычисляется параллельно, и выводится итоговая производительность сети. Кроме разбиения и агрегации данных, код обучения очень похож на реализации из предыдущих глав.

**MXNet:**

```python
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('эпоха', 'точность (тест)', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'Точность на тесте: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} с/эпоху, '
          f'на {str(ctx)}')
```

**PyTorch:**

```python
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # Настраиваем модель на нескольких GPU
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('эпоха', 'точность (тест)', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'Точность на тесте: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} с/эпоху, '
          f'на {str(devices)}')
```

**PaddlePaddle:**

```python
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    
    init_normal = nn.initializer.Normal(mean=0.0, std=0.01)
    for i in net.sublayers():
        if type(i) in [nn.Linear, nn.Conv2D]:        
            init_normal(i.weight)

    # Настраиваем модель на нескольких GPU
    net = paddle.DataParallel(net)
    trainer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('эпоха', 'точность (тест)', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.clear_grad()
            X, y = paddle.to_tensor(X, place=devices[0]), paddle.to_tensor(y, place=devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'Точность на тесте: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} с/эпоху, '
          f'на {str(devices)}')
```

Посмотрим, как это работает на практике. Сначала [**обучим сеть на одном GPU**] в качестве разминки.

**MXNet:**

```python
train(num_gpus=1, batch_size=256, lr=0.1)
```

**PyTorch, PaddlePaddle:**

```python
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Далее [**обучим сеть на 2 GPU**]. По сравнению с LeNet из раздела 12.5 модель ResNet-18 намного сложнее. Здесь заметно проявляется преимущество параллелизма: время вычислений существенно превышает время синхронизации параметров. Поскольку накладные расходы на параллелизм относительно невелики, такой подход повышает масштабируемость модели.

**MXNet:**

```python
train(num_gpus=2, batch_size=512, lr=0.2)
```

**PyTorch:**

```python
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Резюме

**MXNet:**

* Gluon предоставляет примитивы инициализации модели на нескольких устройствах, принимая список контекстов.
* Нейронную сеть можно автоматически вычислять на одном GPU (там, где находятся данные).
* Сеть на каждом устройстве нужно сначала инициализировать, прежде чем обращаться к параметрам этого устройства, иначе возникнет ошибка.
* Алгоритмы оптимизации автоматически агрегируют данные с нескольких GPU.

**PyTorch, PaddlePaddle:**

* Нейронную сеть можно автоматически вычислять на одном GPU (там, где находятся данные).
* Сеть на каждом устройстве нужно сначала инициализировать, прежде чем обращаться к параметрам этого устройства, иначе возникнет ошибка.
* Алгоритмы оптимизации автоматически агрегируют данные с нескольких GPU.

## Упражнения

**MXNet:**

1. В этом разделе используется ResNet-18. Попробуйте разное количество эпох, размеры мини-батчей и скорости обучения, а также больше GPU для вычислений. Что произойдёт, если попробовать это на $16$ GPU (например, на экземпляре AWS p2.16xlarge)?
1. Иногда разные устройства имеют разную вычислительную мощность, и мы можем использовать одновременно GPU и CPU. Как тогда распределять работу? Почему?
1. Что будет, если убрать `npx.waitall()`? Как изменить обучение, чтобы параллельные операции перекрывались не более чем в два шага?

**PyTorch, PaddlePaddle:**

1. В этом разделе используется ResNet-18. Попробуйте разное количество эпох, размеры мини-батчей и скорости обучения, а также больше GPU для вычислений. Что произойдёт, если попробовать это на $16$ GPU (например, на экземпляре AWS p2.16xlarge)?
1. Иногда разные устройства имеют разную вычислительную мощность, и мы можем использовать одновременно GPU и CPU. Как тогда распределять работу? Почему?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2804)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2803)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11861)
