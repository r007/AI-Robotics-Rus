# 13.1 Аугментация изображений

В разделе 7.1 уже говорилось, что большие наборы данных — необходимое условие успешного применения глубоких нейронных сетей. *Аугментация изображений* (image augmentation) на основе обучающих изображений создаёт похожие, но разные обучающие примеры: случайно изменяет изображения и тем самым расширяет обучающий набор. Кроме того, аугментация полезна потому, что случайные изменения примеров уменьшают зависимость модели от некоторых свойств и повышают её способность к обобщению. Например, можно по-разному обрезать изображение, чтобы объект оказывался в разных местах кадра, и тем самым снизить зависимость модели от положения объекта. Можно также менять яркость, цвет и другие параметры, чтобы уменьшить чувствительность модели к цвету. Можно сказать, что методы аугментации были необходимы для успеха AlexNet. В этом разделе мы рассмотрим этот метод, широко применяемый в компьютерном зрении.

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

**PyTorch:**

```python
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

**PaddlePaddle:**

```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.vision as paddlevision
from paddle import nn
```

## 13.1.1 Распространённые методы аугментации

Для изучения методов аугментации мы будем использовать изображение размером $400\times 500$ в качестве примера.

**MXNet:**

```python
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

**PyTorch, PaddlePaddle:**

```python
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

Большинство методов аугментации содержат элемент случайности. Чтобы наглядно увидеть эффект аугментации, определим вспомогательную функцию `apply`. Она многократно применяет метод аугментации `aug` к входному изображению `img` и выводит все результаты.

```python
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### 13.1.1.1 Отражение и обрезка

**Горизонтальное отражение изображения** обычно не меняет класс объекта. Это один из самых ранних и самых широко используемых методов аугментации. Создадим экземпляр `RandomFlipLeftRight` из модуля `transforms`: с вероятностью 50% изображение отразится влево или вправо.

**MXNet:**

```python
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

**PyTorch:**

```python
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

**PaddlePaddle:**

```python
apply(img, paddlevision.transforms.RandomHorizontalFlip())
```

**Вертикальное отражение изображения** используется реже, чем горизонтальное. Однако для данного примера изображения вертикальное отражение не мешает распознаванию. Создадим экземпляр `RandomFlipTopBottom`, чтобы с вероятностью 50% изображение отразилось вверх или вниз.

**MXNet:**

```python
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

**PyTorch:**

```python
apply(img, torchvision.transforms.RandomVerticalFlip())
```

**PaddlePaddle:**

```python
apply(img,  paddlevision.transforms.RandomVerticalFlip())
```

В примере кошка находится в центре изображения, но так бывает не всегда. В разделе 6.5 мы объясняли, что слой пулинга снижает чувствительность свёрточного слоя к положению объекта. Кроме того, можно случайно обрезать изображение, чтобы объект появлялся в разных местах кадра и в разном масштабе. Это тоже снижает чувствительность модели к положению объекта.

В приведённом ниже коде **случайно обрезается** область, площадь которой составляет от 10% до 100% исходной, а отношение сторон случайно выбирается от 0,5 до 2. Затем ширина и высота области масштабируются до 200 пикселей. В этом разделе (если не оговорено иное) случайное число между $a$ и $b$ — это непрерывное значение, равномерно выбранное из отрезка $[a, b]$.

**MXNet:**

```python
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

**PyTorch:**

```python
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

**PaddlePaddle:**

```python
shape_aug =  paddlevision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 13.1.1.2 Изменение цвета

Другой способ аугментации — изменение цвета. Можно изменять четыре параметра цвета изображения: яркость, контраст, насыщенность и оттенок. В примере ниже мы **случайно меняем яркость изображения** в пределах от 50% ($1-0.5$) до 150% ($1+0.5$) от исходного значения.

**MXNet:**

```python
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

**PyTorch:**

```python
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

**PaddlePaddle:**

```python
apply(img,  paddlevision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Аналогично мы можем **случайно менять оттенок изображения**.

**MXNet:**

```python
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

**PyTorch:**

```python
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

**PaddlePaddle:**

```python
apply(img,  paddlevision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Можно также создать экземпляр `RandomColorJitter` и задать, как **случайно менять яркость (`brightness`), контраст (`contrast`), насыщенность (`saturation`) и оттенок (`hue`) изображения одновременно**.

**MXNet:**

```python
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

**PyTorch:**

```python
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

**PaddlePaddle:**

```python
color_aug =  paddlevision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### 13.1.1.3 Комбинирование нескольких методов аугментации

На практике мы комбинируем несколько методов аугментации. Например, можно объединить описанные выше методы с помощью экземпляра `Compose` и применять их к каждому изображению.

**MXNet:**

```python
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

**PyTorch:**

```python
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

**PaddlePaddle:**

```python
augs =  paddlevision.transforms.Compose([
     paddle.vision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## 13.1.2 Обучение с использованием аугментации изображений

Теперь обучим модель с аугментацией. Здесь мы используем набор данных CIFAR-10, а не Fashion-MNIST, который применяли раньше. Причина в том, что в Fashion-MNIST положение и размер объектов уже нормализованы, а в CIFAR-10 различия в цвете и размере объектов выражены сильнее. Первые 32 обучающих изображения CIFAR-10 показаны ниже.

**MXNet:**

```python
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

**PyTorch:**

```python
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

**PaddlePaddle:**

```python
all_images =  paddlevision.datasets.Cifar10(mode='train' , download=True)
print(len(all_images))
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Чтобы результат предсказания был однозначным, аугментацию обычно применяют только к обучающим примерам, а при предсказании случайные методы аугментации не используют. Здесь мы используем только самое простое — случайное горизонтальное отражение. Кроме того, экземпляр `ToTensor` преобразует пакет изображений в формат, который требуется фреймворку глубокого обучения: 32-битные числа с плавающей точкой, форма (размер батча, число каналов, высота, ширина), значения от 0 до 1.

**MXNet:**

```python
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

**PyTorch:**

```python
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

**PaddlePaddle:**

```python
train_augs = paddlevision.transforms.Compose([
     paddlevision.transforms.RandomHorizontalFlip(),
     paddlevision.transforms.ToTensor()])

test_augs = paddlevision.transforms.Compose([
     paddlevision.transforms.ToTensor()])
```

**MXNet:** Далее определим вспомогательную функцию, которая читает изображения и применяет аугментацию. Функция `transform_first` из набора данных Gluon применяет аугментацию к первому элементу каждого обучающего примера (то есть к изображению, а не к метке). Подробнее о `DataLoader` см. в разделе 3.5.

**PyTorch:** Далее **определим вспомогательную функцию, которая читает изображения и применяет аугментацию**. Параметр `transform` набора данных PyTorch применяет аугментацию к изображениям. Подробнее о `DataLoader` см. в разделе 3.5.

**MXNet:**

```python
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

**PyTorch:**

```python
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

**PaddlePaddle:**

```python
def load_cifar10(is_train, augs, batch_size):
    dataset = paddlevision.datasets.Cifar10(mode="train", 
                                            transform=augs, download=True)
    dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size, 
                    num_workers=d2l.get_dataloader_workers(), shuffle=is_train)
    return dataloader
```

### 13.1.2.1 Обучение на нескольких GPU

Мы обучим модель ResNet-18 из раздела 7.6 на CIFAR-10. Напомним, что обучение на нескольких GPU описано в разделе 12.6. Далее **определим функцию, которая обучает и оценивает модель на нескольких GPU**.

**MXNet:**

```python
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Обучение на минибатче с использованием нескольких GPU"""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # Флаг True разрешает использовать устаревшие градиенты, это полезно (например, при дообучении BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

**PyTorch:**

```python
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Обучение на минибатче с использованием нескольких GPU"""
    if isinstance(X, list):
        # Нужно для дообучения BERT
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

**PaddlePaddle:**

```python
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Обучение на минибатче с использованием нескольких GPU.
    PaddlePaddle не поддерживает обучение на нескольких GPU в notebook"""
    if isinstance(X, list):
        # Нужно для дообучения BERT (см. далее)
        X = [paddle.to_tensor(x, place=devices[0]) for x in X]
    else:
        X = paddle.to_tensor(X, place=devices[0])
    y = paddle.to_tensor(y, place=devices[0])
    net.train()
    trainer.clear_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

**MXNet:**

```python
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Обучение модели с использованием нескольких GPU"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['потери (обучение)', 'точность (обучение)',
                                    'точность (тест)'])
    for epoch in range(num_epochs):
        # 4 величины: сумма потерь на обучении, точность на обучении, число примеров, число элементов
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'потери {metric[0] / metric[2]:.3f}, точность на обучении '
          f'{metric[1] / metric[3]:.3f}, точность на тесте {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} примеров/с на '
          f'{str(devices)}')
```

**PyTorch:**

```python
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Обучение модели с использованием нескольких GPU"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['потери (обучение)', 'точность (обучение)',
                                    'точность (тест)'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4 величины: сумма потерь на обучении, точность на обучении, число примеров, число элементов
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'потери {metric[0] / metric[2]:.3f}, точность на обучении '
          f'{metric[1] / metric[3]:.3f}, точность на тесте {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} примеров/с на '
          f'{str(devices)}')
```

**PaddlePaddle:**

```python
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Обучение модели с использованием нескольких GPU"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['потери (обучение)', 'точность (обучение)',
                                    'точность (тест)'])
    net = paddle.DataParallel(net)
    for epoch in range(num_epochs):
        # 4 величины: сумма потерь на обучении, точность на обучении, число примеров, число элементов
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'потери {metric[0] / metric[2]:.3f}, точность на обучении '
          f'{metric[1] / metric[3]:.3f}, точность на тесте {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} примеров/с на '
          f'{str(devices)}')
```

Теперь **определим функцию `train_with_data_aug`, которая обучает модель с аугментацией изображений**. Она использует все доступные GPU, оптимизирует модель алгоритмом Adam, применяет аугментацию к обучающему набору и в конце вызывает определённую выше функцию `train_ch13` для обучения и оценки модели.

**MXNet:**

```python
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

**PyTorch:**

```python
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

**PaddlePaddle:**

```python
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2D]:
        nn.initializer.XavierUniform(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices[:1])
```

Теперь **обучим модель**, используя аугментацию со случайным горизонтальным отражением.

```python
train_with_data_aug(train_augs, test_augs, net)
```

## Резюме

* Аугментация изображений создаёт случайные изображения на основе существующих обучающих данных и повышает способность модели к обобщению.
* Чтобы результат предсказания был однозначным, аугментацию обычно применяют только к обучающим примерам, а при предсказании случайные методы аугментации не используют.
* Фреймворки глубокого обучения предлагают множество методов аугментации изображений, которые можно применять одновременно.

## Упражнения

1. Обучите модель без аугментации: `train_with_data_aug(no_aug, no_aug)`. Сравните результаты обучения и точность на тесте при использовании аугментации и без неё. Подтверждает ли этот эксперимент, что аугментация изображений уменьшает переобучение? Почему?
2. Обучите модель на CIFAR-10, комбинируя несколько разных методов аугментации. Повышает ли это точность на тесте?
3. Изучите онлайн-документацию фреймворка глубокого обучения. Какие ещё методы аугментации изображений в нём есть?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2828)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2829)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11801)
