# 13.13 Практика: соревнование Kaggle по классификации изображений (CIFAR-10)

В предыдущих разделах мы получали наборы изображений сразу в виде тензоров с помощью высокоуровневого API фреймворков. Однако на практике изображения обычно хранятся в виде файлов. В этом разделе мы начнём с исходных файлов изображений, а затем шаг за шагом организуем данные, прочитаем их и преобразуем в формат тензоров.

В разделе 13.1 мы проводили эксперимент с набором данных CIFAR-10, который является важным набором в области компьютерного зрения. В этом разделе мы применим знания из предыдущих разделов и примем участие в соревновании Kaggle по классификации изображений CIFAR-10, **адрес соревнования: https://www.kaggle.com/c/cifar-10**.

На рис. 13.13.1 показана страница соревнования с описанием. Чтобы отправить результаты, сначала нужно зарегистрировать аккаунт на Kaggle.

> *Рис. 13.13.1. Страница соревнования по классификации изображений CIFAR-10 с описанием. Данные для соревнования доступны на вкладке «Data». Скриншот страницы стороннего сайта в перевод не включён.*

Сначала импортируем пакеты и модули, которые нужны для соревнования.

**MXNet:**

```python
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

**PyTorch:**

```python
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import collections
import math
import os
import pandas as pd
import shutil
import paddle
from paddle import nn
import paddle.vision as paddlevision
```

## Получение и организация набора данных

Набор данных соревнования разделён на обучающую и тестовую выборки: в обучающей 50 000 изображений, в тестовой — 300 000. Из тестовой выборки для оценки используются 10 000 изображений, а остальные 290 000 не оцениваются и включены лишь для того, чтобы нельзя было вручную разметить тестовую выборку и отправить разметку. Изображения в обоих наборах имеют формат PNG, высоту и ширину 32 пикселя и три цветовых канала (RGB). Эти изображения относятся к 10 классам: самолёт, автомобиль, птица, кошка, олень, собака, лягушка, лошадь, корабль и грузовик. В левом верхнем углу рисунка 13.13.1 показаны некоторые изображения самолётов, автомобилей и птиц из набора.

### Загрузка набора данных

После входа в Kaggle можно открыть вкладку «Data» на странице соревнования CIFAR-10 (рис. 13.13.1) и нажать кнопку «Download All», чтобы загрузить набор данных. После распаковки загруженного файла в каталоге `../data` и распаковки в нём архивов `train.7z` и `test.7z` весь набор данных будет доступен по следующим путям:

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

Папки `train` и `test` содержат обучающие и тестовые изображения соответственно, файл `trainLabels.csv` содержит метки обучающих изображений, а `sample_submission.csv` — образец файла для отправки.

Чтобы было проще начать, **мы подготовили небольшую версию набора: первые 1000 обучающих изображений и 5 случайных тестовых изображений**. Чтобы использовать полный набор данных соревнования, нужно установить переменную `demo` равной `False`.

```python
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# Если используется полный набор данных соревнования Kaggle, установите demo в False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### **Организация набора данных**

Чтобы обучать и тестировать модель, нужно организовать набор данных. Сначала с помощью функции ниже прочитаем метки из CSV-файла. Она возвращает словарь, который сопоставляет имя файла без расширения с его меткой.

```python
#@save
def read_csv_labels(fname):
    """Читает fname и возвращает словарь, отображающий имя файла в метку"""
    with open(fname, 'r') as f:
        # Пропускаем строку заголовка (названия столбцов)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# обучающих примеров :', len(labels))
print('# классов :', len(set(labels.values())))
```

Далее определим функцию `reorg_train_valid`, которая **выделяет проверочную выборку из исходной обучающей выборки**. Параметр `valid_ratio` этой функции — отношение числа примеров проверочной выборки к числу примеров исходной обучающей выборки. Точнее, пусть $n$ равно числу изображений в наименее представленном классе, а $r$ — это отношение. Тогда для каждого класса в проверочную выборку попадает $\max(\lfloor nr\rfloor,1)$ изображений. Возьмём `valid_ratio=0.1`: поскольку исходная обучающая выборка содержит 50 000 изображений, в каталоге `train_valid_test/train` будет 45 000 изображений для обучения, а оставшиеся 5 000 изображений станут проверочной выборкой в каталоге `train_valid_test/valid`. После организации данных изображения одного класса будут лежать в одной папке.

```python
#@save
def copyfile(filename, target_dir):
    """Копирует файл в целевой каталог"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Выделяет проверочную выборку из исходной обучающей выборки"""
    # Число примеров в наименее представленном классе обучающей выборки
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # Число примеров каждого класса в проверочной выборке
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

Функция `reorg_test` ниже **организует тестовую выборку на этапе предсказания так, чтобы её было удобно читать**.

```python
#@save
def reorg_test(data_dir):
    """Организует тестовую выборку на этапе предсказания так, чтобы её было удобно читать"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

Наконец, используем функцию, которая **вызывает определённые выше функции** `read_csv_labels`, `reorg_train_valid` и `reorg_test`.

```python
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

Здесь для примера размер батча выборки мы задаём равным 32. При реальном обучении и тестировании следует использовать полный набор данных соревнования и установить `batch_size` в большее число, например 128. Мы будем использовать 10% обучающих примеров как проверочную выборку для подбора гиперпараметров.

```python
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## **Аугментация изображений**

Чтобы бороться с переобучением, используем аугментацию изображений. Например, в обучении мы можем случайно отражать изображения по горизонтали. Кроме того, для трёх RGB-каналов цветного изображения можно выполнить стандартизацию. Ниже перечислены некоторые из операций, которые можно настраивать.

**MXNet:**

```python
transform_train = gluon.data.vision.transforms.Compose([
    # Увеличиваем изображение по высоте и ширине до квадрата 40 пикселей
    gluon.data.vision.transforms.Resize(40),
    # Случайно вырезаем квадратное изображение высотой и шириной 40 пикселей:
    # площадь вырезанного квадрата составляет от 0.64 до 1 площади исходного изображения,
    # затем масштабируем его до квадрата высотой и шириной 32 пикселя
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # Стандартизируем каждый канал изображения
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

**PyTorch:**

```python
transform_train = torchvision.transforms.Compose([
    # Увеличиваем изображение по высоте и ширине до квадрата 40 пикселей
    torchvision.transforms.Resize(40),
    # Случайно вырезаем квадратное изображение высотой и шириной 40 пикселей:
    # площадь вырезанного квадрата составляет от 0.64 до 1 площади исходного изображения,
    # затем масштабируем его до квадрата высотой и шириной 32 пикселя
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # Стандартизируем каждый канал изображения
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

**PaddlePaddle:**

```python
transform_train = paddlevision.transforms.Compose([
    # Увеличиваем изображение по высоте и ширине до квадрата 40 пикселей
    paddlevision.transforms.Resize(40),
    # Случайно вырезаем квадратное изображение высотой и шириной 40 пикселей:
    # площадь вырезанного квадрата составляет от 0.64 до 1 площади исходного изображения,
    # затем масштабируем его до квадрата высотой и шириной 32 пикселя
    paddlevision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                              ratio=(1.0, 1.0)),
    paddlevision.transforms.RandomHorizontalFlip(),
    paddlevision.transforms.ToTensor(),
    # Стандартизируем каждый канал изображения
    paddlevision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

На этапе тестирования мы только стандартизируем изображения, чтобы исключить случайность в оценке результатов.

**MXNet:**

```python
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

**PyTorch:**

```python
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

**PaddlePaddle:**

```python
transform_test = paddlevision.transforms.Compose([
    paddlevision.transforms.ToTensor(),
    paddlevision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## Чтение набора данных

Далее **прочитаем набор данных из исходных изображений**: каждый пример состоит из изображения и метки.

**MXNet:**

```python
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

**PyTorch:**

```python
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

**PaddlePaddle:**

```python
train_ds, train_valid_ds = [paddlevision.datasets.DatasetFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]
    
valid_ds, test_ds = [paddlevision.datasets.DatasetFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

На этапе обучения нужно **указать все описанные выше операции аугментации**. Когда проверочная выборка используется для оценки модели при подборе гиперпараметров, случайность аугментации не должна вноситься. Перед окончательным предсказанием мы обучаем модель на объединении обучающей и проверочной выборок, чтобы использовать все размеченные данные.

**MXNet:**

```python
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

**PyTorch:**

```python
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

**PaddlePaddle:**

```python
train_iter, train_valid_iter = [paddle.io.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = paddle.io.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                                  drop_last=True)

test_iter = paddle.io.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 drop_last=False)
```

## Определение **модели**

**MXNet:** Здесь остаточные блоки строятся на основе класса `HybridBlock`, что немного отличается от реализации из раздела 7.6, но повышает вычислительную эффективность.

```python
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

**MXNet:** Далее определим модель ResNet-18.

```python
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

**MXNet:** Перед началом обучения используем инициализацию Xavier, описанную в разделе 4.8.

```python
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

**PyTorch:** Мы определяем модель ResNet-18, описанную в разделе 7.6.

```python
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

**PaddlePaddle:** Мы определяем модель ResNet-18, описанную в разделе 7.6.

```python
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## Определение **функции обучения**

Мы будем выбирать модель и подбирать гиперпараметры по её качеству на проверочной выборке. Ниже определена функция обучения модели `train`.

**MXNet:**

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['потери (обучение)', 'точность (обучение)']
    if valid_iter is not None:
        legend.append('точность (валидация)')
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'потери (обучение) {metric[0] / metric[2]:.3f}, '
                f'точность (обучение) {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', точность (валидация) {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' примеров/с на {str(devices)}')
```

**PyTorch:**

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['потери (обучение)', 'точность (обучение)']
    if valid_iter is not None:
        legend.append('точность (валидация)')
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'потери (обучение) {metric[0] / metric[2]:.3f}, '
                f'точность (обучение) {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', точность (валидация) {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' примеров/с на {str(devices)}')
```

**PaddlePaddle:**

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    scheduler = paddle.optimizer.lr.StepDecay(lr, lr_period, lr_decay)
    trainer = paddle.optimizer.Momentum(learning_rate=scheduler, momentum=0.9, parameters=net.parameters(),
                              weight_decay=wd)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['потери (обучение)', 'точность (обучение)']
    if valid_iter is not None:
        legend.append('точность (валидация)')
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs],
                            legend=legend)
    net = paddle.DataParallel(net)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'потери (обучение) {metric[0] / metric[2]:.3f}, '
                f'точность (обучение) {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', точность (валидация) {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' примеров/с на {str(devices)}')
```

## **Обучение и проверка модели**

Теперь можно обучить и проверить модель; все перечисленные ниже гиперпараметры можно подбирать. Например, можно увеличить число эпох. Если `lr_period` и `lr_decay` равны 4 и 0.9, то скорость обучения оптимизации будет умножаться на 0.9 каждые 4 эпохи. Для наглядности здесь мы обучаем модель только 20 эпох.

**MXNet:**

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

**PyTorch:**

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

**PaddlePaddle:**

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## **Классификация тестовой выборки и отправка результатов на Kaggle**

Получив удовлетворительную модель с подобранными гиперпараметрами, мы переобучаем модель на всех размеченных данных (включая проверочную выборку) и классифицируем тестовую выборку.

**MXNet:**

```python
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

**PyTorch:**

```python
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

**PaddlePaddle:**

```python
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X)
    preds.extend(y_hat.argmax(axis=1).astype(paddle.int32).numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

Отправка результатов на Kaggle выполняется так же, как в разделе 4.10. Код выше создаёт файл `submission.csv` в формате, который требуется соревнованием Kaggle.

## Резюме

* После организации набора данных с исходными файлами изображений в нужном формате его можно прочитать.

**MXNet:**

* В соревновании по классификации изображений можно использовать сверточные нейронные сети, аугментацию изображений и гибридное программирование.

**PyTorch, PaddlePaddle:**

* В соревновании по классификации изображений можно использовать сверточные нейронные сети и аугментацию изображений.

## Упражнения

1. Используйте в этом соревновании Kaggle полный набор данных CIFAR-10. Задайте гиперпараметры `batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50`, `lr_decay = 0.1`. Какой точности и какого места в рейтинге можно добиться? Можно ли улучшить результат?
1. Какую точность можно получить без аугментации изображений?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2830)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2831)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11814)
