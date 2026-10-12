# 13.14 Практика: соревнование Kaggle по распознаванию пород собак (ImageNet Dogs)

В этом разделе мы решим задачу распознавания пород собак на Kaggle. **Адрес соревнования: https://www.kaggle.com/c/dog-breed-identification**. Страница соревнования с описанием показана на рис. 13.14.1. Чтобы отправить результаты, нужен аккаунт Kaggle.

В этом соревновании нужно распознать 120 пород собак. Этот набор данных на самом деле является подмножеством известного набора ImageNet. В отличие от изображений из набора CIFAR-10 из раздела 13.13, изображения ImageNet больше по высоте и ширине и имеют разные размеры.

> **Рис. 13.14.1.** Страница соревнования по распознаванию пород собак. Данные соревнования доступны на вкладке «Data». Скриншот страницы стороннего сайта в перевод не включён

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.vision as paddlevision
from paddle import nn
import os
```

## 13.14.1 Получение и организация набора данных

Наборы данных соревнования разделены на обучающий и тестовый: в них содержится соответственно 10 222 и 10 357 JPEG-изображений в цветовом пространстве RGB. В обучающем наборе 120 пород собак, среди которых лабрадор, пудель, такса, самоед, хаски, чихуахуа и йоркширский терьер.

### 13.14.1.1 Загрузка набора данных

После входа в Kaggle можно открыть вкладку «Data» на странице соревнования (см. рис. 13.14.1), нажать кнопку «Download All» и загрузить набор данных. После распаковки загруженного файла в каталоге `../data` весь набор данных будет доступен по следующим путям:

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

Структура, как и в наборе CIFAR-10 из раздела 13.13, такая: папки `train/` и `test/` содержат обучающие и тестовые изображения собак соответственно, а файл `labels.csv` содержит метки обучающих изображений.

Чтобы было проще начать, мы подготовили **небольшую версию полного набора данных**: `train_valid_test_tiny.zip`. Чтобы использовать полный набор данных соревнования Kaggle, нужно изменить переменную `demo` ниже на `False`.

```python
#@save 
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# Если используется полный набор данных соревнования Kaggle, поменяйте значение переменной на False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### 13.14.1.2 Организация набора данных

Набор данных можно организовать так же, как в разделе 13.13: выделить проверочный набор из исходного обучающего набора и переместить изображения в подпапки, сгруппированные по меткам.

Функция `reorg_dog_data` ниже читает метки обучающих данных, выделяет проверочный набор и организует обучающий набор.

```python
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## 13.14.2 Аугментация изображений

Напомним, что набор данных о породах собак — это подмножество ImageNet, а изображения в нём крупнее, чем в CIFAR-10 из раздела 13.13. Ниже мы посмотрим, как применять аугментацию к относительно большим изображениям.

**MXNet:**

```python
transform_train = gluon.data.vision.transforms.Compose([
    # Случайно вырезаем часть изображения площадью от 0.08 до 1 площади исходного изображения
    # с отношением сторон от 3/4 до 4/3. Затем масштабируем изображение до нового размера 224x224
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Случайно меняем яркость, контраст и насыщенность
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Добавляем случайный шум
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Стандартизируем каждый канал изображения
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

**PyTorch:**

```python
transform_train = torchvision.transforms.Compose([
    # Случайно вырезаем часть изображения площадью от 0.08 до 1 площади исходного изображения
    # с отношением сторон от 3/4 до 4/3. Затем масштабируем изображение до нового размера 224x224
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Случайно меняем яркость, контраст и насыщенность
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Добавляем случайный шум
    torchvision.transforms.ToTensor(),
    # Стандартизируем каждый канал изображения
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

**PaddlePaddle:**

```python
transform_train = paddlevision.transforms.Compose([
    # Случайно вырезаем часть изображения площадью от 0.08 до 1 площади исходного изображения
    # с отношением сторон от 3/4 до 4/3. Затем масштабируем изображение до нового размера 224x224
    paddlevision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    paddlevision.transforms.RandomHorizontalFlip(),
    # Случайно меняем яркость, контраст и насыщенность
    paddlevision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Добавляем случайный шум
    paddlevision.transforms.ToTensor(),
    # Стандартизируем каждый канал изображения
    paddlevision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

На этапе тестирования мы используем только детерминированные операции предобработки.

**MXNet:**

```python
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Вырезаем из центра изображения область 224x224
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

**PyTorch:**

```python
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Вырезаем из центра изображения область 224x224
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

**PaddlePaddle:**

```python
transform_test = paddlevision.transforms.Compose([
    paddlevision.transforms.Resize(256),
    # Вырезаем из центра изображения область 224x224
    paddlevision.transforms.CenterCrop(224),
    paddlevision.transforms.ToTensor(),
    paddlevision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## 13.14.3 Чтение набора данных

Как и в разделе 13.13, мы читаем организованный набор данных, состоящий из исходных файлов изображений.

**MXNet:**

```python
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

Создать экземпляры загрузчиков данных можно так же, как в разделе 13.13.

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

## 13.14.4 Дообучение предобученной модели

Набор данных этого соревнования также является подмножеством ImageNet. Поэтому мы можем использовать методы из раздела 13.2: выбрать модель, предобученную на полном ImageNet, извлекать с её помощью признаки изображений и подавать их на небольшую пользовательскую выходную сеть. Фреймворки предоставляют высокоуровневый API с различными моделями, предобученными на ImageNet. Здесь мы выбираем предобученную модель ResNet-34 и используем повторно только признаки, которые она выдаёт перед выходным слоем. Затем исходный выходной слой мы заменяем небольшой обучаемой пользовательской выходной сетью, например из двух полносвязных слоёв. В отличие от экспериментов из раздела 13.2, предобученную модель для извлечения признаков мы не дообучаем. Это экономит время на градиентный спуск и память.

Напомним, что для нормализации полного набора ImageNet мы используем средние значения и стандартные отклонения трёх RGB-каналов. Это как раз и соответствует стандартизации, которую применяют к моделям, предобученным на ImageNet.

**MXNet:**

```python
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Определяем новую выходную сеть
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # Всего 120 классов на выходе
    finetune_net.output_new.add(nn.Dense(120))
    # Инициализируем выходную сеть
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Распределяем параметры модели по CPU или GPU, которые используются для вычислений
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

**PyTorch:**

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Определяем новую выходную сеть из 120 выходных классов
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Распределяем параметры модели по CPU или GPU, которые используются для вычислений
    finetune_net = finetune_net.to(devices[0])
    # Замораживаем параметры
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

**PaddlePaddle:**

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = paddlevision.models.resnet34(pretrained=True)
    # Определяем новую выходную сеть из 120 выходных классов
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Замораживаем параметры
    for param in finetune_net.features.parameters():
        param.stop_gradient = True
    return finetune_net
```

Перед **вычислением потерь** сначала получим вход выходного слоя предобученной модели, то есть извлечённые признаки. Затем эти признаки подаём на вход нашей небольшой пользовательской выходной сети и вычисляем потери.

**MXNet:**

```python
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

**PyTorch:**

```python
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')
```

**PaddlePaddle:**

```python
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## 13.14.5 Определение функции обучения

Модель мы выбираем и подбираем гиперпараметры по качеству на проверочном наборе. Функция обучения `train` итеративно настраивает только параметры небольшой пользовательской выходной сети.

**MXNet:**

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Обучаем только небольшую пользовательскую выходную сеть
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['потери (обучение)']
    if valid_iter is not None:
        legend.append('потери (валидация)')
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'потери (обучение) {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', потери (валидация) {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' примеров/с на {str(devices)}')
```

**PyTorch:**

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Обучаем только небольшую пользовательскую выходную сеть
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['потери (обучение)']
    if valid_iter is not None:
        legend.append('потери (валидация)')
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             (metric[0] / metric[1], None))
        measures = f'потери (обучение) {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', потери (валидация) {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' примеров/с на {str(devices)}')
```

**PaddlePaddle:**

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Обучаем только небольшую пользовательскую выходную сеть
    net = paddle.DataParallel(net)
    scheduler = paddle.optimizer.lr.StepDecay(lr, lr_period, lr_decay)
    trainer = paddle.optimizer.Momentum(learning_rate=scheduler, 
                                        parameters=(param for param in net.parameters() if not param.stop_gradient), 
                                        momentum=0.9, 
                                        weight_decay=wd)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['потери (обучение)']
    if valid_iter is not None:
        legend.append('потери (валидация)')
    animator = d2l.Animator(xlabel='эпоха', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            trainer.clear_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, 
                             (metric[0] / metric[1], None))
        measures = f'потери (обучение) {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', потери (валидация) {float(valid_loss):.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' примеров/с на {str(devices)}')
```

## 13.14.6 Обучение и проверка модели

Теперь можно обучить и проверить модель; все перечисленные ниже гиперпараметры можно настраивать. Например, можно увеличить число эпох. Поскольку `lr_period` и `lr_decay` равны 2 и 0.9, скорость обучения оптимизатора умножается на 0.9 каждые 2 эпохи.

**MXNet:**

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

**PyTorch:**

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

**PaddlePaddle:**

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## 13.14.7 Классификация тестового набора и отправка результатов на Kaggle

Как и в последнем шаге раздела 13.13, все размеченные данные, включая проверочный набор, используются для обучения модели, а затем модель классифицирует тестовый набор. Классификацию выполняет обученная пользовательская выходная сеть.

**MXNet:**

```python
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

**PyTorch:**

```python
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

**PaddlePaddle:**

```python
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = paddle.nn.functional.softmax(net(data), axis=0)
    preds.extend(output.detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

Код выше создаёт файл `submission.csv`, который можно отправить на Kaggle так же, как описано в разделе 4.10.

## Резюме

* Изображения ImageNet крупнее, чем изображения CIFAR-10, поэтому для задач на разных наборах данных может понадобиться менять операции аугментации изображений.
* Чтобы классифицировать подмножество набора ImageNet, можно использовать модель, предобученную на полном ImageNet, для извлечения признаков и обучать только небольшую пользовательскую выходную сеть. Это экономит время вычислений и память.

## Упражнения

1. Попробуйте использовать полный набор данных соревнования Kaggle, увеличить `batch_size` и `num_epochs`, а также задать другие гиперпараметры: `lr = 0.01`, `lr_period = 10`, `lr_decay = 0.1`. Какой результат вы получите?
1. Дают ли более глубокие предобученные модели лучший результат? Как настроить гиперпараметры? Можно ли дополнительно улучшить результат?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2832)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2833)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11815)
