# 13.6 Набор данных для обнаружения объектов

В задаче обнаружения объектов нет маленьких наборов данных, таких как MNIST или Fashion-MNIST. Чтобы быстро проверить модели обнаружения, [**мы соберём и разметим небольшой набор данных**]. Сначала мы сфотографировали бананы и получили 1000 изображений бананов с разными углами и размерами. Затем на случайные места фоновых изображений мы поместили изображения бананов. Наконец, на этих изображениях мы разметили ограничивающие рамки бананов.

## [**Загрузка набора данных**]

Набор данных для обнаружения бананов, содержащий все изображения и CSV-файл с метками, можно загрузить напрямую из интернета.

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

**PyTorch:**

```python
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

**PaddlePaddle:**

```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import paddle
import paddle.vision as paddlevision
```

```python
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Чтение набора данных

С помощью функции `read_data_bananas` мы [**читаем набор данных для обнаружения бананов**]. Набор включает CSV-файл с метками классов и координатами истинных ограничивающих рамок в левом верхнем и правом нижнем углах.

**MXNet:**

```python
#@save
def read_data_bananas(is_train=True):
    """Читает изображения и метки из набора данных для обнаружения бананов"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Здесь target содержит (класс, x левого верхнего, y левого верхнего, x правого нижнего, y правого нижнего),
        # причём все изображения имеют один и тот же класс «банан» (индекс 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

**PyTorch:**

```python
#@save
def read_data_bananas(is_train=True):
    """Читает изображения и метки из набора данных для обнаружения бананов"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Здесь target содержит (класс, x левого верхнего, y левого верхнего, x правого нижнего, y правого нижнего),
        # причём все изображения имеют один и тот же класс «банан» (индекс 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

**PaddlePaddle:**

```python
#@save
def read_data_bananas(is_train=True):
    """Читает изображения и метки из набора данных для обнаружения бананов"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        paddle.vision.set_image_backend('cv2')
        images.append(paddlevision.image_load(os.path.join(data_dir, 'bananas_train' if is_train else
        'bananas_val', 'images', f'{img_name}'))[..., ::-1])
        # Здесь target содержит (класс, x левого верхнего, y левого верхнего, x правого нижнего, y правого нижнего),
        # причём все изображения имеют один и тот же класс «банан» (индекс 0)
        targets.append(list(target))
    return images, paddle.to_tensor(targets).unsqueeze(1) / 256
```

Используя функцию `read_data_bananas` для чтения изображений и меток, класс `BananasDataset` ниже позволяет [**создать собственный экземпляр `Dataset`**] для загрузки набора данных для обнаружения бананов.

**MXNet:**

```python
#@save
class BananasDataset(gluon.data.Dataset):
    """Пользовательский набор данных для загрузки набора для обнаружения бананов"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('прочитано ' + str(len(self.features)) + (f' обучающих примеров' if
              is_train else f' проверочных примеров'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

**PyTorch:**

```python
#@save
class BananasDataset(torch.utils.data.Dataset):
    """Пользовательский набор данных для загрузки набора для обнаружения бананов"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('прочитано ' + str(len(self.features)) + (f' обучающих примеров' if
              is_train else f' проверочных примеров'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

**PaddlePaddle:**

```python
#@save
class BananasDataset(paddle.io.Dataset):
    """Пользовательский набор данных для загрузки набора для обнаружения бананов"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('прочитано ' + str(len(self.features)) + (f' обучающих примеров' if
              is_train else f' проверочных примеров'))

    def __getitem__(self, idx):
        return (paddle.to_tensor(self.features[idx], dtype='float32').transpose([2, 0, 1]), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Наконец, определим функцию `load_data_bananas`, которая [**возвращает два экземпляра загрузчика данных для обучающего и тестового наборов**]. Для тестового набора не нужно читать данные в случайном порядке.

**MXNet:**

```python
#@save
def load_data_bananas(batch_size):
    """Загружает набор данных для обнаружения бананов"""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

**PyTorch:**

```python
#@save
def load_data_bananas(batch_size):
    """Загружает набор данных для обнаружения бананов"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

**PaddlePaddle:**

```python
#@save
def load_data_bananas(batch_size):
    """Загружает набор данных для обнаружения бананов"""
    train_iter = paddle.io.DataLoader(BananasDataset(is_train=True),
                                      batch_size=batch_size, return_list=True, shuffle=True)
    val_iter = paddle.io.DataLoader(BananasDataset(is_train=False),
                                    batch_size=batch_size, return_list=True)
    return train_iter, val_iter
```

Прочитаем одну мини-батч и выведем формы изображений и меток. Форма мини-батча изображений — (размер батча, число каналов, высота, ширина), как и в задачах классификации изображений. Форма мини-батча меток — (размер батча, $m$, 5), где $m$ — максимальное число ограничивающих рамок, которое может встретиться в изображении набора данных.

Вычисления на мини-батчах эффективны, но требуют, чтобы у всех изображений в одном батче было одинаковое число ограничивающих рамок. Обычно число рамок у изображений разное, поэтому изображения с числом рамок меньше $m$ дополняются фиктивными (нелегальными) рамками до $m$. Тогда метка каждой рамки представлена массивом длины 5. Первый элемент массива — класс объекта в рамке, где -1 означает фиктивную (нелегальную) рамку, предназначенную для дополнения. Остальные четыре элемента — координаты $(x, y)$ левого верхнего и правого нижнего углов рамки (значения от 0 до 1). В наборе данных о бананах на каждом изображении ровно одна рамка, поэтому $m=1$.

```python
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Демонстрация**]

Покажем 10 изображений с истинными ограничивающими рамками. Видно, что на всех этих изображениях бананы повёрнуты по-разному, имеют разный размер и расположены в разных местах. Разумеется, это простой искусственный набор данных; реальные наборы данных на практике обычно гораздо сложнее.

**MXNet:**

```python
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

**PyTorch:**

```python
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

**PaddlePaddle:**

```python
imgs = (batch[0][0:10].transpose([0, 2, 3, 1])) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Резюме

* Собранный нами набор данных для обнаружения бананов можно использовать для демонстрации моделей обнаружения объектов.
* Загрузка данных для обнаружения объектов похожа на загрузку данных для классификации изображений. Однако в обнаружении объектов метки также содержат информацию об истинных ограничивающих рамках, которой нет в классификации изображений.

## Упражнения

1. Продемонстрируйте другие изображения с истинными ограничивающими рамками из набора данных для обнаружения бананов. Чем они отличаются по рамкам и объектам?
1. Допустим, мы хотим применить аугментацию данных (например, случайную обрезку) к обнаружению объектов. Чем это отличается от классификации изображений? Подсказка: что будет, если обрезанное изображение содержит только небольшую часть объекта?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/3203)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/3202)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11806)
:end_tab:
