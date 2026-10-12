# 3.5 Набор данных для классификации изображений

**Набор данных MNIST** (LeCun et al., 1998) — один из самых широко используемых наборов для классификации изображений, но как эталон для сравнения методов он слишком прост.
Поэтому мы будем использовать похожий, но более сложный набор — **Fashion-MNIST** (Xiao et al., 2017).

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

**PyTorch:**

```python
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

**TensorFlow:**

```python
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

**PaddlePaddle:**

```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import sys
import paddle
from paddle.vision import transforms

d2l.use_svg_display()
```

## 3.5.1 Чтение набора данных

Мы можем **загрузить набор Fashion-MNIST в память с помощью встроенных функций фреймворка**.

**MXNet:**

```python
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

**PyTorch:**

```python
# Преобразование ToTensor переводит изображения из формата PIL в 32-битные числа
# с плавающей точкой и делит их на 255, чтобы значения всех пикселей лежали в диапазоне от 0 до 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

**TensorFlow:**

```python
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

**PaddlePaddle:**

```python
trans = transforms.ToTensor()
mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                  transform=trans)
mnist_test = paddle.vision.datasets.FashionMNIST(mode="test", transform=trans)
```

Набор Fashion-MNIST состоит из изображений десяти классов.
Каждый класс представлен 6000 изображениями в *обучающей выборке* (train dataset) и 1000 изображениями в *тестовой выборке* (test dataset).
Таким образом, обучающая и тестовая выборки содержат 60000 и 10000 изображений соответственно.
Тестовая выборка не используется для обучения, а только для оценки качества модели.

**MXNet, PyTorch, PaddlePaddle:**

```python
len(mnist_train), len(mnist_test)
```

**TensorFlow:**

```python
len(mnist_train[0]), len(mnist_test[0])
```

Высота и ширина каждого входного изображения составляют 28 пикселей.
Набор состоит из изображений в оттенках серого, поэтому число каналов равно 1.
Для краткости в этой книге форму изображения высотой $h$ пикселей и шириной $w$ пикселей будем записывать как $h \times w$ или ($h$, $w$).

```python
mnist_train[0][0].shape
```

Набор Fashion-MNIST содержит 10 классов: футболка (t-shirt), брюки (trouser), пуловер (pullover), платье (dress), пальто (coat), сандалия (sandal), рубашка (shirt), кроссовок (sneaker), сумка (bag) и ботильон (ankle boot).
Следующая функция преобразует числовые метки в текстовые названия классов.

```python
def get_fashion_mnist_labels(labels):  #@save
    """Возвращает текстовые метки набора Fashion-MNIST"""
    text_labels = ['футболка', 'брюки', 'пуловер', 'платье', 'пальто',
                   'сандалия', 'рубашка', 'кроссовок', 'сумка', 'ботильон']
    return [text_labels[int(i)] for i in labels]
```

Теперь создадим функцию, которая визуализирует эти образцы.

**MXNet, TensorFlow:**

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Вывод списка изображений"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

**PyTorch:**

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Вывод списка изображений"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Тензор изображения
            ax.imshow(img.numpy())
        else:
            # Изображение PIL
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

**PaddlePaddle:**

```python
#@save
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Вывод списка изображений"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if paddle.is_tensor(img):
            # Тензор изображения
            ax.imshow(img.numpy())
        else:
            # Изображение PIL
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Ниже показаны **изображения и соответствующие метки первых нескольких образцов обучающей выборки**.

**MXNet:**

```python
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

**PyTorch:**

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

**TensorFlow:**

```python
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

**PaddlePaddle:**

```python
X, y = next(iter(paddle.io.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape([18, 28, 28]), 2, 9, titles=get_fashion_mnist_labels(y));
```

## 3.5.2 Чтение мини-батчей

Чтобы упростить чтение обучающей и тестовой выборок, мы используем встроенные итераторы данных, а не создаём их с нуля.
Напомним, что на каждой итерации загрузчик данных читает мини-батч размера `batch_size`.
Встроенный итератор перемешивает все образцы, поэтому мини-батчи читаются без смещения.

**MXNet:**

```python
batch_size = 256

def get_dataloader_workers():  #@save
    """На платформах, отличных от Windows, для чтения данных используются 4 процесса"""
    return 0 if sys.platform.startswith('win') else 4

# Преобразование ToTensor переводит изображения из формата uint8 в 32-битные числа
# с плавающей точкой и делит их на 255, чтобы значения всех пикселей лежали в диапазоне от 0 до 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

**PyTorch:**

```python
batch_size = 256

def get_dataloader_workers():  #@save
    """Для чтения данных используются 4 процесса"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

**TensorFlow:**

```python
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

**PaddlePaddle:**

```python
batch_size = 256

def get_dataloader_workers():  #@save
    """Для чтения данных используются 4 процесса"""
    return 4

train_iter = paddle.io.DataLoader(dataset=mnist_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  return_list=True,
                                  num_workers=get_dataloader_workers())
```

Посмотрим, сколько времени занимает чтение обучающих данных.

```python
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} сек'
```

## 3.5.3 Объединение всех компонентов

Теперь определим функцию **`load_data_fashion_mnist`**, которая загружает набор Fashion-MNIST и читает его.
Функция возвращает итераторы данных для обучающей и проверочной выборок.
Кроме того, она принимает необязательный параметр `resize`, с помощью которого можно изменить размер изображений.

**MXNet:**

```python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Загрузка набора Fashion-MNIST и чтение его в память"""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

**PyTorch:**

```python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Загрузка набора Fashion-MNIST и чтение его в память"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

**TensorFlow:**

```python
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Загрузка набора Fashion-MNIST и чтение его в память"""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Все значения делятся на 255, чтобы значения пикселей лежали в диапазоне от 0 до 1,
    # в конец добавляется измерение пакета, а метки приводятся к типу int32.
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

**PaddlePaddle:**

```python
#@save
def load_data_fashion_mnist(batch_size, resize=None):  
    """Загрузка набора Fashion-MNIST и чтение его в память"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                      transform=trans)
    mnist_test = paddle.vision.datasets.FashionMNIST(mode="test",
                                                     transform=trans)
    return (paddle.io.DataLoader(dataset=mnist_train,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 return_list=True,
                                 num_workers=get_dataloader_workers()),
            paddle.io.DataLoader(dataset=mnist_test,
                                 batch_size=batch_size,
                                 return_list=True,
                                 shuffle=True,
                                 num_workers=get_dataloader_workers()))
```

Проверим работу функции `load_data_fashion_mnist` с изменением размера изображений, передав параметр `resize`.

```python
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

Теперь всё готово к работе с набором Fashion-MNIST: в следующих разделах мы будем использовать его для оценки различных алгоритмов классификации.

## Резюме

* Fashion-MNIST — набор данных с изображениями одежды десяти классов. Мы будем использовать его в следующих главах для оценки различных алгоритмов классификации.
* Форму изображения высотой $h$ и шириной $w$ пикселей будем записывать как $h \times w$ или ($h$, $w$).
* Итераторы данных — ключевой компонент для высокой производительности. Хорошо реализованные итераторы, использующие возможности высокопроизводительных вычислений, не дают замедлять обучение.

## Упражнения

1. Повлияет ли уменьшение `batch_size` (например, до 1) на производительность чтения данных?
1. Производительность итераторов данных очень важна. Достаточно ли быстра текущая реализация? Исследуйте различные способы её улучшить.
1. Посмотрите онлайн-документацию API фреймворка. Какие ещё наборы данных доступны?

* MXNet: [discuss.d2l.ai/t/1788](https://discuss.d2l.ai/t/1788)
* PyTorch: [discuss.d2l.ai/t/1787](https://discuss.d2l.ai/t/1787)
* TensorFlow: [discuss.d2l.ai/t/1786](https://discuss.d2l.ai/t/1786)
* PaddlePaddle: [discuss.d2l.ai/t/11692](https://discuss.d2l.ai/t/11692)
