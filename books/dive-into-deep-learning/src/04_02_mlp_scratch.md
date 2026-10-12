# 4.2 Реализация многослойного перцептрона с нуля

В разделе 4.1 мы описали многослойный перцептрон (MLP).
Теперь попробуем реализовать его сами.
Чтобы результаты было легко сравнить с результатами softmax-регрессии, полученными в разделе 3.6, мы продолжим использовать набор данных для классификации изображений Fashion-MNIST (раздел 3.5).

**MXNet:**
```python
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

**PyTorch:**
```python
from d2l import torch as d2l
import torch
from torch import nn
```

**TensorFlow:**
```python
from d2l import tensorflow as d2l
import tensorflow as tf
```

**PaddlePaddle:**
```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 4.2.1 Инициализация параметров модели

Напомним, что каждое изображение Fashion-MNIST состоит из $28 \times 28 = 784$ значений яркости пикселей.
Всего изображения делятся на 10 классов.
Если не учитывать пространственную структуру пикселей, каждое изображение можно считать вектором из 784 входных признаков, а весь набор данных — простым набором для классификации с 10 классами.
Сначала **реализуем многослойный перцептрон с одним скрытым слоем, содержащим 256 скрытых нейронов**.
Обратите внимание, что обе эти величины можно считать гиперпараметрами.
Обычно ширину слоёв выбирают равной степени двойки.
Это связано с тем, как память выделяется и адресуется в аппаратном обеспечении: такой выбор часто позволяет сделать вычисления эффективнее.

Параметры представим несколькими тензорами.
Обратите внимание, что для каждого слоя нужно хранить матрицу весов и вектор смещений.
Как и раньше, выделяем память для градиентов функции потерь по этим параметрам.

**MXNet:**
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

**PyTorch:**
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

**TensorFlow:**
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.zeros(num_outputs))

params = [W1, b1, W2, b2]
```

**PaddlePaddle:**
```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = paddle.randn([num_inputs, num_hiddens]) * 0.01
W1.stop_gradient = False
b1 = paddle.zeros([num_hiddens])
b1.stop_gradient = False
W2 = paddle.randn([num_hiddens, num_outputs]) * 0.01
W2.stop_gradient = False
b2 = paddle.zeros([num_outputs])
b2.stop_gradient = False

params = [W1, b1, W2, b2]
```

## 4.2.2 Функция активации

Чтобы точно понимать устройство модели, **реализуем функцию активации ReLU** сами, а не вызываем встроенную функцию `relu`.

**MXNet:**
```python
def relu(X):
    return np.maximum(X, 0)
```

**PyTorch:**
```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

**TensorFlow:**
```python
def relu(X):
    return tf.math.maximum(X, 0)
```

**PaddlePaddle:**
```python
def relu(X):
    a = paddle.zeros_like(X)
    return paddle.maximum(X, a)
```

## 4.2.3 Модель

Поскольку пространственную структуру мы не учитываем, с помощью `reshape` преобразуем каждое двумерное изображение в вектор длины `num_inputs`.
**Реализация модели** занимает всего несколько строк.

**MXNet:**
```python
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

**PyTorch:**
```python
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Здесь «@» означает матричное умножение
    return (H@W2 + b2)
```

**TensorFlow:**
```python
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

**PaddlePaddle:**
```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # Здесь «@» означает матричное умножение
    return (H@W2 + b2)
```

## 4.2.4 Функция потерь

Softmax мы уже реализовали с нуля (раздел 3.6), поэтому здесь для softmax и функции потерь кросс-энтропии используем встроенные функции высокоуровневого API.
Напомним, что мы обсуждали эти тонкие вопросы в разделе 3.7.2.
Читателям, которым интересны детали реализации, рекомендуем изучить исходный код функций потерь.

**MXNet:**
```python
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

**PyTorch, PaddlePaddle:**
```python
loss = nn.CrossEntropyLoss(reduction='none')
```

**TensorFlow:**
```python
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## 4.2.5 Обучение

К счастью, **процесс обучения многослойного перцептрона полностью совпадает с процессом обучения softmax-регрессии**.
Можно напрямую вызвать функцию `train_ch3` из пакета `d2l` (см. раздел 3.6), задав число эпох 10 и скорость обучения 0.1.

**MXNet:**
```python
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

**PyTorch:**
```python
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

**TensorFlow:**
```python
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

**PaddlePaddle:**
```python
num_epochs, lr = 10, 0.1
updater = paddle.optimizer.SGD(learning_rate=lr, parameters=params)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

Чтобы оценить обученную модель, **применим её к тестовым данным**.

```python
d2l.predict_ch3(net, test_iter)
```

## Резюме

* Реализовать простой многослойный перцептрон вручную несложно. Однако при большом числе слоёв реализация с нуля становится громоздкой (например, приходится давать имена и вести учёт параметров модели).

## Упражнения

1. Меняя гиперпараметр `num_hiddens` при неизменных остальных параметрах, посмотрите, как это влияет на результат. Определите наилучшее значение этого гиперпараметра.
1. Попробуйте добавить больше скрытых слоёв и посмотрите, как это повлияет на результат.
1. Как на результат влияет скорость обучения? Какую скорость обучения нужно выбрать, чтобы получить лучший результат при неизменной архитектуре модели и остальных гиперпараметрах (включая число эпох)?
1. Какой наилучший результат можно получить, оптимизируя все гиперпараметры совместно (скорость обучения, число эпох, число скрытых слоёв, число скрытых элементов в каждом слое)?
1. Объясните, почему работа сразу с несколькими гиперпараметрами сложнее.
1. Придумайте разумную стратегию поиска по нескольким гиперпараметрам.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/1800)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/1804)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/1798)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11769)
