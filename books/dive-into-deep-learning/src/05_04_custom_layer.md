# 5.4 Пользовательские слои

Одна из причин успеха глубокого обучения — гибкость нейронных сетей:
мы можем комбинировать различные слои самыми разными способами и создавать архитектуры, подходящие для самых разных задач.
Например, исследователи изобрели слои, специально предназначенные для обработки изображений, текста, последовательных данных и для выполнения динамического программирования.
Иногда нам встречается слой, которого ещё нет в фреймворке глубокого обучения, или мы сами хотим придумать новый слой.
В таких случаях приходится создавать пользовательский слой. В этом разделе показано, как это сделать.

## 5.4.1 Слои без параметров

Сначала **создадим пользовательский слой без параметров**.
Вспомним введение блоков в разделе 5.1: этот материал должен быть знаком.
Приведённый ниже класс `CenteredLayer` вычитает среднее значение из своего входа.
Чтобы создать его, нужно унаследоваться от базового класса слоя и реализовать функцию прямого распространения.

**MXNet:**
```python
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

**PyTorch:**
```python
import torch
from torch import nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

**TensorFlow:**
```python
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

**PaddlePaddle:**
```python
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn
import paddle.nn.functional as F

class CenteredLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

Подадим в этот слой какие-нибудь данные, чтобы проверить, что он работает как ожидается.

**MXNet:**
```python
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

**PyTorch:**
```python
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

**TensorFlow:**
```python
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

**PaddlePaddle:**
```python
layer = CenteredLayer()
layer(paddle.to_tensor([1, 2, 3, 4, 5], dtype='float32'))
```

Теперь мы можем **включить слой как компонент в более сложную модель**.

**MXNet:**
```python
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

**PyTorch, PaddlePaddle:**
```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

**TensorFlow:**
```python
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

В качестве дополнительной проверки здравого смысла мы можем отправить в сеть случайные данные и убедиться, что среднее значение выхода равно 0.
Поскольку мы работаем с числами с плавающей точкой, из-за ограниченной точности хранения мы всё равно можем получить очень маленькое ненулевое число.

**MXNet:**
```python
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

**PyTorch:**
```python
Y = net(torch.rand(4, 8))
Y.mean()
```

**TensorFlow:**
```python
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

**PaddlePaddle:**
```python
Y = net(paddle.rand([4, 8]))
Y.mean()
```

## 5.4.2 Слои с параметрами

Выше мы разобрали, как определять простые слои. Теперь определим слои, которые имеют параметры, настраиваемые в процессе обучения.
Для создания параметров мы можем использовать встроенные функции, которые обеспечивают базовое управление: доступом, инициализацией, совместным использованием, сохранением и загрузкой параметров модели.
Одно из преимуществ этого подхода в том, что нам не нужно писать собственный механизм сериализации для каждого пользовательского слоя.

Теперь реализуем пользовательскую версию полносвязного слоя.
Напомним, что этому слою нужны два параметра: один для весов и другой для смещения.
В этой реализации в качестве функции активации используем ReLU.
Слой принимает два входных аргумента: `in_units` и `units`, которые задают количество входов и количество выходов соответственно.

**MXNet:**
```python
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

**PyTorch:**
```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

**TensorFlow:**
```python
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

**PaddlePaddle:**
```python
class MyLinear(nn.Layer):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = paddle.create_parameter(shape=(in_units, units), dtype='float32')
        self.bias = paddle.create_parameter(shape=(units,), dtype='float32')
        
    def forward(self, X):
        linear = paddle.matmul(X, self.weight) + self.bias
        return F.relu(linear)
```

**MXNet, TensorFlow:** Далее мы создаём экземпляр класса `MyDense` и обращаемся к его параметрам модели.

**PyTorch, PaddlePaddle:** Далее мы создаём экземпляр класса `MyLinear` и обращаемся к его параметрам модели.

**MXNet:**
```python
dense = MyDense(units=3, in_units=5)
dense.params
```

**PyTorch, PaddlePaddle:**
```python
linear = MyLinear(5, 3)
linear.weight
```

**TensorFlow:**
```python
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

Мы можем **выполнять прямое распространение, используя пользовательский слой напрямую**.

**MXNet:**
```python
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

**PyTorch:**
```python
linear(torch.rand(2, 5))
```

**TensorFlow:**
```python
dense(tf.random.uniform((2, 5)))
```

**PaddlePaddle:**
```python
linear(paddle.randn([2, 5]))
```

Мы также можем **построить модель с помощью пользовательского слоя**, используя его точно так же, как встроенный полносвязный слой.

**MXNet:**
```python
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

**PyTorch:**
```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

**TensorFlow:**
```python
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

**PaddlePaddle:**
```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(paddle.rand([2, 64]))
```

## Резюме

* Мы можем создавать пользовательские слои на основе базового класса слоя. Это позволяет определять гибкие новые слои, поведение которых отличается от любого существующего слоя фреймворка глубокого обучения.
* После определения пользовательского слоя его можно вызывать в любой среде и в любой архитектуре сети.
* Слои могут иметь локальные параметры, которые можно создавать с помощью встроенных функций.

## Упражнения

1. Разработайте слой, который принимает входные данные и вычисляет понижение размерности тензора: он возвращает $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Разработайте слой, который возвращает первую половину коэффициентов Фурье входных данных.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/1837)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/1835)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/1836)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11780)
