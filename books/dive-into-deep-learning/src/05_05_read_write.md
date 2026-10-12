# 5.5 Чтение и запись файлов

До сих пор мы обсуждали, как обрабатывать данные, а также как строить, обучать и тестировать модели глубокого обучения.
Однако иногда мы хотим сохранить обученную модель, чтобы использовать её в будущем в различных условиях
(например, для предсказаний при развёртывании).
Кроме того, при запуске длительного процесса обучения разумно регулярно сохранять промежуточные результаты,
чтобы при неожиданном отключении питания сервера не потерять результаты вычислений за несколько дней.
Поэтому пора научиться загружать и сохранять векторы весов и целые модели.

## 5.5.1 Загрузка и сохранение тензоров

Для отдельного тензора можно напрямую вызвать функции `load` и `save`, чтобы читать и записывать его.
Обе эти функции требуют указать имя файла; функция `save` также принимает сохраняемую переменную.

**MXNet:**
```python
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

**PyTorch:**
```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

**TensorFlow:**
```python
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save('x-file.npy', x)
```

**PaddlePaddle:**
```python
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn 
from paddle.nn import functional as F

x = paddle.arange(4)  
paddle.save(x, 'x-file')
```

Теперь мы можем считать данные, сохранённые в файле, обратно в память.

**MXNet:**
```python
x2 = npx.load('x-file')
x2
```

**PyTorch:**
```python
x2 = torch.load('x-file')
x2
```

**TensorFlow:**
```python
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

**PaddlePaddle:**
```python
x2 = paddle.load('x-file')
x2
```

Мы можем **сохранить список тензоров и затем прочитать их обратно в память.**

**MXNet:**
```python
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

**PyTorch:**
```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

**TensorFlow:**
```python
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

**PaddlePaddle:**
```python
y = paddle.zeros([4])
paddle.save([x,y], 'x-file')
x2, y2 = paddle.load('x-file')
(x2, y2)
```

Мы даже можем **записывать и читать словарь, который отображает строки в тензоры**.
Это удобно, когда нужно читать или записывать все весовые параметры модели.

**MXNet:**
```python
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

**PyTorch:**
```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

**TensorFlow:**
```python
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

**PaddlePaddle:**
```python
mydict = {'x': x, 'y': y}
paddle.save(mydict, 'mydict')
mydict2 = paddle.load('mydict')
mydict2
```

## 5.5.2 Загрузка и сохранение параметров модели

Сохранять отдельный весовой вектор (или другой тензор) действительно полезно,
но если мы хотим сохранить всю модель и загрузить её позже, сохранять каждый вектор по отдельности будет слишком громоздко:
у нас могут быть сотни параметров, разбросанных по всей модели.
Поэтому фреймворки глубокого обучения предоставляют встроенные функции для сохранения и загрузки всей сети.
Важно учитывать, что при этом сохраняются параметры модели, а не вся модель целиком.
Например, если у нас есть трёхслойный многослойный перцептрон, архитектуру нужно указать отдельно.
Поскольку сама модель может содержать произвольный код, её сложно сериализовать.
Поэтому, чтобы восстановить модель, нужно сгенерировать архитектуру с помощью кода, а затем загрузить параметры с диска.
Попробуем это на знакомом многослойном перцептроне.

**MXNet:**
```python
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

**PyTorch:**
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

**TensorFlow:**
```python
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

**PaddlePaddle:**
```python
class MLP(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = paddle.randn(shape=[2, 20])
Y = net(X)
```

Далее мы **сохраним параметры модели в файл с именем «mlp.params».**

**MXNet:**
```python
net.save_parameters('mlp.params')
```

**PyTorch:**
```python
torch.save(net.state_dict(), 'mlp.params')
```

**TensorFlow:**
```python
net.save_weights('mlp.params')
```

**PaddlePaddle:**
```python
paddle.save(net.state_dict(), 'mlp.pdparams')
```

Чтобы восстановить модель, мы **создаём копию исходного многослойного перцептрона.**
Здесь нам не нужно случайным образом инициализировать параметры модели, вместо этого
**напрямую загружаем параметры, сохранённые в файле.**

**MXNet:**
```python
clone = MLP()
clone.load_parameters('mlp.params')
```

**PyTorch:**
```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

**TensorFlow:**
```python
clone = MLP()
clone.load_weights('mlp.params')
```

**PaddlePaddle:**
```python
clone = MLP()
clone.set_state_dict(paddle.load('mlp.pdparams'))
clone.eval()
```

Поскольку у двух экземпляров одинаковые параметры модели, при одинаковом входе `X`
результаты вычислений обоих экземпляров должны совпадать.
Проверим это.

**MXNet:**
```python
Y_clone = clone(X)
Y_clone == Y
```

**PyTorch, PaddlePaddle:**
```python
Y_clone = clone(X)
Y_clone == Y
```

**TensorFlow:**
```python
Y_clone = clone(X)
Y_clone == Y
```

## Резюме

* Функции `save` и `load` используются для записи и чтения тензорных объектов в файлы.
* Мы можем сохранять и загружать все параметры сети через словарь параметров.
* Архитектуру модели нужно сохранять в коде, а не в параметрах.

## Упражнения

1. Даже если не требуется развёртывать обученную модель на других устройствах, какая практическая польза от сохранения параметров модели?
1. Предположим, мы хотим использовать только часть сети, чтобы включить её в другую архитектуру. Например, мы хотим использовать первые два слоя предыдущей сети в новой сети. Как это сделать?
1. Как сохранить одновременно архитектуру сети и её параметры? Какие ограничения нужно наложить на архитектуру?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/1840)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/1839)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/1838)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11781)
