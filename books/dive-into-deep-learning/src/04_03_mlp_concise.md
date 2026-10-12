# 4.3 Краткая реализация многослойного перцептрона

В этом разделе мы покажем, как **реализовать многослойный перцептрон короче, с помощью высокоуровневого API**.

**MXNet:**
```python
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
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

## 4.3.1 Модель

По сравнению с краткой реализацией softmax-регрессии (раздел 3.7) здесь есть одно отличие: мы добавляем два полносвязных слоя вместо одного.
Первый слой — **скрытый слой**, который **содержит 256 скрытых нейронов и использует функцию активации ReLU**.
Второй слой — выходной.

**MXNet:**
```python
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

**PyTorch:**
```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

**TensorFlow:**
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

**PaddlePaddle:**
```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))


for layer in net:
    if type(layer) == nn.Linear:
        weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))
        layer.weight_attr = weight_attr
```

**Процесс обучения** реализуется точно так же, как при реализации softmax-регрессии.
Такое модульное устройство позволяет отделить всё, что относится к архитектуре модели, от остального.

**MXNet:**
```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

**PyTorch:**
```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

**TensorFlow:**
```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

**PaddlePaddle:**
```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=lr)
```

```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Резюме

* Многослойный перцептрон можно реализовать с помощью высокоуровневого API существенно короче.
* Для одной и той же задачи классификации реализация многослойного перцептрона совпадает с реализацией softmax-регрессии, за исключением того, что в многослойном перцептроне добавляется скрытый слой с функцией активации.

## Упражнения

1. Попробуйте добавить разное число скрытых слоёв (можно также менять скорость обучения). Какая настройка даёт лучший результат?
1. Попробуйте разные функции активации. Какая из них работает лучше всего?
1. Попробуйте разные способы инициализации весов. Какой метод работает лучше всего?

* MXNet: [discuss.d2l.ai/t/1803](https://discuss.d2l.ai/t/1803)
* PyTorch: [discuss.d2l.ai/t/1802](https://discuss.d2l.ai/t/1802)
* TensorFlow: [discuss.d2l.ai/t/1801](https://discuss.d2l.ai/t/1801)
* PaddlePaddle: [discuss.d2l.ai/t/11770](https://discuss.d2l.ai/t/11770)
