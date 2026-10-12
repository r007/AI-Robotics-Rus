# 8.6 Краткая реализация рекуррентных нейронных сетей

Хотя раздел 8.5 полезен для понимания того, как устроены рекуррентные нейронные сети, его реализация неудобна.
В этом разделе мы покажем, как с помощью функций высокоуровневого API фреймворка глубокого обучения реализовать ту же языковую модель эффективнее.
Начнём, как и раньше, с чтения набора данных «Машина времени».

**MXNet:**
```python
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

**PyTorch:**
```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

**TensorFlow:**
```python
from d2l import tensorflow as d2l
import tensorflow as tf
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

**PaddlePaddle:**
```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
from paddle.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## 8.6.1 Определение модели

Высокоуровневый API предоставляет реализацию рекуррентных нейронных сетей.
Мы создадим рекуррентный слой `rnn_layer` с одним скрытым слоем из 256 скрытых элементов.
Многослойные рекуррентные сети мы пока не обсуждали (они будут рассмотрены в разделе 9.3).
Сейчас достаточно понимать многослойность так: выход одного рекуррентного слоя используется как вход следующего.

**MXNet:**
```python
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

**PyTorch:**
```python
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

**TensorFlow:**
```python
num_hiddens = 256
rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,
    kernel_initializer='glorot_uniform')
rnn_layer = tf.keras.layers.RNN(rnn_cell, time_major=True,
    return_sequences=True, return_state=True)
```

**PaddlePaddle:**
```python
num_hiddens = 256
rnn_layer = nn.SimpleRNN(len(vocab), num_hiddens, time_major=True)
```

**MXNet:** инициализировать скрытое состояние просто: достаточно вызвать метод `begin_state`.
Функция возвращает список (`state`), в котором для каждого примера мини-батча содержится начальное скрытое состояние.
Его форма — (число скрытых слоёв, размер батча, число скрытых элементов).
Для некоторых моделей, которые мы рассмотрим позже (например, для долгой краткосрочной памяти, LSTM), этот список содержит и другую информацию.

**PyTorch:** скрытое состояние задаём тензором формы (число скрытых слоёв, размер батча, число скрытых элементов).

**MXNet:**
```python
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

**PyTorch:**
```python
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

**TensorFlow:**
```python
state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
state.shape
```

**PaddlePaddle:**
```python
state = paddle.zeros(shape=[1, batch_size, num_hiddens])
state.shape
```

**Имея скрытое состояние и вход, мы можем с помощью обновлённого скрытого состояния вычислить выход.**
Важно подчеркнуть, что «выход» `rnn_layer` (`Y`) не включает вычислений выходного слоя: это скрытые состояния на каждом временном шаге, которые можно подать на вход последующего выходного слоя.

**MXNet:** обновлённое скрытое состояние `state_new`, которое возвращает `rnn_layer`, — это скрытое состояние последнего временного шага мини-батча.
Его можно использовать для инициализации скрытого состояния следующего мини-батча в пределах одной эпохи при последовательном разбиении.
Если скрытых слоёв несколько, скрытое состояние каждого слоя хранится в переменной `state_new`.
Для некоторых моделей, которые мы рассмотрим позже (например, для LSTM), эта переменная содержит и другую информацию.

**MXNet:**
```python
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

**PyTorch:**
```python
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

**TensorFlow:**
```python
X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

**PaddlePaddle:**
```python
X = paddle.rand(shape=[num_steps, batch_size, len(vocab)])
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

Как и в разделе 8.5, **определим класс `RNNModel` для полной модели рекуррентной нейронной сети**.
Обратите внимание, что `rnn_layer` содержит только скрытый рекуррентный слой; выходной слой нужно создать отдельно.

**MXNet:**
```python
#@save
class RNNModel(nn.Block):
    """Модель рекуррентной нейронной сети"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # Полносвязный слой сначала меняет форму Y на (число временных шагов * размер батча, число скрытых элементов)
        # Его выход имеет форму (число временных шагов * размер батча, размер словаря)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

**PyTorch:**
```python
#@save
class RNNModel(nn.Module):
    """Модель рекуррентной нейронной сети"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # Если RNN двунаправленная (об этом будет сказано позже), num_directions равно 2, иначе 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # Полносвязный слой сначала меняет форму Y на (число временных шагов * размер батча, число скрытых элементов)
        # Его выход имеет форму (число временных шагов * размер батча, размер словаря)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU использует тензор в качестве скрытого состояния
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM использует кортеж в качестве скрытого состояния
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

**TensorFlow:**
```python
#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # rnn возвращает больше двух значений
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
```

**PaddlePaddle:**
```python
#@save
class RNNModel(nn.Layer):
    """Модель рекуррентной нейронной сети"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # Если RNN двунаправленная (об этом будет сказано позже), num_directions равно 2, иначе 1
        if self.rnn.num_directions==1:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # Полносвязный слой сначала меняет форму Y на (число временных шагов * размер батча, число скрытых элементов)
        # Его выход имеет форму (число временных шагов * размер батча, размер словаря)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU использует тензор в качестве скрытого состояния
            return  paddle.zeros(shape=[self.num_directions * self.rnn.num_layers,
                                                           batch_size, self.num_hiddens])
        else:
            # nn.LSTM использует кортеж в качестве скрытого состояния
            return (paddle.zeros(
                shape=[self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens]),
                    paddle.zeros(
                        shape=[self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens]))
```

## 8.6.2 Обучение и предсказание

Прежде чем обучать модель, **сделаем предсказание с помощью модели со случайными весами**.

**MXNet:**
```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

**PyTorch:**
```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

**TensorFlow:**
```python
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = RNNModel(rnn_layer, vocab_size=len(vocab))

d2l.predict_ch8('time traveller', 10, net, vocab)
```

**PaddlePaddle:**
```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

Очевидно, что такая модель не выдаёт хороших результатов.
Далее мы вызываем `train_ch8` с гиперпараметрами, определёнными в разделе 8.5, и **обучаем модель с помощью высокоуровневого API**.

**MXNet, PyTorch:**
```python
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

**TensorFlow:**
```python
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

**PaddlePaddle:**
```python
num_epochs, lr = 500, 1.0
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

По сравнению с предыдущим разделом модель благодаря более оптимизированному коду высокоуровневого API фреймворка за меньшее время достигает более низкой перплексии.

## Резюме

* Высокоуровневый API фреймворков глубокого обучения предоставляет реализацию слоя рекуррентной нейронной сети.
* Рекуррентный слой высокоуровневого API возвращает выход и обновлённое скрытое состояние, а выходной слой всей модели нужно вычислить отдельно.
* По сравнению с рекуррентной нейронной сетью, реализованной с нуля, реализация с помощью высокоуровневого API ускоряет обучение.

## Упражнения

1. Попробуйте с помощью высокоуровневого API добиться переобучения модели рекуррентной нейронной сети. Получится ли?
1. Что произойдёт, если увеличить число скрытых слоёв в модели рекуррентной нейронной сети? Будет ли модель работать корректно?
1. Попробуйте с помощью рекуррентной нейронной сети реализовать авторегрессионную модель из раздела 8.1.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2105)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2106)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/5766)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11800)
