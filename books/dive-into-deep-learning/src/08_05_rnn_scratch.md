# 8.5 Реализация рекуррентной нейронной сети с нуля

В этом разделе на основе описания из раздела 8.4 мы с нуля реализуем символьную языковую модель на основе рекуррентной нейронной сети.
Модель будем обучать на наборе данных «Машина времени» Герберта Уэллса.
Как и в разделе 8.3, сначала прочитаем набор данных.

**MXNet:**
```python
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

**PyTorch:**
```python
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

**TensorFlow:**
```python
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

**PaddlePaddle:**
```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn
from paddle.nn import functional as F
```

```python
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

**TensorFlow:**
```python
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## 8.5.1 One-hot кодирование

Напомним, что в `train_iter` каждый токен представлен числовым индексом.
Если подавать такие индексы напрямую в нейронную сеть, обучение может быть затруднено.
Поэтому обычно каждый токен представляют более выразительным вектором признаков.
Самое простое представление называется *one-hot кодированием* (one-hot encoding); оно было введено в разделе 3.4.1.

Коротко говоря, каждый индекс отображается в свой единичный вектор.
Пусть число различных токенов в словаре равно $N$ (то есть `len(vocab)`), а индексы токенов принимают значения от $0$ до $N-1$.
Если индекс токена равен целому числу $i$, мы создаём вектор длины $N$, заполненный нулями, и устанавливаем его элемент с индексом $i$ равным $1$.
Полученный вектор — one-hot вектор исходного токена.
Ниже показаны one-hot векторы для индексов $0$ и $2$:

**MXNet:**
```python
npx.one_hot(np.array([0, 2]), len(vocab))
```

**PyTorch:**
```python
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

**TensorFlow:**
```python
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

**PaddlePaddle:**
```python
F.one_hot(paddle.to_tensor([0, 2]), len(vocab))
```

Каждый выбранный минибатч имеет форму двумерного тензора: (размер батча, число временных шагов).
Функция `one_hot` превращает такой минибатч в трёхмерный тензор, последняя размерность которого равна размеру словаря (`len(vocab)`).
Мы часто переставляем оси входа, чтобы получить выход формы (число временных шагов, размер батча, размер словаря).
Это позволяет удобно перебирать минибатч по самой внешней оси и шаг за шагом обновлять скрытое состояние минибатча.

**MXNet:**
```python
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

**PyTorch:**
```python
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

**TensorFlow:**
```python
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

**PaddlePaddle:**
```python
X = paddle.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape
```

## 8.5.2 Инициализация параметров модели

Теперь **инициализируем параметры модели рекуррентной нейронной сети**.
Число скрытых элементов `num_hiddens` — настраиваемый гиперпараметр.
При обучении языковой модели входы и выходы берутся из одного и того же словаря.
Поэтому их размерности совпадают и равны размеру словаря.

**MXNet:**
```python
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # Параметры скрытого слоя
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Параметры выходного слоя
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Включаем вычисление градиентов
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

**PyTorch:**
```python
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Параметры скрытого слоя
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Параметры выходного слоя
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Включаем вычисление градиентов
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

**TensorFlow:**
```python
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    # Параметры скрытого слоя
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Параметры выходного слоя
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

**PaddlePaddle:**
```python
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return paddle.randn(shape=shape)* 0.01

    # Параметры скрытого слоя
    W_xh = normal([num_inputs, num_hiddens])
    W_hh = normal([num_hiddens, num_hiddens])
    b_h = d2l.zeros(shape=[num_hiddens])
    # Параметры выходного слоя
    W_hq = normal([num_hiddens, num_outputs])
    b_q = d2l.zeros(shape=[num_outputs])
    # Включаем вычисление градиентов
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.stop_gradient=False
    return params
```

## 8.5.3 Модель рекуррентной нейронной сети

Чтобы определить модель рекуррентной нейронной сети,
сначала нужна **функция `init_rnn_state`, которая возвращает скрытое состояние при инициализации**.
Эта функция возвращает тензор, заполненный нулями, формы (размер батча, число скрытых элементов).
В следующих главах мы встретимся с ситуациями, когда скрытое состояние содержит несколько переменных; их удобнее хранить в кортеже.

**MXNet:**
```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

**PyTorch:**
```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

**TensorFlow:**
```python
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

**PaddlePaddle:**
```python
def init_rnn_state(batch_size, num_hiddens):
    return (paddle.zeros(shape=[batch_size, num_hiddens]), )
```

**Функция `rnn` ниже определяет, как вычислить скрытое состояние и выход на одном временном шаге.**
Модель рекуррентной нейронной сети реализует цикл по самой внешней оси `inputs`, чтобы последовательно обновлять скрытое состояние `H` минибатча.
Кроме того, здесь в качестве функции активации используется $\tanh$.
Как описано в разделе 4.1, при равномерном распределении элементов на вещественной оси среднее значение функции $\tanh$ равно 0.

**MXNet:**
```python
def rnn(inputs, state, params):
    # Форма inputs: (число временных шагов, размер батча, размер словаря)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Форма X: (размер батча, размер словаря)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

**PyTorch:**
```python
def rnn(inputs, state, params):
    # Форма inputs: (число временных шагов, размер батча, размер словаря)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Форма X: (размер батча, размер словаря)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

**TensorFlow:**
```python
def rnn(inputs, state, params):
    # Форма inputs: (число временных шагов, размер батча, размер словаря)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Форма X: (размер батча, размер словаря)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

**PaddlePaddle:**
```python
def rnn(inputs, state, params):
    # Форма inputs: (число временных шагов, размер батча, размер словаря)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Форма X: (размер батча, размер словаря)
    for X in inputs:
        H = paddle.tanh(paddle.mm(X, W_xh) + paddle.mm(H, W_hh) + b_h)
        Y = paddle.mm(H, W_hq) + b_q
        outputs.append(Y)
    return paddle.concat(x=outputs, axis=0), (H,)
```

Определив все нужные функции, создадим **класс, который объединяет эти функции** и хранит параметры модели рекуррентной нейронной сети, реализованной с нуля.

**MXNet:**
```python
class RNNModelScratch:  #@save
    """Модель рекуррентной нейронной сети, реализованная с нуля"""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

**PyTorch:**
```python
class RNNModelScratch: #@save
    """Модель рекуррентной нейронной сети, реализованная с нуля"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

**TensorFlow:**
```python
class RNNModelScratch: #@save
    """Модель рекуррентной нейронной сети, реализованная с нуля"""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)
```

**PaddlePaddle:**
```python
class RNNModelScratch: #@save
    """Модель рекуррентной нейронной сети, реализованная с нуля"""
    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)
```

Проверим, **правильна ли форма выхода**: например, сохраняется ли размерность скрытого состояния.

**MXNet:**
```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

**PyTorch:**
```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

**TensorFlow:**
```python
# Определяем стратегию обучения для TensorFlow
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

**PaddlePaddle:**
```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

Видно, что форма выхода равна (число временных шагов $\times$ размер батча, размер словаря), а форма скрытого состояния не меняется и равна (размер батча, число скрытых элементов).

## 8.5.4 Предсказание

Сначала определим функцию предсказания, которая **генерирует новые символы после `prefix`**; `prefix` — это строка из нескольких символов, которую задаёт пользователь.
При проходе по начальным символам `prefix` мы продолжаем передавать скрытое состояние на следующий временной шаг, но ничего не выводим.
Это называется *периодом разогрева* (warm-up), потому что в этот период модель обновляет себя (например, обновляет скрытое состояние), но предсказаний не делает.
После периода разогрева скрытое состояние обычно лучше подходит для предсказаний, чем исходное, и тогда модель начинает предсказывать символы и выводить их.

**MXNet:**
```python
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Генерирует новые символы после prefix"""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Период разогрева
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Предсказываем num_preds шагов
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

**PyTorch:**
```python
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Генерирует новые символы после prefix"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Период разогрева
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Предсказываем num_preds шагов
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

**TensorFlow:**
```python
def predict_ch8(prefix, num_preds, net, vocab):  #@save
    """Генерирует новые символы после prefix"""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), 
                                    (1, 1)).numpy()
    for y in prefix[1:]:  # Период разогрева
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Предсказываем num_preds шагов
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

**PaddlePaddle:**
```python
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Генерирует новые символы после prefix"""
    state = net.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(outputs[-1], place=device), (1, 1))
    for y in prefix[1:]:  # Период разогрева
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Предсказываем num_preds шагов
        y, state = net(get_input(), state)
        outputs.append(int(paddle.reshape(paddle.argmax(y,axis=1),shape=[1])))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Теперь можно проверить функцию `predict_ch8`.
Зададим префикс `time traveller ` и сгенерируем по нему 10 последующих символов.
Поскольку сеть ещё не обучена, предсказания будут бессмысленными.

**MXNet, PyTorch, PaddlePaddle:**
```python
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

**TensorFlow:**
```python
predict_ch8('time traveller ', 10, net, vocab)
```

## 8.5.5 Отсечение градиентов

Для последовательности длины $T$ мы вычисляем градиенты на всех $T$ временных шагах; при обратном распространении возникает цепочка матричных умножений длины $\mathcal{O}(T)$.
Как описано в разделе 4.8, при большом $T$ это может привести к численной неустойчивости, например к взрыву или затуханию градиентов.
Поэтому модели рекуррентных сетей часто требуют дополнительных приёмов, которые помогают стабилизировать обучение.

В общем случае при решении задачи оптимизации мы обновляем параметры модели.
Пусть $\mathbf{x}$ — вектор параметров, а $\mathbf{g}$ — градиент для минибатча, взятый с отрицательным знаком.
Если скорость обучения равна $\eta > 0$, то за одну итерацию $\mathbf{x}$ обновляется до $\mathbf{x} - \eta \mathbf{g}$.
Допустим дополнительно, что целевая функция $f$ хорошо себя ведёт: она липшицево непрерывна с постоянной $L$ (Lipschitz continuous).
То есть для любых $\mathbf{x}$ и $\mathbf{y}$ выполнено:

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

В этом случае можно уверенно утверждать, что если обновить вектор параметров на $\eta \mathbf{g}$, то

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

то есть значение функции не изменится больше чем на $L \eta \|\mathbf{g}\|$.
Это одновременно и плохо, и хорошо.
Плохо, что такая оценка ограничивает скорость продвижения; хорошо, что она ограничивает масштаб ухудшения, особенно когда мы движемся в неверном направлении.

Иногда градиенты могут быть очень большими, и тогда алгоритм оптимизации может не сойтись.
Можно решить эту проблему, уменьшив скорость обучения $\eta$.
Но что делать, если большие градиенты встречаются редко? Тогда уменьшать скорость обучения не имеет смысла.
Распространённая альтернатива — *отсечь* градиент $\mathbf{g}$, спроецировав его обратно на шар заданного радиуса $\theta$:

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

Так мы гарантируем, что норма градиента никогда не превысит $\theta$, а обновлённый градиент будет точно сонаправлен с исходным $\mathbf{g}$.
Есть и полезный побочный эффект: отсечение ограничивает влияние любого отдельного минибатча (и любого отдельного примера в нём) на вектор параметров, что придаёт модели определённую устойчивость.
Отсечение градиентов — быстрый способ справиться со взрывом градиентов; оно не решает проблему полностью, но входит в число эффективных приёмов.

Ниже мы определяем функцию, которая отсекает градиенты модели; модель может быть реализована с нуля или построена с помощью высокоуровневого API.
Мы вычисляем норму градиентов по всем параметрам модели.

**MXNet:**
```python
def grad_clipping(net, theta):  #@save
    """Отсечение градиентов"""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

**PyTorch:**
```python
def grad_clipping(net, theta):  #@save
    """Отсечение градиентов"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

**TensorFlow:**
```python
def grad_clipping(grads, theta):  #@save
    """Отсечение градиентов"""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad
```

**PaddlePaddle:**
```python
def grad_clipping(net, theta):  #@save
    """Отсечение градиентов"""
    if isinstance(net, nn.Layer):
        params = [p for p in net.parameters() if not p.stop_gradient]
    else:
        params = net.params
    norm = paddle.sqrt(sum(paddle.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        with paddle.no_grad():
            for param in params:
                param.grad.set_value(param.grad * theta / norm)
```

## 8.5.6 Обучение

Прежде чем обучать модель, **определим функцию, которая обучает модель в течение одной эпохи**.
Она отличается от функции обучения модели из раздела 3.6 тремя моментами.

1. Разные способы выборки последовательных данных (случайная выборка и последовательное разбиение) приводят к разной инициализации скрытого состояния.
1. Мы отсекаем градиенты перед обновлением параметров модели.
   Это гарантирует, что даже если в какой-то точке обучения произошёл взрыв градиента, модель не расходится.
1. Модель оцениваем по перплексии, как описано в разделе 8.4.4.
   Такая мера гарантирует, что последовательности разной длины можно сравнивать.

Конкретно, при последовательном разбиении мы инициализируем скрытое состояние только в начале каждой эпохи.
Поскольку $i$-й пример следующего минибатча является соседним с $i$-м примером текущего минибатча, скрытое состояние, полученное в последнем примере текущего минибатча, используется для инициализации скрытого состояния первого примера следующего минибатча.
Так информация об истории последовательности, сохранённая в скрытом состоянии, может передаваться между соседними подпоследовательностями в пределах одной эпохи.
Однако вычисление скрытого состояния в любой точке зависит от всех предыдущих минибатчей той же эпохи, и это усложняет вычисление градиентов.
Чтобы уменьшить объём вычислений, перед обработкой каждого минибатча мы отсоединяем (detach) скрытое состояние от графа вычислений, так что вычисление градиента скрытого состояния ограничено временными шагами одного минибатча.

При случайной выборке каждый пример берётся из случайной позиции, поэтому скрытое состояние нужно переинициализировать для каждой эпохи.
Так же, как функция `train_epoch_ch3` из раздела 3.6, `updater` — это общая функция обновления параметров модели.
Это может быть функция `d2l.sgd`, реализованная с нуля, или встроенная оптимизационная функция фреймворка глубокого обучения.

**MXNet:**
```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Обучает модель в течение одной эпохи (определение см. в главе 8)"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Сумма потерь на обучении, количество токенов
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Инициализируем state на первой итерации или при случайной выборке
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Поскольку функция mean уже вызвана
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

**PyTorch:**
```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Обучает сеть в течение одной эпохи (определение см. в главе 8)"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Сумма потерь на обучении, количество токенов
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Инициализируем state на первой итерации или при случайной выборке
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # Для nn.GRU state — это тензор
                state.detach_()
            else:
                # Для nn.LSTM и модели, написанной с нуля, state состоит из тензоров
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Поскольку функция mean уже вызвана
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

**TensorFlow:**
```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """Обучает модель в течение одной эпохи (определение см. в главе 8)"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Сумма потерь на обучении, количество токенов
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Инициализируем state на первой итерации или при случайной выборке
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))
        # Keras по умолчанию возвращает среднюю потерю по батчу
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

**PaddlePaddle:**
```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Обучает сеть в течение одной эпохи (определение см. в главе 8)"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Сумма потерь на обучении, количество токенов
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Инициализируем state на первой итерации или при случайной выборке
            state = net.begin_state(batch_size=X.shape[0])
        else:
            if isinstance(net, nn.Layer) and not isinstance(state, tuple):
                # Для nn.GRU state — это тензор
                state.stop_gradient=True
            else:
                # Для nn.LSTM и модели, написанной с нуля, state состоит из тензоров
                for s in state:
                    s.stop_gradient=True
        y = paddle.reshape(Y.T,shape=[-1])
        X = paddle.to_tensor(X, place=device)
        y = paddle.to_tensor(y, place=device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()
        if isinstance(updater, paddle.optimizer.Optimizer):
            updater.clear_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Поскольку функция mean уже вызвана
            updater(batch_size=1)
        
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

**Функция обучения модели рекуррентной нейронной сети поддерживает как реализацию с нуля, так и высокоуровневый API.**

**MXNet:**
```python
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """Обучает модель (определение см. в главе 8)"""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='эпоха', ylabel='перплексия',
                            legend=['обучение'], xlim=[10, num_epochs])
    # Инициализация
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Обучение и предсказание
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'перплексия {ppl:.1f}, {speed:.1f} токенов/с {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

**PyTorch:**
```python
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Обучает модель (определение см. в главе 8)"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='эпоха', ylabel='перплексия',
                            legend=['обучение'], xlim=[10, num_epochs])
    # Инициализация
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Обучение и предсказание
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'перплексия {ppl:.1f}, {speed:.1f} токенов/с {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

**TensorFlow:**
```python
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy,
              use_random_iter=False):
    """Обучает модель (определение см. в главе 8)"""
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='эпоха', ylabel='перплексия',
                            legend=['обучение'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # Обучение и предсказание
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'перплексия {ppl:.1f}, {speed:.1f} токенов/с {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

**PaddlePaddle:**
```python
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """Обучает модель (определение см. в главе 8)"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='эпоха', ylabel='перплексия',
                            legend=['обучение'], xlim=[10, num_epochs])
    # Инициализация
    if isinstance(net, nn.Layer):
        updater = paddle.optimizer.SGD(
                learning_rate=lr, parameters=net.parameters())
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Обучение и предсказание
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'перплексия {ppl:.1f}, {speed:.1f} токенов/с {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

**Теперь обучим модель рекуррентной нейронной сети.**
Поскольку в наборе данных используется только 10 000 токенов, модели требуется больше эпох, чтобы хорошо сойтись.

**MXNet, PyTorch, PaddlePaddle:**
```python
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

**TensorFlow:**
```python
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

**Наконец, посмотрим, что получится при случайной выборке.**

**MXNet, PyTorch:**
```python
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

**TensorFlow:**
```python
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
train_ch8(net, train_iter, vocab_random_iter, lr, num_epochs, strategy,
          use_random_iter=True)
```

**PaddlePaddle:**
```python
net = RNNModelScratch(len(vocab), num_hiddens, get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

Реализация описанной модели рекуррентной нейронной сети с нуля, хотя и поучительна, неудобна.
В следующем разделе мы узнаем, как улучшить модель: например, как сделать её реализацию проще и быстрее.

## Резюме

* Мы можем обучить символьную языковую модель на основе рекуррентной нейронной сети, которая генерирует продолжение текста по заданному пользователем префиксу.
* Простая языковая модель на основе рекуррентной нейронной сети состоит из кодирования входа, рекуррентной нейронной сети и генерации выхода.
* Перед обучением модели рекуррентной нейронной сети нужно инициализировать её состояние, причём при случайной выборке и последовательном разбиении используются разные способы инициализации.
* При последовательном разбиении нужно отделять градиенты, чтобы сократить объём вычислений.
* Перед любыми предсказаниями модель обновляет себя в период разогрева (например, получает скрытое состояние лучше начального).
* Отсечение градиентов помогает предотвратить взрыв градиентов, но не справляется с затуханием градиентов.

## Упражнения

1. Покажите, что one-hot кодирование эквивалентно выбору отдельного вектора внедрения (embedding) для каждого объекта.
1. Улучшите перплексию, подбирая гиперпараметры (число эпох, число скрытых элементов, число временных шагов в минибатче, скорость обучения и т. д.).
    * До какого значения можно снизить перплексию?
    * Улучшит ли результат замена one-hot кодирования обучаемыми векторами внедрения?
    * Как будет работать модель, если в качестве набора данных взять другую книгу Герберта Уэллса, например [«Война миров»](http://www.gutenberg.org/ebooks/36)?
1. Измените функцию предсказания так, чтобы вместо выбора наиболее вероятного следующего символа использовалась выборка.
    * Что произойдёт?
    * Настройте модель так, чтобы она отдавала предпочтение более вероятным вариантам: например, при $\alpha > 1$ выполняйте выборку из $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$.
1. Что произойдёт, если запустить код этого раздела без отсечения градиентов?
1. Измените последовательное разбиение так, чтобы скрытое состояние не отделялось от вычислительного графа. Изменится ли время работы? А перплексия?
1. Замените функцию активации на ReLU и повторите эксперименты этого раздела. Нужно ли по-прежнему отсекать градиенты? Почему?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2102)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2103)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/2104)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11799)
