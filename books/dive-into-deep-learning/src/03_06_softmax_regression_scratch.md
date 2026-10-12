# 3.6 Softmax-регрессия с нуля

Так же как мы реализовали линейную регрессию с нуля, softmax-регрессия тоже является важной основой, поэтому **полезно знать детали её реализации**.
В этом разделе мы используем набор Fashion-MNIST, который ввели в разделе 3.5, и зададим размер батча для итератора данных равным 256.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
from IPython import display
```

**TensorFlow:**

```python
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from IPython import display
```

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 3.6.1 Инициализация параметров модели

Как и в примере с линейной регрессией, каждый образец здесь представляется вектором фиксированной длины.
Каждое исходное изображение из набора данных имеет размер $28 \times 28$.
В этом разделе **мы разворачиваем каждое изображение в вектор длины 784**.
В следующих главах мы поговорим о признаках, которые учитывают пространственную структуру изображения, а пока будем рассматривать каждую позицию пикселя как отдельный признак.

Напомним, что в softmax-регрессии число выходов равно числу классов.
**Поскольку в нашем наборе данных 10 классов, размерность выхода сети равна 10**.
Поэтому веса образуют матрицу размера $784 \times 10$, а смещение — строку размера $1 \times 10$.
Как и в линейной регрессии, веса `W` инициализируем из нормального распределения, а смещение `b` — нулями.

**MXNet:**

```python
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

**PyTorch:**

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

**TensorFlow:**

```python
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

**PaddlePaddle:**

```python
num_inputs = 784
num_outputs = 10

W = paddle.normal(0, 0.01, shape=(num_inputs, num_outputs))
b = paddle.zeros(shape=(num_outputs,))
W.stop_gradient=False
b.stop_gradient=False
```

## 3.6.2 Определение операции softmax

Прежде чем реализовать модель softmax-регрессии, кратко напомним, как оператор `sum` работает вдоль заданного измерения тензора.
Как описано в разделах 2.3.6 и 2.3.6.1, **для матрицы `X` можно просуммировать все элементы** (по умолчанию).
Также можно суммировать только по одной оси: по столбцам (ось 0) или по строкам (ось 1).
Если `X` — тензор формы `(2, 3)` и мы суммируем по столбцам, то результатом будет вектор формы `(3,)`.
При вызове оператора `sum` можно указать, что нужно сохранить количество осей исходного тензора и не схлопывать суммируемое измерение.
Тогда получится двумерный тензор формы `(1, 3)`.

**PyTorch, PaddlePaddle:**

```python
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

**MXNet, TensorFlow:**

```python
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Напомним, что **реализация softmax** состоит из трёх шагов:

1. возведение каждого элемента в экспоненту (с помощью `exp`);
1. суммирование по каждой строке (каждый образец в минибатче — это строка), что даёт нормирующую константу для каждого образца;
1. деление каждой строки на её нормирующую константу, чтобы сумма элементов строки стала равна 1.

Прежде чем смотреть на код, запишем формулу:

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$

Знаменатель, или нормирующая константа, иногда называется *статистической суммой* (partition function), а её логарифм — логарифмом статистической суммы.
Это название пришло из [статистической физики](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)), где такое уравнение описывает распределение совокупности частиц.

**MXNet, TensorFlow:**

```python
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # Здесь используется broadcasting
```

**PyTorch, PaddlePaddle:**

```python
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # Здесь используется broadcasting
```

Как видно из кода, **для любого случайного входа каждый элемент становится неотрицательным числом.
Кроме того, по законам теории вероятностей сумма каждой строки равна 1**.

**MXNet, PyTorch, PaddlePaddle:**

```python
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

**TensorFlow:**

```python
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

Заметим, что хотя математически всё выглядит корректно, наша реализация недостаточно аккуратна.
Очень большие или очень маленькие элементы матрицы могут вызвать численное переполнение или антипереполнение (underflow), но мы никак этого не предотвращаем.

## 3.6.3 Определение модели

После определения операции softmax мы можем **реализовать модель softmax-регрессии**.
В приведённом ниже коде описано, как входные данные отображаются в выход сети.
Обратите внимание: перед передачей данных в модель мы используем функцию `reshape`, чтобы развернуть каждое исходное изображение в вектор.

```python
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## 3.6.4 Определение функции потерь

Далее реализуем функцию кросс-энтропийной потери, введённую в разделе 3.4.
Это, вероятно, самая распространённая функция потерь в глубоком обучении, поскольку задач классификации гораздо больше, чем задач регрессии.

Напомним, что кросс-энтропия — это отрицательный логарифм предсказанной вероятности истинной метки.
Вместо того чтобы перебирать предсказания циклом `for` в Python (что обычно неэффективно), мы выберем нужные элементы с помощью операции индексирования.
Ниже **создадим данные `y_hat` с предсказанными вероятностями для 2 образцов в 3 классах и их метки `y`.**
По `y` видно, что для первого образца правильным является первый класс, а для второго — третий.
**Используя `y` как индексы для `y_hat`**, выберем вероятность первого класса для первого образца и вероятность третьего класса для второго образца.

**MXNet, PyTorch, PaddlePaddle:**

```python
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

**TensorFlow:**

```python
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Теперь **реализация кросс-энтропийной потери** занимает всего одну строку кода.

**MXNet, PyTorch:**

```python
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

**TensorFlow:**

```python
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

**PaddlePaddle:**

```python
def cross_entropy(y_hat, y):
    return - paddle.log(y_hat[[i for i in range(len(y_hat))], y.squeeze()])

cross_entropy(y_hat, y)
```

## 3.6.5 Классификационная точность

Для заданного распределения вероятностей `y_hat`, когда нужно выдать жёсткое предсказание (hard prediction), обычно выбирают класс с наибольшей вероятностью.
Многие приложения требуют именно такого выбора.
Например, Gmail должен отнести письмо к одной из категорий: «Основные», «Социальные», «Обновления» или «Форумы».
Внутри Gmail, возможно, оценивает вероятности, но в итоге ему всё равно нужно выбрать один класс.

Предсказание считается правильным, если оно совпадает с меткой `y`.
*Классификационная точность* (accuracy) — это отношение числа правильных предсказаний к общему числу предсказаний.
Напрямую оптимизировать точность может быть трудно, потому что она не дифференцируема.
Тем не менее точность — это обычно та мера качества, которая нас интересует больше всего, и при обучении классификаторов за ней почти всегда следят.

Чтобы вычислить точность, выполним следующие действия.
Во-первых, если `y_hat` — матрица, то предполагаем, что во втором измерении хранятся оценки каждого класса.
Мы используем `argmax`, чтобы получить индекс максимального элемента в каждой строке и определить предсказанный класс.
Затем **сравниваем предсказанные классы с истинными значениями `y`**.
Поскольку оператор равенства `==` чувствителен к типу данных, приводим тип `y_hat` к типу `y`.
В результате получаем тензор из нулей (неверно) и единиц (верно).
Наконец, суммируем его и получаем число правильных предсказаний.

```python
def accuracy(y_hat, y):  #@save
    """Вычисляет количество правильных предсказаний"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

Ниже приведена версия для PaddlePaddle.

**PaddlePaddle:**

```python
#@save
def accuracy(y_hat, y):
    """Вычисляет количество правильных предсказаний"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    if len(y_hat.shape) < len(y.shape):
        cmp = y_hat.astype(y.dtype) == y.squeeze()
    else:
        cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum())
```

Продолжим использовать переменные `y_hat` и `y`, определённые выше, как предсказанное распределение вероятностей и метки.
Первый образец получает предсказанный класс 2 (максимальный элемент строки равен 0.6 и стоит на индексе 2), что не совпадает с истинной меткой 0.
Второй образец получает предсказанный класс 2 (максимальный элемент строки равен 0.5 и стоит на индексе 2), что совпадает с истинной меткой 2.
Итого классификационная точность для этих двух образцов равна 0.5.

```python
accuracy(y_hat, y) / len(y)
```

Аналогично, **для любого набора данных, доступного через итератор `data_iter`, можно вычислить точность произвольной модели `net`**.

**MXNet, TensorFlow:**

```python
def evaluate_accuracy(net, data_iter):  #@save
    """Вычисляет точность модели на заданном наборе данных"""
    metric = Accumulator(2)  # Количество правильных предсказаний, общее количество предсказаний
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

**PyTorch:**

```python
def evaluate_accuracy(net, data_iter):  #@save
    """Вычисляет точность модели на заданном наборе данных"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Переводим модель в режим оценки
    metric = Accumulator(2)  # Количество правильных предсказаний, общее количество предсказаний
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

**PaddlePaddle:**

```python
#@save
def evaluate_accuracy(net, data_iter):
    """Вычисляет точность модели на заданном наборе данных"""
    if isinstance(net, paddle.nn.Layer):
        net.eval()  # Переводим модель в режим оценки
    metric = Accumulator(2)  # Количество правильных предсказаний, общее количество предсказаний
    with paddle.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Здесь мы определяем вспомогательный класс `Accumulator`, который накапливает суммы нескольких переменных.
В функции `evaluate_accuracy` **в экземпляре `Accumulator` создаются 2 переменные**: одна хранит количество правильных предсказаний, другая — общее количество предсказаний.
Обе переменные накапливаются по мере прохода по набору данных.

```python
class Accumulator:  #@save
    """Накопление сумм по n переменным"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

Поскольку модель `net` инициализирована случайными весами, её точность должна быть близка к случайному угадыванию.
Например, при 10 классах точность равна 0.1.

```python
evaluate_accuracy(net, test_iter)
```

## 3.6.6 Обучение

Код обучения softmax-регрессии должен выглядеть очень знакомо, если мы вспомним реализацию линейной регрессии в разделе 3.2.
Здесь мы переработаем процесс обучения так, чтобы его можно было повторно использовать.
Сначала определим функцию, которая обучает модель на одной эпохе.
Обратите внимание, что `updater` — это функция обновления параметров модели, которая принимает размер батча в качестве аргумента.
Это может быть функция `d2l.sgd` или встроенная оптимизационная функция фреймворка.

**MXNet:**

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Обучает модель в течение одной эпохи (определение см. в главе 3)"""
    # Суммарные потери на обучении, суммарная точность на обучении, число образцов
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Вычисляем градиенты и обновляем параметры
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Возвращаем потери на обучении и точность на обучении
    return metric[0] / metric[2], metric[1] / metric[2]
```

**PyTorch:**

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Обучает модель в течение одной эпохи (определение см. в главе 3)"""
    # Переводим модель в режим обучения
    if isinstance(net, torch.nn.Module):
        net.train()
    # Суммарные потери на обучении, суммарная точность на обучении, число образцов
    metric = Accumulator(3)
    for X, y in train_iter:
        # Вычисляем градиенты и обновляем параметры
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Используем встроенный в PyTorch оптимизатор и функцию потерь
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Используем самописные оптимизатор и функцию потерь
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Возвращаем потери на обучении и точность на обучении
    return metric[0] / metric[2], metric[1] / metric[2]
```

**TensorFlow:**

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Обучает модель в течение одной эпохи (определение см. в главе 3)"""
    # Суммарные потери на обучении, суммарная точность на обучении, число образцов
    metric = Accumulator(3)
    for X, y in train_iter:
        # Вычисляем градиенты и обновляем параметры
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Встроенные в Keras функции потерь принимают аргументы в порядке (метки, предсказания),
            # это отличается от реализаций, которые мы используем в этой книге.
            # Здесь реализация принимает (предсказания, метки), например описанная выше «кросс-энтропия»
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Функции потерь Keras по умолчанию возвращают среднюю потерю по батчу
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Возвращаем потери на обучении и точность на обучении
    return metric[0] / metric[2], metric[1] / metric[2]
```

**PaddlePaddle:**

```python
#@save
def train_epoch_ch3(net, train_iter, loss, updater):
    """Обучает модель в течение одной эпохи (определение см. в главе 3)"""
    # Переводим модель в режим обучения
    if isinstance(net, paddle.nn.Layer):
        net.train()
    # Суммарные потери на обучении, суммарная точность на обучении, число образцов
    metric = Accumulator(3)

    for X, y in train_iter:
        # Вычисляем градиенты и обновляем параметры
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, paddle.optimizer.Optimizer):
            # Используем встроенный в PaddlePaddle оптимизатор и функцию потерь
            updater.clear_grad()
            l.mean().backward()
            updater.step()
        else:
            # Используем самописные оптимизатор и функцию потерь
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
```

Прежде чем показать реализацию функции обучения, **определим вспомогательный класс `Animator` для рисования данных в виде анимации**.
Это упростит код в остальных разделах книги.

```python
class Animator:  #@save
    """Рисование данных в анимации"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Поэтапно рисуем несколько линий
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Используем lambda-функцию, чтобы зафиксировать аргументы
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Добавляем на график несколько точек данных
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

Далее реализуем **функцию обучения** `train_ch3`: она обучает модель `net` на данных из `train_iter` и выполняет несколько эпох (число задаётся параметром `num_epochs`).
В конце каждой эпохи модель оценивается на тестовых данных из `test_iter`.
Ход обучения визуализируем с помощью класса `Animator`.

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Обучает модель (определение см. в главе 3)"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

Для реализации с нуля мы используем определённый в разделе 3.2 **мини-батчевый стохастический градиентный спуск для оптимизации функции потерь модели**, скорость обучения установим равной 0.1.

**MXNet, PyTorch, PaddlePaddle:**

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

**TensorFlow:**

```python
class Updater():  #@save
    """Обновление параметров методом мини-батчевого стохастического градиентного спуска"""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Теперь **обучим модель за 10 эпох**.
Обратите внимание: число эпох (`num_epochs`) и скорость обучения (`lr`) — настраиваемые гиперпараметры.
Изменяя их значения, можно повысить точность классификации модели.

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 3.6.7 Предсказание

Теперь, когда обучение завершено, модель готова **предсказывать классы изображений**.
Для набора изображений сравним их истинные метки (первая строка текстового вывода) с предсказаниями модели (вторая строка текстового вывода).

```python
def predict_ch3(net, test_iter, n=6):  #@save
    """Предсказывает метки (определение см. в главе 3)"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Резюме

* С помощью softmax-регрессии можно обучать модели многоклассовой классификации.
* Обучение softmax-регрессии очень похоже на обучение линейной регрессии: сначала читаем данные, затем определяем модель и функцию потерь, после чего обучаем модель с помощью алгоритма оптимизации. Большинство распространённых моделей глубокого обучения имеют похожий процесс обучения.

## Упражнения

1. В этом разделе функция `softmax` реализована напрямую по математическому определению операции softmax. Какие проблемы это может вызвать? Подсказка: попробуйте вычислить величину $\exp(50)$.
1. Функция `cross_entropy` в этом разделе реализована по определению кросс-энтропийной потери. Что в ней может быть не так? Подсказка: подумайте, на какой области определён логарифм.
1. Предложите решения для двух предыдущих проблем.
1. Всегда ли оптимально возвращать класс с наибольшей вероятностью? Например, допустимо ли так поступать в медицинской диагностике?
1. Предположим, мы используем softmax-регрессию, чтобы предсказывать следующее слово. Какие проблемы может создать слишком большое число возможных слов?

* MXNet: [discuss.d2l.ai/t/1791](https://discuss.d2l.ai/t/1791)
* PyTorch: [discuss.d2l.ai/t/1789](https://discuss.d2l.ai/t/1789)
* TensorFlow: [discuss.d2l.ai/t/1790](https://discuss.d2l.ai/t/1790)
* PaddlePaddle: [discuss.d2l.ai/t/11760](https://discuss.d2l.ai/t/11760)
