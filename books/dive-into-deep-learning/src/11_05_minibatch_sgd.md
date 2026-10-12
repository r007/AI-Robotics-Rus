# 11.5 Минибатчевый стохастический градиентный спуск

До сих пор в градиентных методах обучения мы встречались с двумя крайними случаями:
в разделе 11.3 для вычисления градиента и обновления параметров использовался полный набор данных,
а в разделе 11.4 на каждом шаге обрабатывался один обучающий пример.
У обоих подходов есть свои недостатки: градиентный спуск не слишком «эффективен по данным», когда данные очень похожи друг на друга.
А стохастический градиентный спуск не особенно «эффективен по вычислениям», поскольку процессоры CPU и GPU не могут полностью использовать векторизацию.
Это подсказывает, что между ними может существовать компромисс — и это и есть *минибатчевый стохастический градиентный спуск* (minibatch gradient descent).

## 11.5.1 Векторизация и кэш

Ключевой аргумент в пользу минибатчей — вычислительная эффективность.
Это особенно наглядно видно при параллельных вычислениях на нескольких GPU и нескольких серверах.
В этом случае каждому GPU нужно передать хотя бы одно изображение.
При 8 GPU на каждом сервере и 16 серверах мы получаем минибатч размером 128.

Когда речь идёт об одном GPU или даже CPU, всё устроено тоньше:
у этих устройств есть несколько типов памяти, обычно несколько типов вычислительных блоков и разные ограничения пропускной способности между ними.
Например, у CPU есть небольшое число регистров, кэши L1 и L2, а также кэш L3, общий для разных ядер процессора.
С ростом объёма кэша растёт и его задержка, а пропускная способность падает.
Можно сказать, что процессор способен выполнять гораздо больше операций, чем может обеспечить интерфейс основной памяти.

Во-первых, 2-гигагерцовый CPU с 16 ядрами и векторизацией AVX-512 может обработать до $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ байт в секунду.
В то же время производительность GPU легко превышает эту величину в 100 раз.
С другой стороны, пропускная способность типичного серверного процессора среднего класса вряд ли превышает 100 Гбит/с, то есть меньше одной десятой того, что требуется для полной загрузки процессора.
Более того, не все обращения к памяти одинаково выгодны: интерфейс памяти обычно имеет ширину 64 бита или больше (например, у GPU — до 384 бит).
Поэтому чтение одного байта приводит к накладным расходам, поскольку фактически читается более широкий блок.

Во-вторых, первое обращение к памяти сопряжено со значительными накладными расходами, тогда как последовательный доступ (sequential access) или пакетное чтение (burst read) обходится относительно дёшево.
Подробнее об этом можно прочитать в [статье Википедии об иерархии кэшей](https://en.wikipedia.org/wiki/Cache_hierarchy).

Смягчить эти ограничения позволяет достаточно быстрая иерархия кэшей процессора, которая снабжает процессор данными.
Именно эта идея стоит за пакетной обработкой в глубоком обучении.
Простой пример — умножение матриц.
Пусть $\mathbf{A} = \mathbf{B}\mathbf{C}$. Существует много способов вычислить $\mathbf{A}$. Например, можно попробовать следующие:

1. Вычислить $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$, то есть поэлементно через скалярные произведения.
1. Вычислить $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$, то есть по одному столбцу за раз. Аналогично можно вычислять $\mathbf{A}$ по одной строке $\mathbf{A}_{i,:}$.
1. Просто вычислить $\mathbf{A} = \mathbf{B} \mathbf{C}$.
1. Разбить $\mathbf{B}$ и $\mathbf{C}$ на блочные матрицы меньшего размера и вычислять $\mathbf{A}$ по одному блоку за раз.

Если использовать первый вариант, то при каждом вычислении элемента $\mathbf{A}_{ij}$ строку и столбец нужно копировать в CPU.
Ещё хуже то, что поскольку элементы матрицы выровнены последовательно, при чтении из памяти приходится обращаться ко многим несмежным позициям в обоих векторах.
Второй вариант относительно выгоднее: мы можем держать столбец $\mathbf{C}_{:,j}$ в кэше CPU, пока проходим по $\mathbf{B}$.
Это вдвое снижает требования к пропускной способности памяти и соответственно ускоряет доступ.
Третий вариант на первый взгляд самый предпочтительный, однако большинство матриц, скорее всего, целиком не поместятся в кэш.
Четвёртый вариант даёт практически полезное решение: можно переносить в кэш блоки матриц и перемножать их локально.
Посмотрим, насколько эффективны эти операции на практике.

Помимо эффективности вычислений, существенные накладные расходы создают сам Python и фреймворк глубокого обучения.
Напомним, что каждый раз, когда мы выполняем код, интерпретатор Python отправляет фреймворку глубокого обучения команду, которая должна добавить операцию в вычислительный граф и обработать её при планировании выполнения.
Такие накладные расходы могут быть весьма ощутимыми.
Итак, лучше всего использовать векторизацию (и матрицы).

**MXNet:**

```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

**PyTorch:**

```python
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

**TensorFlow:**

```python
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

**PaddlePaddle:**

```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import numpy as np

timer = d2l.Timer()
A = d2l.zeros((256, 256))
B = d2l.randn((256, 256))
C = d2l.randn((256, 256))
```

Поэлементное присваивание требует обойти все строки и столбцы $\mathbf{B}$ и $\mathbf{C}$, чтобы присвоить значение элементу $\mathbf{A}$.

**MXNet:**

```python
# Вычисление A=BC поэлементно
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

**PyTorch:**

```python
# Вычисление A=BC поэлементно
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

**TensorFlow:**

```python
# Вычисление A=BC поэлементно
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

**PaddlePaddle:**

```python
# Вычисление A=BC поэлементно
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = paddle.dot(B[i, :], C[:, j])
timer.stop()
```

Более быстрая стратегия — вычислять по столбцам.

**MXNet:**

```python
# Вычисление A=BC по столбцам
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

**PyTorch:**

```python
# Вычисление A=BC по столбцам
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

**TensorFlow:**

```python
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

**PaddlePaddle:**

```python
# Вычисление A=BC по столбцам
timer.start()
for j in range(256):
    A[:, j] = paddle.mv(B, C[:, j])
timer.stop()
```

Наиболее эффективный способ — выполнить всю операцию одним блоком. Посмотрим, с какой скоростью работает каждый из вариантов.

**MXNet:**

```python
# Вычисление A=BC за один раз
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Умножение и сложение как отдельные операции (на практике объединяются)
gigaflops = [2/i for i in timer.times]
print(f'производительность в гигафлопсах: поэлементно {gigaflops[0]:.3f}, '
      f'по столбцам {gigaflops[1]:.3f}, полностью {gigaflops[2]:.3f}')
```

**PyTorch:**

```python
# Вычисление A=BC за один раз
timer.start()
A = torch.mm(B, C)
timer.stop()

# Умножение и сложение как отдельные операции (на практике объединяются)
gigaflops = [2/i for i in timer.times]
print(f'производительность в гигафлопсах: поэлементно {gigaflops[0]:.3f}, '
      f'по столбцам {gigaflops[1]:.3f}, полностью {gigaflops[2]:.3f}')
```

**TensorFlow:**

```python
# Вычисление A=BC за один раз
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Умножение и сложение как отдельные операции (на практике объединяются)
gigaflops = [2/i for i in timer.times]
print(f'производительность в гигафлопсах: поэлементно {gigaflops[0]:.3f}, '
      f'по столбцам {gigaflops[1]:.3f}, полностью {gigaflops[2]:.3f}')
```

**PaddlePaddle:**

```python
# Вычисление A=BC за один раз
timer.start()
A = paddle.mm(B, C)
timer.stop()

# Умножение и сложение как отдельные операции (на практике объединяются)
gigaflops = [2/i for i in timer.times]
print(f'производительность в гигафлопсах: поэлементно {gigaflops[0]:.3f}, '
      f'по столбцам {gigaflops[1]:.3f}, полностью {gigaflops[2]:.3f}')
```

## 11.5.2 Минибатчи


Ранее мы без раздумий считывали *минибатчи* данных, а не отдельные наблюдения, чтобы обновлять параметры. Теперь кратко объясним причины.
Обработка одного наблюдения требует выполнения множества отдельных умножений матрицы на вектор (или даже вектора на вектор), что довольно затратно и сопряжено с большими накладными расходами со стороны фреймворка.
Это справедливо как для вычисления градиентов при обновлении параметров, так и для предсказания с помощью нейронной сети.
То есть каждое выполнение $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ обходится дорого, где

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w}).$$

Мы можем повысить *вычислительную* эффективность этой операции, применив её к минибатчу наблюдений.
То есть вместо градиента $\mathbf{g}_t$ для одного наблюдения мы используем градиент для минибатча:

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}).$$

Посмотрим, как это влияет на статистические свойства $\mathbf{g}_t$. Поскольку $\mathbf{x}_t$ и все элементы минибатча $\mathcal{B}_t$ выбираются случайно из обучающего набора, математическое ожидание градиента остаётся неизменным.
С другой стороны, дисперсия существенно уменьшается.
Поскольку градиент минибатча состоит из $b := |\mathcal{B}_t|$ независимых градиентов, которые усредняются, его стандартное отклонение уменьшается в $b^{-\frac{1}{2}}$ раз.
Это само по себе хорошо, поскольку обновление становится ближе к полному градиенту.

Интуитивно понятно, что выбор большого минибатча $\mathcal{B}_t$ обычно оправдан.
Однако через некоторое время дополнительное уменьшение стандартного отклонения становится незначительным по сравнению с линейным ростом вычислительных затрат.
На практике мы выбираем минибатч достаточно большим, чтобы обеспечить хорошую вычислительную эффективность, но при этом он всё ещё помещался в память GPU.
Далее рассмотрим код, реализующий эту идею.
Мы выполним то же самое умножение матриц, но теперь разобьём его на «минибатчи» по 64 столбца.

```python
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'производительность в гигафлопсах: блоками {2 / timer.times[3]:.3f}')
```

**PyTorch:**

```python
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'производительность в гигафлопсах: блоками {2 / timer.times[3]:.3f}')
```

**TensorFlow:**

```python
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'производительность в гигафлопсах: блоками {2 / timer.times[3]:.3f}')
```

**PaddlePaddle:**

```python
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = paddle.mm(B, C[:, j:j+64])
timer.stop()
print(f'производительность в гигафлопсах: блоками {2 / timer.times[3]:.3f}')
```

Очевидно, что вычисления на минибатчах практически так же эффективны, как и на полной матрице.
Стоит отметить, что в разделе 7.5 мы использовали вид регуляризации, который в значительной степени зависит от дисперсии внутри минибатча.
По мере её роста дисперсия уменьшается, и вместе с этим исчезает польза от шума, который вносит пакетная нормализация.
Подробности о том, как пересчитывать масштаб и вычислять соответствующие величины, см. в работе (Ioffe, 2017).

## 11.5.3 Чтение набора данных

Посмотрим, как эффективно генерировать минибатчи из данных.
Ниже для сравнения алгоритмов оптимизации мы используем набор данных о шуме, который создают различные летательные аппараты и который измерило НАСА на испытательных аэродинамических профилях крыла (airfoil self-noise dataset).
Для удобства возьмём только первые $1\,500$ примеров.
Данные уже предобработаны: мы убрали среднее значение и масштабировали дисперсию так, что для каждой координаты она равна $1$.

**MXNet:**

```python
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

**PyTorch:**

```python
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

**TensorFlow:**

```python
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

**PaddlePaddle:**

```python
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = d2l.tensor((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 11.5.4 Реализация с нуля

Минибатчевый стохастический градиентный спуск уже был реализован в разделе 3.2.
Здесь мы сделаем его входные параметры более общими, главным образом для того, чтобы другие алгоритмы оптимизации, описанные в этой главе, могли использовать те же входы.
А именно, мы добавляем входное состояние `states` и помещаем гиперпараметры в словарь `hyperparams`.
Кроме того, мы будем усреднять потери по отдельным минибатчам в функции обучения, поэтому градиенты в алгоритме оптимизации не нужно делить на размер батча.

**MXNet:**

```python
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

**PyTorch:**

```python
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

**TensorFlow:**

```python
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

**PaddlePaddle:**

```python
def sgd(params, states, hyperparams):
    a = []
    with paddle.no_grad():
        for p in params:
            p = p - hyperparams['lr'] * p.grad
            p.stop_gradient = False
            a.append(p)
        return a
```

Ниже реализуем общую функцию обучения, которую будем использовать для других алгоритмов оптимизации в этой главе.
Она инициализирует модель линейной регрессии, а затем позволяет обучать модель с помощью минибатчевого стохастического градиентного спуска и других алгоритмов, которые описаны в последующих подразделах.

**MXNet:**

```python
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Инициализация модели
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Обучение модели
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
    return timer.cumsum(), animator.Y[0]
```

**PyTorch:**

```python
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Инициализация модели
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Обучение модели
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
    return timer.cumsum(), animator.Y[0]
```

**TensorFlow:**

```python
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Инициализация модели
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Обучение модели
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
    return timer.cumsum(), animator.Y[0]
```

**PaddlePaddle:**

```python
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Инициализация модели
    w = d2l.tensor(d2l.normal(mean=0.0, std=0.01, shape=(feature_dim, 1)), stop_gradient=False)
    b = d2l.tensor(d2l.zeros((1,)), stop_gradient=False)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Обучение модели
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            w, b = trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
    return timer.cumsum(), animator.Y[0]
```

Посмотрим, как проходит оптимизация пакетным градиентным спуском.
Этого можно добиться, установив размер минибатча равным 1500 (то есть общему числу примеров).
Тогда параметры модели обновляются один раз за эпоху.

```python
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Когда размер минибатча равен 1, оптимизация использует стохастический градиентный спуск.
Для упрощения реализации мы выбрали очень маленькую скорость обучения.
В эксперименте со стохастическим градиентным спуском параметры модели обновляются каждый раз, когда обработан один пример.
В данном случае это соответствует 1500 обновлениям за эпоху.
Видно, что снижение значения целевой функции становится довольно пологим уже после одной эпохи.
Хотя оба примера обрабатывают 1500 примеров за одну эпоху, стохастическому градиентному спуску требуется больше времени на эпоху.
Это объясняется тем, что стохастический градиентный спуск чаще обновляет параметры и обрабатывать отдельные наблюдения менее эффективно.

```python
sgd_res = train_sgd(0.005, 1)
```

Наконец, при размере минибатча 100 мы используем минибатчевый стохастический градиентный спуск.
Время на эпоху здесь меньше, чем у стохастического и пакетного градиентного спуска.

```python
mini1_res = train_sgd(.4, 100)
```

Если уменьшить размер минибатча до 10, время на эпоху снова вырастет, поскольку каждая порция работы выполняется менее эффективно.

```python
mini2_res = train_sgd(.05, 10)
```

Теперь сравним время и потери для первых четырёх экспериментов.
Видно, что хотя по числу обработанных примеров стохастический градиентный спуск сходится быстрее градиентного спуска, для достижения того же значения потерь ему требуется больше времени, поскольку вычисление градиента по одному примеру не так эффективно.
Минибатчевый стохастический градиентный спуск позволяет сбалансировать скорость сходимости и вычислительную эффективность.
Минибатч размером 10 эффективнее стохастического градиентного спуска;
минибатч размером 100 даже по времени выполнения опережает градиентный спуск.

```python
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'время (сек)', 'потери', xlim=[1e-2, 10],
         legend=['градиентный спуск', 'стохастический градиентный спуск',
                 'размер батча=100', 'размер батча=10'])
d2l.plt.gca().set_xscale('log')
```

## 11.5.5 Краткая реализация

Ниже мы используем алгоритмы, встроенные в фреймворк глубокого обучения, чтобы реализовать общую функцию обучения; её мы будем использовать в других подразделах этой главы.

**MXNet:**

```python
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Инициализация модели
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
```

**PyTorch:**

```python
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Инициализация модели
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss вычисляет квадратичную ошибку без множителя 1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
```

**TensorFlow:**

```python
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Инициализация модели
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # MeanSquaredError вычисляет квадратичную ошибку без множителя 1/2
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
```

**PaddlePaddle:**

```python
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Инициализация модели
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            paddle.nn.initializer.Normal(m.weight, std=0.01)

    net.apply(init_weights)

    optimizer = trainer_fn(parameters=net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.clear_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss вычисляет квадратичную ошибку без множителя 1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'потери: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} сек/эпоху')
```

Теперь используем эту функцию обучения, чтобы повторить предыдущие эксперименты.

**MXNet:**

```python
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

**PyTorch:**

```python
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

**TensorFlow:**

```python
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

**PaddlePaddle:**

```python
data_iter, _ = get_data_ch11(10)
trainer = paddle.optimizer.SGD
train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Резюме

* Векторизация делает код более эффективным благодаря сокращению накладных расходов фреймворка глубокого обучения, лучшему размещению данных в памяти и использованию кэшей CPU и GPU.
* Существует компромисс между «статистической эффективностью» стохастического градиентного спуска и «вычислительной эффективностью» пакетной обработки больших объёмов данных сразу. Минибатчевый стохастический градиентный спуск даёт то и другое: вычислительную и статистическую эффективность.
* В минибатчевом стохастическом градиентном спуске мы обрабатываем батчи, полученные из случайной перестановки обучающих данных (то есть каждое наблюдение обрабатывается один раз, но в случайном порядке).
* Снижение скорости обучения в ходе обучения помогает обучению.
* В целом минибатчевый стохастический градиентный спуск быстрее стохастического и обычного градиентного спуска и менее рискован с точки зрения сходимости.

## Упражнения

1. Измените размер минибатча и скорость обучения и наблюдайте за скоростью снижения значения целевой функции и временем, затрачиваемым на эпоху.
1. Сравните минибатчевый стохастический градиентный спуск с вариантом, который выбирает примеры из обучающего набора *с возвращением*. Что вы заметите?
1. Злой дух скопировал ваш набор данных без вашего ведома (то есть каждое наблюдение встречается дважды, и набор данных вырос до удвоенного исходного размера, но никто вам об этом не сказал). Как изменится поведение стохастического градиентного спуска, минибатчевого стохастического градиентного спуска и градиентного спуска?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/4324)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/4325)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/4326)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11850)

## Литература

* Ioffe, S. (2017). Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. *Advances in Neural Information Processing Systems*, 1945–1953.
