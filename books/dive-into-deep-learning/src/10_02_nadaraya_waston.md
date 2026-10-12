# 10.2 Агрегация внимания: ядерная регрессия Надарая — Ватсона

В предыдущем разделе мы ввели основные компоненты механизма внимания в рамках схемы из рис. 10.1.3: взаимодействие запроса (произвольная подсказка) и ключей (непроизвольные подсказки) образует агрегацию внимания, которая избирательно собирает значения (сенсорные входы) для получения итогового выхода. В этом разделе мы подробнее разберём агрегацию внимания, чтобы в общих чертах понимать, как механизм внимания работает на практике. Конкретно, модель ядерной регрессии Надарая — Ватсона 1964 года — простой, но полный пример, на котором можно продемонстрировать машинное обучение с механизмом внимания.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
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
tf.random.set_seed(seed=1322)
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

## Генерация набора данных

Для простоты рассмотрим следующую задачу регрессии: по заданному набору пар «вход — выход» $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ как научиться функции $f$, которая предсказывает выход $\hat{y} = f(x)$ для любого нового входа $x$?

Сгенерируем искусственный набор данных по нелинейной функции с добавлением шума $\epsilon$:

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

где $\epsilon$ распределён нормально, со средним $0$ и стандартным отклонением $0.5$. Здесь мы создаём $50$ обучающих и $50$ тестовых примеров. Чтобы лучше визуализировать модели внимания, которые появятся позже, обучающие примеры отсортируем.

**MXNet:**

```python
n_train = 50  # Число обучающих примеров
x_train = np.sort(d2l.rand(n_train) * 5)   # Отсортированные обучающие примеры
```

**PyTorch:**

```python
n_train = 50  # Число обучающих примеров
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Отсортированные обучающие примеры
```

**TensorFlow:**

```python
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

**PaddlePaddle:**

```python
n_train = 50  # Число обучающих примеров
x_train = paddle.sort(paddle.rand([n_train]) * 5)   # Отсортированные обучающие примеры
```

**MXNet:**

```python
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Выходы обучающих примеров
x_test = d2l.arange(0, 5, 0.1)  # Тестовые примеры
y_truth = f(x_test)  # Истинные выходы тестовых примеров
n_test = len(x_test)  # Число тестовых примеров
n_test
```

**PyTorch:**

```python
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Выходы обучающих примеров
x_test = d2l.arange(0, 5, 0.1)  # Тестовые примеры
y_truth = f(x_test)  # Истинные выходы тестовых примеров
n_test = len(x_test)  # Число тестовых примеров
n_test
```

**TensorFlow:**

```python
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # Выходы обучающих примеров
x_test = d2l.arange(0, 5, 0.1)  # Тестовые примеры
y_truth = f(x_test)  # Истинные выходы тестовых примеров
n_test = len(x_test)  # Число тестовых примеров
n_test
```

**PaddlePaddle:**

```python
def f(x):
    return 2 * paddle.sin(x) + x**0.8

y_train = f(x_train) + paddle.normal(0.0, 0.5, (n_train,))  # Выходы обучающих примеров
x_test = d2l.arange(0, 5, 0.1, dtype='float32')   # Тестовые примеры
y_truth = f(x_test)  # Истинные выходы тестовых примеров
n_test = len(x_test)  # Число тестовых примеров
n_test
```

Следующая функция нарисует все обучающие примеры (они показаны кружками), истинную функцию генерации данных без шума (обозначена «Truth»), а также предсказанную функцию (обозначена «Pred»).

```python
#@save
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Среднее агрегирование

Начнём с самой простой оценки для решения задачи регрессии. Вычислим среднее значение выходов всех обучающих примеров:

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,\tag{10.2.1}$$

Как видно на рисунке ниже, такая оценка действительно не очень умна: истинная функция $f$ («Truth») и предсказанная функция («Pred») сильно расходятся.

**MXNet:**

```python
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

**PyTorch:**

```python
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

**TensorFlow:**

```python
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```

**PaddlePaddle:**

```python
y_hat = paddle.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

## Непараметрическая агрегация внимания

Очевидно, что среднее агрегирование игнорирует входы $x_i$. Поэтому Надарая (Nadaraya, 1964) и Ватсон (Watson, 1964) предложили более удачную идею: взвешивать выходы $y_i$ по положению входов:

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,\tag{10.2.2}$$

где $K$ — *ядро* (kernel). Оценка, описанная формулой (10.2.2), называется *ядерной регрессией Надарая — Ватсона* (Nadaraya-Watson kernel regression). Здесь мы не будем углубляться в детали ядерных функций. Но, вдохновляясь этим примером, мы можем переписать (10.2.2) в рамках схемы механизма внимания из рис. 10.1.3 и получить более общую формулу *агрегации внимания* (attention pooling):

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,\tag{10.2.3}$$

где $x$ — запрос, а $(x_i, y_i)$ — пары «ключ — значение». Сравнивая (10.2.3) и (10.2.1), видим, что агрегация внимания — это взвешенное среднее $y_i$. Связь между запросом $x$ и ключом $x_i$ моделируется как *вес внимания* (attention weight) $\alpha(x, x_i)$, который, как показано в (10.2.3), назначается каждому соответствующему значению $y_i$. Для любого запроса веса внимания по всем парам «ключ — значение» образуют корректное распределение вероятностей: они неотрицательны и в сумме дают $1$.

Чтобы лучше понять агрегацию внимания, рассмотрим *гауссово ядро* (Gaussian kernel):

$$K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).$$

Подставляя гауссово ядро в (10.2.3) и (10.2.2), получаем:

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}\tag{10.2.4}$$

В (10.2.4) чем ближе ключ $x_i$ к заданному запросу $x$, тем больше вес внимания, назначенный соответствующему значению $y_i$, то есть тем больше «внимания» он получает.

Важно отметить, что ядерная регрессия Надарая — Ватсона — непараметрическая модель. Поэтому (10.2.4) — это *непараметрическая агрегация внимания* (nonparametric attention pooling). Далее мы построим предсказания именно на основе этой непараметрической модели. Из полученного графика видно, что линия предсказаний новой модели гладкая и ближе к истинной функции, чем предсказания среднего агрегирования.

**MXNet:**

```python
# Форма X_repeat: (n_test, n_train),
# каждая строка содержит один и тот же тестовый вход (то есть один и тот же запрос)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# x_train содержит ключи. Форма attention_weights: (n_test, n_train),
# каждая строка содержит веса внимания, которые распределяются между значениями (y_train) для каждого запроса
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Каждый элемент y_hat — взвешенное среднее значений, где весами служат веса внимания
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

**PyTorch:**

```python
# Форма X_repeat: (n_test, n_train),
# каждая строка содержит один и тот же тестовый вход (то есть один и тот же запрос)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# x_train содержит ключи. Форма attention_weights: (n_test, n_train),
# каждая строка содержит веса внимания, которые распределяются между значениями (y_train) для каждого запроса
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Каждый элемент y_hat — взвешенное среднее значений, где весами служат веса внимания
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

**TensorFlow:**

```python
# Форма X_repeat: (n_test, n_train),
# каждая строка содержит один и тот же тестовый вход (то есть один и тот же запрос)
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# x_train содержит ключи. Форма attention_weights: (n_test, n_train),
# каждая строка содержит веса внимания, которые распределяются между значениями (y_train) для каждого запроса
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# Каждый элемент y_hat — взвешенное среднее значений, где весами служат веса внимания
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

**PaddlePaddle:**

```python
# Форма X_repeat: (n_test, n_train),
# каждая строка содержит один и тот же тестовый вход (то есть один и тот же запрос)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# x_train содержит ключи. Форма attention_weights: (n_test, n_train),
# каждая строка содержит веса внимания, которые распределяются между значениями (y_train) для каждого запроса
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, axis=1)
# Каждый элемент y_hat — взвешенное среднее значений, где весами служат веса внимания
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

Теперь посмотрим на веса внимания. Входы тестовых данных играют роль запросов, а входы обучающих данных — роль ключей. Поскольку оба набора входов отсортированы, видно, что чем ближе пара «запрос — ключ», тем выше [**вес внимания**] в агрегации внимания.

**MXNet:**

```python
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

**PyTorch:**

```python
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

**TensorFlow:**

```python
d2l.show_heatmaps(tf.expand_dims(
                      tf.expand_dims(attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

**PaddlePaddle:**

```python
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## [**Параметрическая агрегация внимания**]

Непараметрическая ядерная регрессия Надарая — Ватсона обладает свойством *состоятельности* (consistency): при достаточном количестве данных эта модель сходится к оптимальному результату. Тем не менее обучаемые параметры легко встроить в агрегацию внимания.

Например, в отличие от (10.2.4), расстояние между запросом $x$ и ключом $x_i$ умножается на обучаемый параметр $w$:

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}\tag{10.2.5}$$

Оставшаяся часть раздела посвящена обучению этой модели (10.2.5), то есть обучению параметров агрегации внимания.

### Пакетное матричное умножение

Чтобы эффективнее вычислять внимание для мини-батчей, мы можем использовать пакетное матричное умножение, которое предоставляют фреймворки глубокого обучения.

Пусть в первом мини-батче есть $n$ матриц $\mathbf{X}_1,\ldots, \mathbf{X}_n$ формы $a\times b$, а во втором мини-батче — $n$ матриц $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ формы $b\times c$. Их пакетное матричное умножение даёт $n$ матриц $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ формы $a\times c$. Таким образом, [**если два тензора имеют формы $(n,a,b)$ и $(n,b,c)$, то результат их пакетного матричного умножения имеет форму $(n,a,c)$**].

**MXNet:**

```python
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

**PyTorch:**

```python
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

**TensorFlow:**

```python
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape
```

**PaddlePaddle:**

```python
X = paddle.ones((2, 1, 4))
Y = paddle.ones((2, 4, 6))
paddle.bmm(X, Y).shape
```

В контексте механизма внимания пакетное матричное умножение можно использовать, чтобы [**вычислять взвешенные средние значения в мини-батче**].

**MXNet:**

```python
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

**PyTorch:**

```python
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

**TensorFlow:**

```python
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

**PaddlePaddle:**

```python
weights = paddle.ones((2, 10)) * 0.1
values = paddle.arange(20, dtype='float32').reshape((2, 10))
paddle.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

### Определение модели

На основе [**параметрической агрегации внимания**] (10.2.5) с использованием пакетного матричного умножения определим параметрическую версию ядерной регрессии Надарая — Ватсона:

**MXNet:**

```python
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Форма queries и attention_weights: (число запросов, число пар «ключ — значение»)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Форма values: (число запросов, число пар «ключ — значение»)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

**PyTorch:**

```python
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Форма queries и attention_weights: (число запросов, число пар «ключ — значение»)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Форма values: (число запросов, число пар «ключ — значение»)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

**TensorFlow:**

```python
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))
        
    def call(self, queries, keys, values, **kwargs):
        # При обучении «запросы» — это x_train, «ключи» — расстояния от каждой точки до обучающих данных, «значения» — y_train
        # Форма queries и attention_weights: (число запросов, число пар «ключ — значение»)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # Форма values: (число запросов, число пар «ключ — значение»)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```

**PaddlePaddle:**

```python
class NWKernelRegression(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = paddle.create_parameter((1,), dtype='float32')

    def forward(self, queries, keys, values):
        # Форма queries и attention_weights: (число запросов, число пар «ключ — значение»)
        queries = queries.reshape((queries.shape[0], 1)) \
        .tile([keys.shape[1]]) \
        .reshape((-1, keys.shape[1]))
        self.attention_weight = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, axis=1)
        # Форма values: (число запросов, число пар «ключ — значение»)
        return paddle.bmm(self.attention_weight.unsqueeze(1),
                          values.unsqueeze(-1)).reshape((-1, ))
```

### Обучение

Далее [**преобразуем обучающий набор данных в ключи и значения**] для обучения модели внимания. В параметрической модели агрегации внимания вход каждого обучающего примера вычисляется со всеми парами «ключ — значение» остальных обучающих примеров (кроме его самого), и так получается его предсказанный выход.

**MXNet:**

```python
# Форма X_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие входы
X_tile = np.tile(x_train, (n_train, 1))
# Форма Y_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие выходы
Y_tile = np.tile(y_train, (n_train, 1))
# Форма keys: (n_train, n_train - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Форма values: (n_train, n_train - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

**PyTorch:**

```python
# Форма X_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие входы
X_tile = x_train.repeat((n_train, 1))
# Форма Y_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие выходы
Y_tile = y_train.repeat((n_train, 1))
# Форма keys: (n_train, n_train - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Форма values: (n_train, n_train - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

**TensorFlow:**

```python
# Форма X_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие входы
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Форма Y_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие выходы
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# Форма keys: (n_train, n_train - 1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# Форма values: (n_train, n_train - 1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

**PaddlePaddle:**

```python
# Форма X_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие входы
X_tile = x_train.tile([n_train, 1])
# Форма Y_tile: (n_train, n_train), каждая строка содержит одни и те же обучающие выходы
Y_tile = y_train.tile([n_train, 1])
# Форма keys: (n_train, n_train - 1)
keys = X_tile[(1 - paddle.eye(n_train)).astype(paddle.bool)].reshape((n_train, -1))
# Форма values: (n_train, n_train - 1)
values = Y_tile[(1 - paddle.eye(n_train)).astype(paddle.bool)].reshape((n_train, -1))
```

[**Обучая параметрическую модель агрегации внимания**], мы используем функцию потерь «квадрат ошибки» и стохастический градиентный спуск.

**MXNet:**

```python
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

**PyTorch:**

```python
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

**TensorFlow:**

```python
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```

**PaddlePaddle:**

```python
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = paddle.optimizer.SGD(learning_rate=0.5, parameters=net.parameters())
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.clear_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

Как видно ниже, после обучения параметрической модели агрегации внимания при попытке подогнать зашумлённые обучающие данные [**график предсказаний**] получается менее гладким, чем у непараметрической модели.

**MXNet:**

```python
# Форма keys: (n_test, n_train), каждая строка содержит одни и те же обучающие входы (то есть одни и те же ключи)
keys = np.tile(x_train, (n_test, 1))
# Форма values: (n_test, n_train)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

**PyTorch:**

```python
# Форма keys: (n_test, n_train), каждая строка содержит одни и те же обучающие входы (то есть одни и те же ключи)
keys = x_train.repeat((n_test, 1))
# Форма values: (n_test, n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

**TensorFlow:**

```python
# Форма keys: (n_test, n_train), каждая строка содержит одни и те же обучающие входы (то есть одни и те же ключи)
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# Форма values: (n_test, n_train)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

**PaddlePaddle:**

```python
# Форма keys: (n_test, n_train), каждая строка содержит одни и те же обучающие входы (то есть одни и те же ключи)
keys = x_train.tile([n_test, 1])
# Форма values: (n_test, n_train)
values = y_train.tile([n_test, 1])
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

Почему новая модель стала менее гладкой? Посмотрим на график весов внимания. По сравнению с непараметрической моделью агрегации внимания, в параметрической модели после добавления обучаемых параметров [**кривая становится менее гладкой в областях с большими весами внимания**].

**MXNet:**

```python
d2l.show_heatmaps(np.expand_dims(
                      np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

**PyTorch:**

```python
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

**TensorFlow:**

```python
d2l.show_heatmaps(tf.expand_dims(
                      tf.expand_dims(net.attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

**PaddlePaddle:**

```python
d2l.show_heatmaps(net.attention_weight.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## Резюме

* Ядерная регрессия Надарая — Ватсона — это машинное обучение с механизмом внимания.
* Агрегация внимания в ядерной регрессии Надарая — Ватсона — это взвешенное среднее обучающих выходов. С точки зрения внимания вес внимания, назначенный каждому значению, зависит от функции ключа и запроса, которые подаются на вход.
* Агрегация внимания бывает непараметрической и параметрической.

## Упражнения

1. Увеличив число обучающих примеров, можно ли получить лучшую непараметрическую модель ядерной регрессии Надарая — Ватсона?
1. Какую роль играет обученный параметр $w$ в эксперименте с параметрической агрегацией внимания? Почему при визуализации весов внимания он делает взвешенные области более резкими?
1. Как добавить гиперпараметры в непараметрическую ядерную регрессию Надарая — Ватсона, чтобы улучшить предсказания?
1. Предложите новую параметрическую модель агрегации внимания для ядерной регрессии из этого раздела. Обучите её и визуализируйте её веса внимания.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5759)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5760)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11840)

## Литература

* Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability & Its Applications*, 9(1), 141–142.
* Watson, G. S. (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of Statistics, Series A*, 359–372.
