# 3.7 Softmax-регрессия: краткая реализация

В разделе 3.3 мы убедились, что высокоуровневый API фреймворков упрощает реализацию линейной регрессии.
Точно так же с помощью высокоуровневого API удобно реализовать и softmax-регрессию.
В этом разделе, как и в разделе 3.6, мы продолжаем работать с набором Fashion-MNIST и сохраняем размер батча равным 256.

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

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 3.7.1 Инициализация параметров модели

Как описано в разделе 3.4, **выходной слой softmax-регрессии — полносвязный слой**.
Поэтому для реализации модели достаточно добавить в `Sequential` один полносвязный слой с 10 выходами.
Здесь `Sequential` формально тоже не нужен, но он служит основой для построения глубоких моделей.
Веса по-прежнему инициализируем случайными значениями из нормального распределения со средним 0 и стандартным отклонением 0.01.

**MXNet:**

```python
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

**PyTorch:**

```python
# PyTorch не меняет форму входных данных неявно. Поэтому перед линейным слоем
# мы определяем слой Flatten, который приводит входную форму сети к нужному виду
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

**TensorFlow:**

```python
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

**PaddlePaddle:**

```python
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.initializer.Normal(m.weight, std=0.01)

net.apply(init_weights);
```

## 3.7.2 Переосмысление реализации softmax

В примере из раздела 3.6 мы сначала вычисляли выход модели, а затем передавали его в кросс-энтропийную потерю.
С математической точки зрения это вполне корректно.
Однако с вычислительной точки зрения экспоненты могут создавать проблемы с численной стабильностью.

Напомним, что softmax задаётся формулой $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$, где $\hat y_j$ — предсказанное распределение вероятностей, а $o_j$ — $j$-й элемент ненормированного предсказания $\mathbf{o}$.
Если некоторые $o_k$ очень велики, то $\exp(o_k)$ может превысить максимальное число, которое допускает тип данных, — возникает *переполнение* (overflow).
Тогда числитель или знаменатель становится равным `inf` (бесконечности), и в итоге получаем $\hat y_j$, равные 0, `inf` или `nan` (не число).
В таких случаях корректное значение кросс-энтропии вычислить невозможно.

Один из приёмов решения этой проблемы: перед вычислением softmax вычесть из всех $o_k$ величину $\max(o_k)$.
Видно, что сдвиг всех $o_k$ на одну и ту же константу не меняет результат softmax:

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$

После вычитания и нормировки некоторые величины $o_j - \max(o_k)$ могут иметь большие по модулю отрицательные значения.
Из-за ограниченной точности $\exp(o_j - \max(o_k))$ окажется очень близким к нулю, то есть возникнет *антипереполнение* (underflow).
Эти значения могут округлиться до нуля, так что $\hat y_j$ станет равным нулю, а $\log(\hat y_j)$ — `-inf`.
После нескольких шагов обратного распространения мы можем получить целый экран неприятных `nan`.

Хотя мы вычисляем экспоненты, в итоге в кросс-энтропийной потере берётся их логарифм.
Поэтому, объединив softmax и кросс-энтропию, можно избежать проблем численной стабильности при обратном распространении.
Как видно из следующих равенств, можно не вычислять $\exp(o_j - \max(o_k))$, а сразу использовать $o_j - \max(o_k)$, поскольку $\log(\exp(\cdot))$ сокращается:

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

Традиционная функция softmax по-прежнему нужна, если мы хотим оценивать вероятности, которые выдаёт модель.
Однако в функцию потерь мы передаём не вероятности softmax, а ненормированные предсказания: **softmax и его логарифм вычисляются внутри кросс-энтропийной потери одновременно**.
Это разумный приём, родственный [«трюку LogSumExp»](https://en.wikipedia.org/wiki/LogSumExp).

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
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## 3.7.3 Алгоритм оптимизации

Здесь **используем мини-батчевый стохастический градиентный спуск с скоростью обучения 0.1**.
Это тот же алгоритм, что и в примере линейной регрессии, что показывает универсальность оптимизаторов.

**MXNet:**

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

**PyTorch:**

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

**TensorFlow:**

```python
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

**PaddlePaddle:**

```python
trainer = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
```

## 3.7.4 Обучение

Далее **вызываем функцию обучения, определённую в разделе 3.6, и обучаем модель**.

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

Как и раньше, алгоритм сходится к довольно высокой точности, а код теперь заметно короче.

## Резюме

* С помощью высокоуровневого API фреймворков softmax-регрессию можно реализовать значительно короче.
* С вычислительной точки зрения реализация softmax-регрессии довольно сложна. Во многих случаях фреймворки дополнительно принимают меры предосторожности сверх известных приёмов, чтобы обеспечить численную стабильность. Это избавляет нас от ловушек, которые возникают на практике, когда модели пишутся с нуля.

## Упражнения

1. Попробуйте подобрать гиперпараметры, например размер батча, число эпох и скорость обучения, и посмотрите, как меняются результаты.
1. Увеличьте число эпох. Почему через некоторое время точность на тестовых данных начинает снижаться? Как можно решить эту проблему?

* MXNet: [discuss.d2l.ai/t/1794](https://discuss.d2l.ai/t/1794)
* PyTorch: [discuss.d2l.ai/t/1793](https://discuss.d2l.ai/t/1793)
* TensorFlow: [discuss.d2l.ai/t/1792](https://discuss.d2l.ai/t/1792)
* PaddlePaddle: [discuss.d2l.ai/t/11761](https://discuss.d2l.ai/t/11761)

## Литература

* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
* Russell, S. J., & Norvig, P. (2016). *Artificial intelligence: A modern approach*. Pearson Education.
* Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.
* Zhang, A., Tay, Y., Zhang, S., Chan, A., Luu, A. T., Hui, S. C., & Fu, J. (2021). Beyond fully-connected layers with quaternions: Parameterization of hypercomplex multiplications with 1/n parameters. In *International Conference on Learning Representations*.
