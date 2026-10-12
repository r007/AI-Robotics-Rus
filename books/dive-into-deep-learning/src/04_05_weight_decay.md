# 4.5 Затухание весов

В предыдущем разделе мы описали проблему переобучения.
В этом разделе мы познакомимся с некоторыми методами регуляризации моделей.
Переобучение всегда можно смягчить, собрав больше обучающих данных.
Однако это может быть дорого, требовать много времени или быть вне нашего контроля, поэтому в краткосрочной перспективе такой путь обычно невозможен.
Если же у нас уже есть как можно больше качественных данных, мы можем сосредоточиться на методах регуляризации.

Напомним, что в примере с полиномиальной регрессией (раздел 4.4) мы могли ограничивать ёмкость модели, меняя степень полинома.
На практике ограничение числа признаков — распространённый способ борьбы с переобучением.
Однако простое отбрасывание признаков может оказаться слишком грубым инструментом.
Продолжим пример с полиномиальной регрессией и рассмотрим, что происходит при многомерных входах.
Естественное обобщение многочленов на многомерные данные называется *одночленами* (monomials) — это произведения степеней переменных.
Степень одночлена равна сумме степеней его множителей.
Например, $x_1^2 x_2$ и $x_3 x_5^2$ — одночлены степени 3.

Заметим, что с ростом степени $d$ число членов степени $d$ быстро возрастает.
При $k$ переменных число членов степени $d$ равно ${k - 1 + d} \choose {k - 1}$, то есть $C^{k-1}_{k-1+d} = \frac{(k-1+d)!}{(d)!(k-1)!}$.
Поэтому даже небольшое изменение степени, например с 2 до 3, заметно увеличивает сложность модели.
Если ограничивать только число признаков (в полиномиальной регрессии — степень многочлена), модель может оставаться либо слишком простой, либо слишком сложной.
Нам нужен более тонкий инструмент, который позволяет регулировать сложность функции и находить подходящий баланс.

## 4.5.1 Нормы и затухание весов

В разделе 2.3.10 мы описали $L_2$- и $L_1$-нормы, которые являются частными случаями более общих $L_p$-норм.
В обучении параметризованных моделей машинного обучения *затухание весов* (weight decay) — один из самых широко используемых методов регуляризации; его также называют $L_2$-*регуляризацией*.
Этот метод измеряет сложность функции через её расстояние до нуля, поскольку среди всех функций $f$ функция $f = 0$ (которая для всех входов возвращает $0$) в некотором смысле самая простая.
Но как точно измерить расстояние между функцией и нулём? Единого правильного ответа нет.
Фактически над этим вопросом работают функциональный анализ и теория банаховых пространств.

Простой способ — измерять сложность линейной функции $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ через некоторую норму вектора весов, например $\| \mathbf{w} \|^2$.
Чтобы вектор весов оставался небольшим, чаще всего его норму добавляют в задачу минимизации потерь в качестве штрафного члена.
Вместо того чтобы *минимизировать только ошибку предсказания на обучающих метках*, мы минимизируем *сумму ошибки предсказания и штрафного члена*.
Теперь, если вектор весов становится слишком большим, алгоритм обучения будет в большей степени заниматься минимизацией нормы весов $\| \mathbf{w} \|^2$.
Именно этого мы и хотим.
Вернёмся к примеру линейной регрессии из раздела 3.1.
Наши потери задаются формулой:

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Напомним, что $\mathbf{x}^{(i)}$ — признаки образца $i$, $y^{(i)}$ — метка образца $i$, а $(\mathbf{w}, b)$ — параметры весов и смещения.
Чтобы штрафовать размер вектора весов, нужно каким-то образом добавить $\| \mathbf{w} \|^2$ в функцию потерь.
Но как модель должна сбалансировать этот новый дополнительный штраф с исходными потерями?
На практике этот компромисс задаётся *коэффициентом регуляризации* $\lambda$ — неотрицательным гиперпараметром, который подбирают по проверочным данным:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

При $\lambda = 0$ мы возвращаемся к исходной функции потерь.
При $\lambda > 0$ мы ограничиваем размер $\| \mathbf{w} \|$.
Здесь по-прежнему делим на $2$: при взятии производной квадратичной функции двойка и $1/2$ сокращаются, и выражения для обновления получаются аккуратными и простыми.
Почему здесь используется квадрат нормы, а не сама норма (то есть евклидово расстояние)?
Так удобнее вычислять.
Взяв квадрат $L_2$-нормы, мы избавляемся от квадратного корня и получаем сумму квадратов компонент вектора весов.
Благодаря этому производная штрафа легко вычисляется: производная суммы равна сумме производных.

Почему для начала мы выбираем именно $L_2$-норму, а не $L_1$-норму?
На самом деле этот выбор и эффективен, и популярен во всей статистике.
$L_2$-регуляризованная линейная модель образует классический алгоритм *гребневой регрессии* (ridge regression), а $L_1$-регуляризованная линейная регрессия — близкую базовую модель статистики, обычно называемую *лассо-регрессией* (lasso regression).
Одна из причин использовать $L_2$-норму в том, что она сильно штрафует большие компоненты вектора весов.
Это заставляет алгоритм обучения отдавать предпочтение моделям, в которых веса равномерно распределены по большому числу признаков.
На практике это может делать такие модели более устойчивыми к ошибкам наблюдений в отдельных переменных.
Напротив, $L_1$-штраф приводит к тому, что модель концентрирует веса на небольшом числе признаков, а остальные веса обнуляет.
Это называется *отбором признаков* (feature selection), и такая возможность может быть полезна в других задачах.

Используя те же обозначения, что и в формуле (3.1.4), запишем обновление мини-батчевого стохастического градиентного спуска для регрессии с $L_2$-регуляризацией:

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

Как и раньше, мы обновляем $\mathbf{w}$ по разности между предсказанием и наблюдаемым значением.
Но одновременно мы стремимся уменьшить размер $\mathbf{w}$ до нуля.
Поэтому этот метод иногда называют *затуханием весов*.
Если учитывать только штрафной член, на каждом шаге оптимизации веса *затухают*.
По сравнению с отбором признаков затухание весов даёт непрерывный механизм настройки сложности функции.
Меньшие значения $\lambda$ соответствуют менее ограниченным $\mathbf{w}$, а большие значения $\lambda$ накладывают более сильные ограничения на $\mathbf{w}$.

Штрафовать ли соответствующее смещение $b^2$, на практике решают по-разному, и в разных слоях нейронной сети это тоже может различаться.
Обычно смещение выходного слоя сети не регуляризуют.

## 4.5.2 Многомерная линейная регрессия

Продемонстрируем затухание весов на простом примере.

**MXNet:**
```python
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

**PyTorch:**
```python
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

**TensorFlow:**
```python
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

**PaddlePaddle:**
```python
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

Сначала **сгенерируем данные, как и раньше** по формуле:

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ где }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$

Мы выбираем метку как линейную функцию входов.
Метка также искажается гауссовым шумом с математическим ожиданием 0 и стандартным отклонением 0.01.
Чтобы эффект переобучения стал заметнее, можно увеличить размерность задачи до $d = 200$ и использовать небольшой обучающий набор из 20 образцов.

```python
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## 4.5.3 Реализация с нуля

Теперь реализуем затухание весов с нуля: достаточно добавить к исходной целевой функции квадратичный штраф $L_2$.

### 4.5.3.1 Инициализация параметров модели

Сначала определим функцию, которая случайным образом инициализирует параметры модели.

**MXNet:**
```python
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

**PyTorch:**
```python
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

**TensorFlow:**
```python
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

**PaddlePaddle:**
```python
def init_params():
    w = paddle.normal(0, 1, shape=(num_inputs, 1))
    w.stop_gradient = False
    b = paddle.zeros(shape=[1])
    b.stop_gradient = False
    return [w, b]
```

### 4.5.3.2 Определение штрафа $L_2$-нормы

Самый удобный способ реализовать этот штраф — возвести все элементы в квадрат и просуммировать их.

**MXNet:**
```python
def l2_penalty(w):
    return (w**2).sum() / 2
```

**PyTorch:**
```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

**TensorFlow:**
```python
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

**PaddlePaddle:**
```python
def l2_penalty(w):
    return paddle.sum(w.pow(2)) / 2
```

### 4.5.3.3 Определение кода обучения

Приведённый ниже код подгоняет модель под обучающий набор и оценивает её на тестовом наборе.
Начиная с главы 3 линейная сеть и квадратичная функция потерь не меняются, поэтому мы импортируем их через `d2l.linreg` и `d2l.squared_loss`.
Единственное отличие: теперь в потери входит штрафной член.

**MXNet:**
```python
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # Добавлен штраф L2-нормы,
                # broadcasting делает l2_penalty(w) вектором длины batch_size
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', np.linalg.norm(w))
```

**PyTorch:**
```python
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # Добавлен штраф L2-нормы,
            # broadcasting делает l2_penalty(w) вектором длины batch_size
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', torch.norm(w).item())
```

**TensorFlow:**
```python
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # Добавлен штраф L2-нормы,
                # broadcasting делает l2_penalty(w) вектором длины batch_size
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', tf.norm(w).numpy())
```

**PaddlePaddle:**
```python
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter():
            # Добавлен штраф L2-нормы,
            # broadcasting делает l2_penalty(w) вектором длины `batch_size`
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', paddle.norm(w).item())
```

### 4.5.3.4 Обучение без регуляризации

Теперь запустим код с `lambd = 0`, то есть без затухания весов.
Обратите внимание: ошибка обучения уменьшилась, а ошибка на тесте — нет.
Это означает серьёзное переобучение.

```python
train(lambd=0)
```

### 4.5.3.5 Обучение с затуханием весов

Теперь запустим код с затуханием весов.
Обратите внимание: ошибка обучения возросла, а ошибка на тесте уменьшилась.
Это именно тот эффект, которого мы ожидаем от регуляризации.

```python
train(lambd=3)
```

## 4.5.4 Краткая реализация

Поскольку затухание весов широко используется при оптимизации нейронных сетей, фреймворки глубокого обучения встраивают его в алгоритмы оптимизации, чтобы его можно было применять с любой функцией потерь.
Кроме того, такая интеграция удобна с вычислительной точки зрения: затухание весов можно добавить в алгоритм без дополнительных вычислительных затрат.
Поскольку часть обновления, отвечающая за затухание весов, зависит только от текущего значения каждого параметра, оптимизатор должен обращаться к каждому параметру хотя бы один раз.

**MXNet:** В приведённом ниже коде мы задаём гиперпараметр затухания весов `wd` прямо при создании `Trainer`.
По умолчанию Gluon затухают и веса, и смещения.
Обратите внимание, что при обновлении параметров модели гиперпараметр `wd` умножается на `wd_mult`.
Поэтому, если установить `wd_mult` равным нулю, смещение $b$ не будет затухать.

**PyTorch:** В приведённом ниже коде мы задаём гиперпараметр затухания весов `weight_decay` прямо при создании оптимизатора.
По умолчанию PyTorch затухают и веса, и смещения.
Здесь мы задаём `weight_decay` только для весов, поэтому смещение $b$ не затухает.

**TensorFlow:** В приведённом ниже коде мы создаём $L_2$-регуляризатор с гиперпараметром затухания весов `wd` и применяем его к слою сети через аргумент `kernel_regularizer`.

**MXNet:**
```python
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # Параметры смещения не затухают. Имена смещений обычно заканчиваются на «bias»
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', np.linalg.norm(net[0].weight.data()))
```

**PyTorch:**
```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # Параметры смещения не затухают
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', net[0].weight.norm().item())
```

**TensorFlow:**
```python
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # В tf.keras для пользовательского цикла обучения потери нужно добавлять вручную.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', tf.norm(net.get_weights()[0]).numpy())
```

**PaddlePaddle:**
```python
def train_concise(wd):
    weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0))
    bias_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0))
    net = nn.Sequential(nn.Linear(num_inputs, 1, weight_attr=weight_attr, bias_attr=bias_attr))
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # Параметры смещения не затухают.
    trainer = paddle.optimizer.SGD(parameters=net[0].parameters(), learning_rate=lr, weight_decay=wd*1.0)
    animator = d2l.Animator(xlabel='эпохи', ylabel='потери', yscale='log',
                            xlim=[5, num_epochs], legend=['обучение', 'тест'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            trainer.clear_grad()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2-норма w:', net[0].weight.norm().item())
```

**Эти графики выглядят так же, как графики при реализации затухания весов с нуля**.
Однако они работают быстрее и проще в реализации.
Для более сложных задач это преимущество будет заметнее.

```python
train_concise(0)
```

```python
train_concise(3)
```

До сих пор мы касались только понятия простой линейной функции.
Какая функция считается простой нелинейной, — более сложный вопрос.
Например, [воспроизводящие ядерные гильбертовы пространства (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
позволяют применять для нелинейных функций инструменты, введённые для линейных.
К сожалению, алгоритмы на основе RKHS плохо масштабируются на большие многомерные данные.
В этой книге мы по умолчанию будем использовать простую эвристику: применять затухание весов ко всем слоям глубокой сети.

## Резюме

* Регуляризация — распространённый способ борьбы с переобучением: к функции потерь на обучающем наборе добавляется штрафной член, чтобы снизить сложность полученной модели.
* Особый способ сохранить модель простой — затухание весов с $L_2$-штрафом. Это приводит к затуханию весов на шаге обновления алгоритма обучения.
* Затухание весов поддерживается оптимизаторами фреймворков глубокого обучения.
* В одной и той же реализации обучения разные наборы параметров могут вести себя по-разному при обновлении.

## Упражнения

1. Поэкспериментируйте со значениями $\lambda$ в задаче из этого раздела. Постройте график точности на обучении и на тесте как функцию $\lambda$. Что вы наблюдаете?
1. Используйте проверочный набор, чтобы найти наилучшее значение $\lambda$. Действительно ли оно оптимально? Имеет ли это значение?
1. Какими будут уравнения обновления, если в качестве штрафа используется $\sum_i |w_i|$ (регуляризация $L_1$)?
1. Мы знаем, что $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Можно ли найти аналогичное матричное уравнение (см. норму Фробениуса в разделе 2.3.10)?
1. Вспомните связь между ошибкой обучения и ошибкой обобщения. Кроме затухания весов, увеличения объёма обучающих данных и использования модели подходящей сложности, какие ещё способы борьбы с переобучением вы можете придумать?
1. В байесовской статистике мы используем произведение априорного распределения и правдоподобия. Через формулу $P(w \mid x) \propto P(x \mid w) P(w)$ получаем апостериорное распределение. Как получить регуляризацию через $P(w)$?

* MXNet: [discuss.d2l.ai/t/1810](https://discuss.d2l.ai/t/1810)
* PyTorch: [discuss.d2l.ai/t/1808](https://discuss.d2l.ai/t/1808)
* TensorFlow: [discuss.d2l.ai/t/1809](https://discuss.d2l.ai/t/1809)
* PaddlePaddle: [discuss.d2l.ai/t/11773](https://discuss.d2l.ai/t/11773)
