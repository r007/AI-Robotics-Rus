# 5.2 Управление параметрами

В выборе архитектуры и задании гиперпараметров мы переходим к этапу обучения.
На этом этапе цель состоит в том, чтобы найти значения параметров модели, которые минимизируют функцию потерь.
После обучения эти параметры понадобятся нам для будущих предсказаний.
Кроме того, иногда мы хотим извлечь параметры, чтобы повторно использовать их в других условиях,
сохранить модель, чтобы её можно было выполнить в другом программном обеспечении,
или изучить параметры для научного понимания.

В предыдущих разделах мы полагались только на фреймворк глубокого обучения при обучении модели
и пропускали детали работы с параметрами.
В этом разделе мы рассмотрим следующее:

* доступ к параметрам для отладки, диагностики и визуализации;
* инициализация параметров;
* совместное использование параметров между различными компонентами модели.

**Сначала рассмотрим многослойный перцептрон с одним скрытым слоем.**

**MXNet:**
```python
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Используем метод инициализации по умолчанию

X = np.random.uniform(size=(2, 4))
net(X)  # Прямое распространение
```

**PyTorch:**
```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

**TensorFlow:**
```python
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

**PaddlePaddle:**
```python
import warnings
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = paddle.rand([2, 4])
net(X)
```

## 5.2.1 Доступ к параметрам

Параметры можно получить из уже существующей модели.
Если модель задана с помощью `Sequential`, к любому её слою можно обратиться по индексу.
Модель ведёт себя как список, а параметры каждого слоя находятся в его атрибутах.
Как показано ниже, мы можем проверить параметры второго полносвязного слоя.

**MXNet:**
```python
print(net[1].params)
```

**PyTorch, PaddlePaddle:**
```python
print(net[2].state_dict())
```

**TensorFlow:**
```python
print(net.layers[2].weights)
```

Вывод даёт нам несколько важных сведений.
Во-первых, этот полносвязный слой содержит два параметра: весовую матрицу и вектор смещений.
Оба хранятся как числа с плавающей точкой одинарной точности (float32).
Обратите внимание, что имена параметров позволяют однозначно идентифицировать каждый параметр,
даже в сети из сотен слоёв.

### 5.2.1.1 Целевые параметры

Каждый параметр представлен экземпляром класса параметров.
Чтобы выполнить с параметром какую-либо операцию, сначала нужно получить доступ к его числовому значению.
Существует несколько способов это сделать: одни проще, другие универсальнее.
Код ниже извлекает смещение из второго полносвязного слоя (то есть из третьего слоя сети).
Извлечённый объект является экземпляром класса параметров, и далее мы обращаемся к значению этого параметра.

**MXNet:**
```python
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

**PyTorch:**
```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

**TensorFlow:**
```python
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

**PaddlePaddle:**
```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.value)
```

**MXNet, PyTorch, PaddlePaddle:** Параметры — составные объекты, которые содержат значения, градиенты и дополнительную информацию.
Поэтому нам и нужно явно обращаться к значениям параметров.
Помимо значения, мы также можем получить градиент каждого параметра.
В приведённой сети мы ещё не вызывали обратное распространение, поэтому градиенты параметров находятся в начальном состоянии.

**MXNet:**
```python
net[1].weight.grad()
```

**PyTorch, PaddlePaddle:**
```python
net[2].weight.grad == None
```

### 5.2.1.2 Одновременный доступ ко всем параметрам

Когда нужно выполнить операцию над всеми параметрами, обращаться к каждому по отдельности может быть неудобно.
Это становится особенно сложно при работе с более сложными блоками (например, с вложенными блоками),
так как тогда приходится рекурсивно обходить всё дерево, чтобы извлечь параметры каждого вложенного блока.
Ниже мы сравним доступ к параметрам первого полносвязного слоя с доступом ко всем слоям.

**MXNet:**
```python
print(net[0].collect_params())
print(net.collect_params())
```

**PyTorch, PaddlePaddle:**
```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

**TensorFlow:**
```python
print(net.layers[1].weights)
print(net.get_weights())
```

Это даёт нам другой способ доступа к параметрам сети, как показано ниже.

**MXNet:**
```python
net.collect_params()['dense1_bias'].data()
```

**PyTorch:**
```python
net.state_dict()['2.bias'].data
```

**TensorFlow:**
```python
net.get_weights()[1]
```

**PaddlePaddle:**
```python
net.state_dict()['2.bias']
```

### 5.2.1.3 Сбор параметров из вложенных блоков

Посмотрим, как работает соглашение об именовании параметров, если вкладывать блоки друг в друга.
Сначала определим функцию, которая создаёт блоки (её можно назвать «фабрикой блоков»),
а затем объединим эти блоки в более крупный блок.

**MXNet:**
```python
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Вложение блоков
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

**PyTorch:**
```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Вложение блоков
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

**TensorFlow:**
```python
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # Вложение блоков
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

**PaddlePaddle:**
```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), 
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Вложение блоков
        net.add_sublayer(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

**Спроектировав сеть, посмотрим, как она работает.**

**MXNet:**
```python
print(rgnet.collect_params)
print(rgnet.collect_params())
```

**PyTorch, PaddlePaddle:**
```python
print(rgnet)
```

**TensorFlow:**
```python
print(rgnet.summary())
```

Поскольку слои вложены иерархически, к ним можно обращаться так же, как по индексам вложенного списка.
Ниже мы получаем смещение первого слоя во втором подблоке первого основного блока.

**MXNet:**
```python
rgnet[0][1][0].bias.data()
```

**PyTorch:**
```python
rgnet[0][1][0].bias.data
```

**TensorFlow:**
```python
rgnet.layers[0].layers[1].layers[1].weights[1]
```

**PaddlePaddle:**
```python
print(rgnet[0].state_dict()['block 0.0.bias'])
```

## 5.2.2 Инициализация параметров

Теперь, когда мы знаем, как получить доступ к параметрам, рассмотрим, как правильно их инициализировать.
Мы обсуждали необходимость хорошей инициализации в разделе 4.8.
Фреймворки глубокого обучения предоставляют инициализацию по умолчанию,
а также позволяют создавать собственные методы инициализации, которые удовлетворяют нашим правилам.

**MXNet:** По умолчанию MXNet инициализирует весовые параметры, случайно выбирая их из равномерного распределения $U(-0.07, 0.07)$, а параметры смещения задаёт равными 0.
Модуль `init` MXNet предоставляет несколько готовых методов инициализации.

**PyTorch:** По умолчанию PyTorch инициализирует весовые матрицы и векторы смещений равномерно в диапазоне, который вычисляется по размерностям входа и выхода.
Модуль `nn.init` PyTorch предоставляет несколько готовых методов инициализации.

**TensorFlow:** По умолчанию Keras инициализирует весовые матрицы равномерно в диапазоне, который вычисляется по размерностям входа и выхода.
Параметры смещения задаются равными 0.
TensorFlow предоставляет различные методы инициализации в корневом модуле и в модуле `keras.initializers`.

**PaddlePaddle:** По умолчанию PaddlePaddle инициализирует весовые матрицы методом Xavier, а параметры смещения задаёт равными 0.
Модуль `nn.initializer` PaddlePaddle предоставляет несколько готовых методов инициализации.

### 5.2.2.1 Встроенная инициализация

Начнём с вызова встроенного инициализатора.
Приведённый ниже код инициализирует все весовые параметры гауссовскими случайными величинами со стандартным отклонением 0.01,
а параметры смещения задаёт равными нулю.

**MXNet:**
```python
# Параметр force_reinit гарантирует, что параметры будут заново инициализированы,
# независимо от того, инициализировались ли они ранее
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

**PyTorch:**
```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

**TensorFlow:**
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

**PaddlePaddle:**
```python
def init_normal(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.Normal(mean=0.0, std=0.01)
        paddle.zeros(m.bias)    
net.apply(init_normal)
net[0].weight[0],net[0].state_dict()['bias']
```

Мы также можем инициализировать все параметры заданной константой, например 1.

**MXNet:**
```python
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

**PyTorch:**
```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

**TensorFlow:**
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

**PaddlePaddle:**
```python
def init_constant(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.Constant(value = 1)
        paddle.zeros(m.bias)
net.apply(init_constant)
net[0].weight[0],net[0].state_dict()['bias']
```

Также мы можем **применять разные методы инициализации к разным блокам**.
Например, ниже мы инициализируем первый слой нейронной сети методом Xavier,
а затем третий слой инициализируем константой 42.

**MXNet:**
```python
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

**PyTorch:**
```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

**TensorFlow:**
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

**PaddlePaddle:**
```python
def xavier(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.XavierUniform(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        paddle.nn.initializer.Constant(42)
        
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight[0])
print(net[2].weight)
```

### 5.2.2.2 Пользовательская инициализация

Иногда фреймворк глубокого обучения не предоставляет нужный нам метод инициализации.
В следующем примере мы используем приведённое ниже распределение, чтобы определить метод инициализации для произвольного весового параметра $w$:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ с вероятностью } \frac{1}{4} \\
            0    & \text{ с вероятностью } \frac{1}{2} \\
        U(-10, -5) & \text{ с вероятностью } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

**MXNet:** Здесь мы определяем подкласс класса `Initializer`.
Обычно достаточно реализовать функцию `_init_weight`, которая принимает тензор-аргумент (`data`) и присваивает ему нужное значение инициализации.

**PyTorch:** Аналогично мы реализуем функцию `my_init` и применяем её к `net`.

**TensorFlow:** Здесь мы определяем подкласс `Initializer` и реализуем функцию `__call__`.
Эта функция возвращает требуемый тензор заданной формы и типа данных.

**PaddlePaddle:** Аналогично мы реализуем функцию `my_init` и применяем её к `net`.

**MXNet:**
```python
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Инициализация', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

**PyTorch:**
```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Инициализация", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

**TensorFlow:**
```python
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

**PaddlePaddle:**
```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Инициализация", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        paddle.nn.initializer.XavierUniform(m.weight, -10, 10)
        h = paddle.abs(m.weight) >= 5
        h = paddle.to_tensor(h)
        m = paddle.to_tensor(m.weight)
        m *= h       

net.apply(my_init)
net[0].weight[:2]
```

Обратите внимание, что мы всегда можем задавать параметры напрямую.

**MXNet:**
```python
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

**PyTorch:**
```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

**TensorFlow:**
```python
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

**PaddlePaddle:**
```python
net[0].weight.set_value(net[0].weight.numpy() + 1)
val = net[0].weight.numpy()
val[0, 0] = 42
net[0].weight.set_value(val)
net[0].weight[0]
```

**MXNet:** Для продвинутых пользователей: если нужно изменять параметры внутри области `autograd`, следует использовать `set_data`, чтобы не вводить в заблуждение механизм автоматического дифференцирования.

## 5.2.3 Связывание параметров

Иногда нам хочется совместно использовать параметры между несколькими слоями:
мы можем определить один плотный слой, а затем использовать его параметры для задания параметров другого слоя.

**MXNet:**
```python
net = nn.Sequential()
# Нужно дать общему слою имя, чтобы можно было обращаться к его параметрам
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Проверяем, совпадают ли параметры
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Убедимся, что это действительно один и тот же объект, а не просто совпадающие значения
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

**PyTorch:**
```python
# Нужно дать общему слою имя, чтобы можно было обращаться к его параметрам
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Проверяем, совпадают ли параметры
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Убедимся, что это действительно один и тот же объект, а не просто совпадающие значения
print(net[2].weight.data[0] == net[4].weight.data[0])
```

**TensorFlow:**
```python
# tf.keras ведёт себя немного иначе: он автоматически удаляет повторяющиеся слои
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Проверяем, различаются ли параметры
print(len(net.layers) == 3)
```

**PaddlePaddle:**
```python
# Нужно дать общему слою имя, чтобы можно было обращаться к его параметрам.
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Проверяем, совпадают ли параметры
print(net[2].weight[0] == net[4].weight[0])
```

**MXNet:** Этот пример показывает, что параметры второго и третьего слоёв связаны.
Они не только имеют одинаковые значения, но и представлены одним и тем же тензором.
Поэтому, если мы изменим один из параметров, изменится и другой.
Возникает вопрос: что происходит с градиентами, когда параметры связаны?
Ответ: поскольку модельные параметры содержат градиенты, при обратном распространении градиенты второго и третьего скрытых слоёв складываются.

**PyTorch:** Этот пример показывает, что параметры третьего и пятого слоёв нейронной сети связаны.
Они не только имеют одинаковые значения, но и представлены одним и тем же тензором.
Поэтому, если мы изменим один из параметров, изменится и другой.
Возникает вопрос: что происходит с градиентами, когда параметры связаны?
Ответ: поскольку модельные параметры содержат градиенты, при обратном распространении градиенты второго скрытого слоя (то есть третьего слоя сети) и третьего скрытого слоя (то есть пятого слоя сети) складываются.

## Резюме

* Существует несколько способов доступа к параметрам модели, их инициализации и связывания.
* Мы можем использовать пользовательские методы инициализации.

## Упражнения

1. Используя модель `FancyMLP`, определённую в разделе 5.1, получите доступ к параметрам её слоёв.
1. Изучите документацию модуля инициализации, чтобы узнать о различных методах инициализации.
1. Постройте многослойный перцептрон с общими параметрами слоёв и обучите его. Во время обучения наблюдайте за параметрами и градиентами слоёв модели.
1. Почему совместное использование параметров — хорошая идея?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/1831)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/1829)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/1830)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11778)
