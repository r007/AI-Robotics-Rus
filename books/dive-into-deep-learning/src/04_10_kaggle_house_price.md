# 4.10 Практика: соревнование Kaggle по прогнозированию цен на жильё

В предыдущих разделах мы познакомились с базовыми инструментами обучения глубоких сетей и методами регуляризации (такими как затухание весов и исключение).
В этом разделе мы применим изученное на практике, участвуя в соревновании на Kaggle.
Соревнование по прогнозированию цен на жильё на Kaggle — хорошая отправная точка.
Этот набор данных собрал Барт Де Кок в 2011 году (De Cock, 2011); он охватывает цены на жильё в городе Эймс (штат Айова) в 2006–2010 годах.
Набор довольно универсален, и для него не нужна сложная архитектура модели.
Он гораздо больше и содержит больше признаков, чем известный набор данных о ценах на жильё в Бостоне, собранный Харрисоном и Рубинфельдом.

В этом разделе мы подробно рассмотрим предобработку данных, проектирование модели и выбор гиперпараметров.
Проделав всё самостоятельно, вы получите практический опыт, который пригодится в работе специалиста по данным.

## 4.10.1 Загрузка и кэширование набора данных

На протяжении всей книги мы будем загружать различные наборы данных и обучать и тестировать на них модели.
Здесь мы **реализуем несколько функций, которые упрощают загрузку данных**.
Сначала создадим словарь `DATA_HUB`, который сопоставляет строковое имя набора данных с парой значений: URL набора данных и хеш-суммой SHA-1 для проверки целостности файла.
Все такие наборы данных размещены на сайте по адресу `DATA_URL`.

```python
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

Функция `download` ниже загружает набор данных, кэширует его в локальном каталоге (по умолчанию `../data`) и возвращает имя загруженного файла.
Если файл набора данных уже есть в каталоге кэша и его SHA-1 совпадает с значением из `DATA_HUB`, будет использован кэшированный файл, чтобы не скачивать его повторно.

```python
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Скачать файл из DATA_HUB и вернуть локальное имя файла"""
    assert name in DATA_HUB, f"{name} отсутствует в {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Попадание в кэш
    print(f'Загрузка {fname} из {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

Нам также понадобятся две вспомогательные функции: одна скачивает и распаковывает zip- или tar-архив, а другая скачивает в кэш все наборы данных из `DATA_HUB`, используемые в книге.

```python
def download_extract(name, folder=None):  #@save
    """Скачать и распаковать zip/tar-файл"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Распаковывать можно только zip/tar-файлы'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Скачать все файлы из DATA_HUB"""
    for name in DATA_HUB:
        download(name)
```

## 4.10.2 Kaggle

[Kaggle](https://www.kaggle.com) — популярная сегодня платформа для соревнований по машинному обучению; каждое соревнование строится вокруг как минимум одного набора данных.
Многие соревнования спонсируются организациями, которые выплачивают премии за лучшие решения.
Платформа помогает участникам общаться через форумы и обмениваться кодом, что способствует сотрудничеству и соперничеству.
Погоня за местом в таблице лидеров нередко отвлекает от главного: некоторые исследователи близоруко сосредотачиваются на этапах предобработки, вместо того чтобы думать о фундаментальных вопросах.
Но объективная платформа очень ценна: она позволяет напрямую и количественно сравнивать конкурирующие методы и делиться кодом.
Так каждый может узнать, какие методы работают, а какие — нет.
Чтобы участвовать в соревнованиях Kaggle, сначала нужно зарегистрировать аккаунт.

На странице соревнования по прогнозированию цен на жильё данные можно найти на вкладке «Data».
Прогнозы можно отправить по адресу, указанному ниже, там же можно посмотреть свою позицию в рейтинге:

>https://www.kaggle.com/c/house-prices-advanced-regression-techniques

*Скриншоты страниц сайта Kaggle в книгу не включены: это страницы стороннего сайта.*

## 4.10.3 Чтение набора данных

Обратите внимание, что данные соревнования разделены на обучающий и тестовый наборы.
Каждая запись содержит характеристики дома и его атрибуты: тип улицы, год постройки, тип крыши, состояние подвала и так далее.
Признаки имеют разные типы данных.
Например, год постройки задан целым числом, тип крыши — дискретной категорией, остальные признаки — вещественными числами.
Здесь реальность усложняет задачу: например, некоторые данные полностью отсутствуют, и пропуски просто помечены как «NA».
Цена каждого дома указана только в обучающем наборе (в конце концов, это соревнование).
Мы хотим разделить обучающий набор, чтобы создать проверочный набор, но после отправки прогнозов в Kaggle оценить модель мы сможем только на официальном тестовом наборе.
На вкладке «Data» есть ссылки для скачивания данных.

Прежде чем начать, **загрузим и обработаем данные с помощью `pandas`**, с которым мы познакомились в разделе 2.2.
Поэтому перед продолжением нужно убедиться, что `pandas` установлен.
К счастью, если вы читаете эту книгу в Jupyter, можно установить `pandas`, не покидая блокнота.

**MXNet:**
```python
# Если pandas не установлен, раскомментируйте следующую строку
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

**PyTorch:**
```python
# Если pandas не установлен, раскомментируйте следующую строку
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

**TensorFlow:**
```python
# Если pandas не установлен, раскомментируйте следующую строку
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

**PaddlePaddle:**
```python
# Если pandas не установлен, раскомментируйте следующую строку
# !pip install pandas

%matplotlib inline
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn
warnings.filterwarnings("ignore", category=DeprecationWarning)
from d2l import paddle as d2l
```

Для удобства можно использовать описанный выше скрипт, чтобы скачать и закэшировать набор данных о домах Kaggle.

```python
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

Загрузим два CSV-файла с обучающими и тестовыми данными с помощью `pandas`.

```python
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

Обучающий набор содержит 1460 примеров, каждый из которых имеет 80 признаков и 1 метку.
Тестовый набор содержит 1459 примеров, каждый из которых имеет 80 признаков.

```python
print(train_data.shape)
print(test_data.shape)
```

Посмотрим **на первые четыре и последние два признака, а также на соответствующие метки** (цены домов).

```python
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

Видно, что в каждом примере **первый признак — идентификатор**.
Он помогает модели различать обучающие примеры.
Хотя это удобно, для прогноза этот признак никакой информации не несёт.
Поэтому, прежде чем передавать данные модели, **мы удаляем его из набора данных**.

```python
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 4.10.4 Предобработка данных

Как уже сказано, данные имеют разные типы.
Прежде чем начать моделирование, нужно их предобработать.
Сначала **заменим все пропущенные значения средним значением соответствующего признака.**
Затем, чтобы привести все признаки к единому масштабу, **стандартизуем данные, приводя признаки к нулевому среднему и единичной дисперсии**:

$$x \leftarrow \frac{x - \mu}{\sigma},$$

где $\mu$ и $\sigma$ — среднее значение и стандартное отклонение соответственно.
Теперь признаки имеют нулевое среднее и единичную дисперсию, то есть $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$ и $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Интуитивно стандартизация данных полезна по двум причинам.
Во-первых, это упрощает оптимизацию.
Во-вторых, поскольку мы не знаем, какие признаки связаны между собой, мы не хотим, чтобы штраф назначал коэффициент одному признаку больше, чем другим.

```python
# Если тестовые данные недоступны, среднее и стандартное отклонение можно вычислить по обучающим данным
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# После стандартизации все средние равны нулю, поэтому пропуски можно заменить нулём
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

Далее **обработаем дискретные значения.**
К ним относятся такие признаки, как «MSZoning».
**Мы заменяем их прямым кодированием (one-hot)**, тем же способом, что и при преобразовании многоклассовых меток в векторы ранее (см. раздел 3.4.1).
Например, признак «MSZoning» содержит значения «RL» и «Rm».
Мы создадим два новых индикаторных признака «MSZoning_RL» и «MSZoning_RM» со значениями 0 или 1.
Согласно прямому кодированию, если исходное значение «MSZoning» — «RL», то «MSZoning_RL» равно 1, а «MSZoning_RM» равно 0.
Пакет `pandas` делает это за нас автоматически.

```python
# Dummy_na=True считает «na» (пропущенное значение) допустимым значением признака и создаёт для него индикаторный признак
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

Видно, что это преобразование увеличивает общее число признаков с 79 до 331.
Наконец, через атрибут `values` мы
**извлекаем данные из формата `pandas` в формат NumPy и преобразуем их в тензоры** для обучения.

```python
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## 4.10.5 Обучение

Для начала обучим линейную модель с квадратичной функцией потерь.
Линейная модель вряд ли позволит выиграть соревнование, но она служит проверкой здравого смысла: есть ли в данных осмысленная информация.
Если здесь мы не сможем работать лучше случайного угадывания, скорее всего, в обработке данных есть ошибка.
Если всё пойдёт хорошо, линейная модель будет служить *базовой* (baseline) моделью, и мы поймём, насколько лучшая модель превосходит простую.

**MXNet:**
```python
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

**PyTorch, PaddlePaddle:**
```python
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

**TensorFlow:**
```python
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

Цены на жильё, как и цены акций, нас интересуют в относительных величинах, а не в абсолютных.
Поэтому **мы больше заботимся об относительной ошибке $\frac{y - \hat{y}}{y}$**, чем об абсолютной ошибке $y - \hat{y}$.
Например, если мы оцениваем стоимость дома в сельской местности Огайо и ошибаемся на 100 000 долларов, при том что типичный дом там стоит 125 000 долларов, то модель, скорее всего, работает плохо.
С другой стороны, та же ошибка в 100 000 долларов для особняка в районе Калифорнии, где медианная цена превышает 4 миллиона долларов, может означать хороший прогноз.

**Один из способов решить эту проблему — измерять расхождение логарифмом цены.**
Фактически именно эту метрику ошибки официально использует соревнование для оценки качества решений.
Условие $|\log y - \log \hat{y}| \leq \delta$ равносильно $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
Поэтому среднеквадратичная ошибка между логарифмом предсказанной цены и логарифмом истинной метки:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

**MXNet:**
```python
def log_rmse(net, features, labels):
    # Чтобы стабилизировать значение при взятии логарифма, значения меньше 1 заменяем на 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

**PyTorch:**
```python
def log_rmse(net, features, labels):
    # Чтобы стабилизировать значение при взятии логарифма, значения меньше 1 заменяем на 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

**TensorFlow:**
```python
def log_rmse(y_true, y_pred):
    # Чтобы стабилизировать значение при взятии логарифма, значения меньше 1 заменяем на 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

**PaddlePaddle:**
```python
def log_rmse(net, features, labels):
    # Чтобы стабилизировать значение при взятии логарифма, значения меньше 1 заменяем на 1
    clipped_preds = paddle.clip(net(features), 1, float('inf'))
    rmse = paddle.sqrt(loss(paddle.log(clipped_preds),
                            paddle.log(labels)))
    return rmse.item()
```

В отличие от предыдущих разделов, **наша функция обучения будет опираться на оптимизатор Adam**
(его подробно опишем в следующих главах).
Главное достоинство Adam в том, что он менее чувствителен к выбору начальной скорости обучения.

**MXNet:**
```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # Здесь используется оптимизатор Adam
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

**PyTorch:**
```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # Здесь используется оптимизатор Adam
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

**TensorFlow:**
```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # Здесь используется оптимизатор Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

**PaddlePaddle:**
```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # Здесь используется оптимизатор Adam
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate*1.0, 
                                      parameters=net.parameters(), 
                                      weight_decay=weight_decay*1.0)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            optimizer.clear_grad()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

## 4.10.6 K-кратная перекрёстная проверка

В разделе, посвящённом выбору модели (раздел 4.4), мы описали **K-кратную перекрёстную проверку**.
Она помогает выбирать модель и настраивать гиперпараметры.
Сначала определим функцию, которая возвращает данные i-го фолда при K-кратной перекрёстной проверке.
Конкретно, она выбирает i-й фрагмент в качестве проверочных данных, а остальные части — в качестве обучающих.
Заметим, что это не самый эффективный способ работы с данными: при гораздо большем наборе данных могут подойти другие решения.

```python
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

Когда мы обучили модель K раз в K-кратной перекрёстной проверке, **возвращаем среднее значение ошибок обучения и проверки**.

```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='эпоха', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['обучение', 'проверка'], yscale='log')
        print(f'Фолд {i + 1}, log rmse на обучении {float(train_ls[-1]):f}, '
              f'log rmse на проверке {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## 4.10.7 Выбор модели

В этом примере мы выбираем набор ненастроенных гиперпараметров и оставляем читателю возможность улучшить модель.
Поиск хорошего набора гиперпараметров может занять время — в зависимости от того, сколько переменных мы оптимизируем.
При достаточно большом наборе данных и разумно заданных гиперпараметрах K-кратная перекрёстная проверка обычно даёт устойчивые результаты при повторных запусках.
Однако если мы опробуем неудачные гиперпараметры, проверочная ошибка может перестать отражать истинную ошибку.

```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-кратная проверка: средняя log rmse на обучении: {float(train_l):f}, '
      f'средняя log rmse на проверке: {float(valid_l):f}')
```

Обратите внимание: иногда ошибка обучения для набора гиперпараметров может быть очень низкой, а ошибка K-кратной перекрёстной проверки — намного выше.
Это указывает на переобучение модели.
На протяжении всего обучения мы хотим следить за двумя величинами: ошибкой обучения и ошибкой проверки.
Небольшое переобучение может говорить о том, что имеющихся данных достаточно для более мощной модели.
Сильное переобучение, напротив, может означать, что стоит воспользоваться методами регуляризации.

## 4.10.8 Отправка прогнозов на Kaggle

Теперь, когда мы знаем, какие гиперпараметры выбрать, можно обучить модель на всех данных
(а не только на 1 − 1/K от данных, как в перекрёстной проверке).
Затем полученную модель можно применить к тестовому набору.
Сохранение прогнозов в CSV-файл упрощает отправку результатов на Kaggle.

```python
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='эпоха',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'log rmse на обучении: {float(train_ls[-1]):f}')
    # Применить сеть к тестовому набору
    preds = d2l.numpy(net(test_features))
    # Преобразовать формат для экспорта в Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

Если прогнозы на тестовом наборе похожи на прогнозы, полученные во время K-кратной перекрёстной проверки, пора загружать их на Kaggle.
Следующий код создаст файл `submission.csv`.

```python
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

Далее прогнозы можно отправить на Kaggle и посмотреть, как они соотносятся с реальными ценами домов (метками) на тестовом наборе.
Процедура очень простая.

* Войдите на сайт Kaggle и откройте страницу соревнования по прогнозированию цен на жильё.
* Нажмите кнопку «Submit Predictions» или «Late Submission» (на момент написания этих строк она находится справа).
* Нажмите кнопку «Upload Submission File» в пунктирной рамке внизу страницы и выберите файл с прогнозами.
* Нажмите кнопку «Make Submission» внизу страницы, чтобы увидеть результат.

## Резюме

* Реальные данные обычно содержат смесь разных типов данных, их нужно предобработать.
* Распространённые приёмы предобработки: приведение вещественных данных к нулевому среднему и единичной дисперсии; замена пропущенных значений средним.
* Преобразование категориальных признаков в индикаторные позволяет трактовать их как one-hot вектор.
* Для выбора модели и настройки гиперпараметров можно использовать K-кратную перекрёстную проверку.
* Логарифм полезен, когда важна относительная ошибка.

## Упражнения

1. Насколько хорошо работает ваш прогноз, отправленный на Kaggle?
1. Можно ли улучшить модель, напрямую минимизируя логарифм цены? Что произойдёт, если предсказывать логарифм цены, а не саму цену?
1. Всегда ли хорошая идея заменять пропущенные значения средними? Подсказка: можно ли построить ситуацию, в которой значения пропадают не случайно?
1. Улучшите результат на Kaggle, настраивая гиперпараметры с помощью K-кратной перекрёстной проверки.
1. Улучшите модель (например, слои, затухание весов и исключение), чтобы повысить результат.
1. Что произойдёт, если мы не стандартизируем непрерывные числовые признаки, как это сделано в этом разделе?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/1823)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/1824)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/1825)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11775)
