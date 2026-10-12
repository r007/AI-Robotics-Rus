# 15.1 Анализ тональности и набор данных

С быстрым развитием социальных сетей и платформ отзывов накапливается огромное количество отзывов. Эти данные обладают большим потенциалом для поддержки принятия решений. *Анализ тональности* (sentiment analysis) изучает «скрытые» в тексте настроения людей — в отзывах о товарах, комментариях в блогах, обсуждениях на форумах и т. д. Он широко применяется в политике (например, анализ отношения общества к государственным решениям), финансах (например, анализ рыночных настроений) и маркетинге (например, исследование продуктов и управление брендом).

Поскольку тональность можно классифицировать по дискретным полярностям или шкалам (например, положительная и отрицательная), анализ тональности можно рассматривать как задачу классификации текстов, которая преобразует текстовую последовательность переменной длины в категорию текста фиксированной длины. В этом разделе мы будем использовать для анализа тональности [большой набор данных рецензий на фильмы (large movie review dataset)](https://ai.stanford.edu/~amaas/data/sentiment/) Стэнфордского университета. Он состоит из обучающей и тестовой частей и содержит по 25 000 рецензий на фильмы, загруженных с IMDb. В обеих частях поровну «положительных» и «отрицательных» меток, обозначающих разные полярности тональности.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
from torch import nn
import os
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import os
```

## 15.1.1 Чтение набора данных

Сначала скачаем и распакуем набор данных рецензий IMDb в каталог `../data/aclImdb`.

```python
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Затем прочитаем обучающий и тестовый наборы данных. Каждый пример — рецензия с меткой: 1 означает «положительная», 0 — «отрицательная».

```python
#@save
def read_imdb(data_dir, is_train):
    """Читает текстовые последовательности и метки набора рецензий IMDb"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('число обучающих примеров:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('метка:', y, 'рецензия:', x[0:60])
```

## 15.1.2 Предобработка набора данных

Считая каждое слово токеном и отфильтровывая слова, встречающиеся меньше 5 раз, создадим словарь по обучающему набору данных.

```python
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

После токенизации построим гистограмму длин рецензий в токенах.

```python
d2l.set_figsize()
d2l.plt.xlabel('# токенов в рецензии')
d2l.plt.ylabel('количество')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Как и ожидалось, длины рецензий различаются. Чтобы обрабатывать такие рецензии минибатчами, мы приводим длину каждой рецензии к 500 с помощью усечения и дополнения. Это похоже на этапы предобработки набора данных машинного перевода из раздела 9.5.

```python
num_steps = 500  # Длина последовательности
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## 15.1.3 Создание итераторов данных

Теперь можно создать итераторы данных. На каждой итерации возвращается минибатч примеров.

**MXNet:**

```python
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('число минибатчей:', len(train_iter))
```

**PyTorch:**

```python
train_iter = d2l.load_array((train_features, 
    torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('число минибатчей:', len(train_iter))
```

**PaddlePaddle:**

```python
train_iter = d2l.load_array((train_features,
    d2l.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('число минибатчей:', len(train_iter))
```

## 15.1.4 Сборка кода

Наконец, мы упаковываем описанные шаги в функцию `load_data_imdb`. Она возвращает итераторы обучающих и тестовых данных и словарь набора рецензий IMDb.

**MXNet:**

```python
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Возвращает итераторы данных и словарь набора рецензий IMDb"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

**PyTorch:**

```python
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Возвращает итераторы данных и словарь набора рецензий IMDb"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

**PaddlePaddle:**

```python
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Возвращает итераторы данных и словарь набора рецензий IMDb"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = d2l.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = d2l.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, d2l.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, d2l.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Резюме

* Анализ тональности изучает настроения людей в тексте; он рассматривается как задача классификации текстов, преобразующая текстовую последовательность переменной длины в категорию текста фиксированной длины.
* После предобработки с помощью словаря можно загрузить набор рецензий IMDb в итераторы данных.

## Упражнения

1. Какие гиперпараметры из этого раздела можно изменить, чтобы ускорить обучение модели анализа тональности?
1. Реализуйте функцию загрузки набора данных [отзывов Amazon](https://snap.stanford.edu/data/web-Amazon.html) в итераторы данных для анализа тональности.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5725)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5726)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11825)
