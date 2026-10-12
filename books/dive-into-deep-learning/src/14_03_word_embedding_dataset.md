# 14.3 Набор данных для предобучения вложений слов

Теперь, когда мы знаем технические детали модели word2vec и общий порядок её обучения, перейдём к реализации. Конкретно, в качестве примера мы возьмём модель skip-gram из раздела 14.1 и отрицательную выборку из раздела 14.2. Этот раздел начинается с набора данных для предобучения модели вложений слов: исходный формат данных будет преобразован в минибатчи, по которым можно итерироваться во время обучения.

**MXNet:**

```python
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

**PyTorch:**

```python
from d2l import torch as d2l
import math
import torch
import os
import random
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
import os
import random
```

## 14.3.1 Чтение набора данных

Мы используем набор данных [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42). Этот корпус составлен из статей «Уолл-стрит джорнал» и разделён на обучающую, проверочную и тестовую части. В исходном формате каждая строка текстового файла — одно предложение, слова в котором разделены пробелами. Каждое слово мы будем считать токеном.

```python
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Загружает набор данных PTB в список строк текста"""
    data_dir = d2l.download_extract('ptb')
    # Читаем обучающую часть
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# предложений: {len(sentences)}'
```

После чтения обучающей части мы строим словарь корпуса: все слова, встречающиеся меньше 10 раз, заменяются токеном `<unk>`. Заметим, что исходный набор данных уже содержит токены `<unk>`, обозначающие редкие (неизвестные) слова.

```python
vocab = d2l.Vocab(sentences, min_freq=10)
f'размер словаря: {len(vocab)}'
```

## 14.3.2 Прореживание частых слов

В текстовых данных обычно есть частые слова — «the», «a», «in»: в очень большом корпусе они могут встречаться миллиарды раз. Однако такие слова часто встречаются в контекстных окнах вместе со множеством разных слов и дают мало полезной информации. Например, рассмотрим слово «chip» в контекстном окне: интуитивно его совместная встречаемость с редким словом «intel» полезнее для обучения, чем с частым словом «a». Кроме того, обучение на большом числе (частых) слов идёт медленно. Поэтому при обучении модели вложений слов частые слова можно *прореживать* (subsampling) (Mikolov et al., 2013b). Точнее, каждое слово $w_i$ набора данных будет отброшено с вероятностью

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

где $f(w_i)$ — отношение числа вхождений слова $w_i$ к общему числу слов набора данных, а константа $t$ — гиперпараметр (в эксперименте $10^{-4}$). Видно, что (частое) слово $w_i$ может быть отброшено, только когда его относительная частота $f(w_i) > t$, причём чем выше относительная частота слова, тем больше вероятность его отбросить.

```python
#@save
def subsample(sentences, vocab):
    """Прореживает частые слова"""
    # Исключаем неизвестные токены '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Возвращает True, если токен сохраняется при прореживании
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Следующий фрагмент кода строит гистограмму числа токенов в предложении до и после прореживания. Как и ожидалось, прореживание заметно укорачивает предложения, удаляя частые слова, что ускорит обучение.

```python
d2l.show_list_len_pair_hist(
    ['до прореживания', 'после прореживания'], 'токенов в предложении',
    'количество', sentences, subsampled);
```

Для отдельных токенов видно, что частое слово «the» сохраняется с вероятностью менее 1/20.

```python
def compare_counts(token):
    return (f'количество "{token}": '
            f'до={sum([l.count(token) for l in sentences])}, '
            f'после={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Напротив, редкое слово «join» сохраняется полностью.

```python
compare_counts('join')
```

После прореживания мы отображаем токены в их индексы в корпусе.

```python
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 14.3.3 Извлечение центральных и контекстных слов

Следующая функция `get_centers_and_contexts` извлекает из `corpus` все центральные слова и их контекстные слова. Для каждого центрального слова она случайно выбирает целое число от 1 до `max_window_size` в качестве размера контекстного окна. Контекстными словами центрального слова считаются слова, расстояние до которых не превышает выбранный размер окна.

```python
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Возвращает центральные и контекстные слова модели skip-gram"""
    centers, contexts = [], []
    for line in corpus:
        # Чтобы получить пару «центральное слово — контекстное слово»,
        # в предложении должно быть не меньше 2 слов
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Контекстное окно с центром в i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Исключаем центральное слово из контекстных
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Далее создадим искусственный набор данных из двух предложений по 7 и 3 слова. Зададим максимальный размер контекстного окна 2 и выведем все центральные слова и их контекстные слова.

```python
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('набор данных', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('центральное слово', center, '— контекстные слова:', context)
```

При обучении на наборе данных PTB мы задаём максимальный размер контекстного окна 5. Ниже извлекаются все центральные слова набора данных и их контекстные слова.

```python
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# пар «центральное слово — контекстное слово»: {sum([len(contexts) for contexts in all_contexts])}'
```

## 14.3.4 Отрицательная выборка

Для приближённого обучения мы используем отрицательную выборку. Чтобы выбирать шумовые слова по заранее заданному распределению, определим следующий класс `RandomGenerator`, в который (возможно, ненормированное) распределение выборки передаётся через переменную `sampling_weights`.

```python
#@save
class RandomGenerator:
    """Случайно выбирает из {1, ..., n} согласно n весам выборки"""
    def __init__(self, sampling_weights):
        # Индекс 0 исключён
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Кэшируем k случайных результатов выборки
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Например, мы можем выбрать 10 значений случайной величины $X$ из индексов 1, 2 и 3 с вероятностями $P(X=1)=2/9, P(X=2)=3/9$ и $P(X=3)=4/9$ следующим образом.

```python
#@save
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Для пары «центральное слово — контекстное слово» мы случайно выбираем `K` (в эксперименте 5) шумовых слов. Следуя рекомендации статьи о word2vec, вероятность выбора шумового слова $w$, $P(w)$, полагается равной его относительной частоте в словаре, возведённой в степень 0.75 (Mikolov et al., 2013b).

```python
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Возвращает шумовые слова для отрицательной выборки"""
    # Индексы 1, 2, ... (индекс 0 — неизвестный токен, исключённый из словаря)
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Шумовое слово не может быть контекстным
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 14.3.5 Загрузка обучающих примеров минибатчами

После извлечения всех центральных слов с их контекстными словами и выборки шумовых слов мы преобразуем их в минибатчи примеров, по которым можно итерироваться во время обучения.

В минибатче $i$-й пример состоит из центрального слова, его $n_i$ контекстных слов и $m_i$ шумовых слов. Из-за разного размера контекстного окна $n_i+m_i$ различается для разных $i$. Поэтому для каждого примера мы конкатенируем его контекстные и шумовые слова в переменной `contexts_negatives` и дополняем нулями, пока длина конкатенации не достигнет $\max_i n_i+m_i$ (`max_len`). Чтобы исключить дополнение при вычислении потерь, мы определяем маску `masks`. Элементы `masks` и `contexts_negatives` находятся во взаимно однозначном соответствии: нули в `masks` отвечают дополнению в `contexts_negatives` (иначе в `masks` стоят единицы).

Чтобы различать положительные и отрицательные примеры, в `contexts_negatives` мы отделяем контекстные слова от шумовых с помощью переменной `labels`. Как и для `masks`, между элементами `labels` и `contexts_negatives` есть взаимно однозначное соответствие: единицы в `labels` отвечают контекстным словам (положительным примерам) в `contexts_negatives`, а в остальных позициях стоят нули.

Эта идея реализована в следующей функции `batchify`. Её вход `data` — список длины, равной размеру минибатча, в котором каждый элемент — пример из центрального слова `center`, его контекстных слов `context` и его шумовых слов `negative`. Функция возвращает минибатч, который можно загружать для вычислений во время обучения, в том числе с переменной-маской.

```python
#@save
def batchify(data):
    """Возвращает минибатч примеров для модели skip-gram с отрицательной выборкой"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Проверим эту функцию на минибатче из двух примеров.

```python
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 14.3.6 Сборка кода

Наконец, определим функцию `load_data_ptb`, которая читает набор данных PTB и возвращает итератор данных и словарь.

**MXNet:**

```python
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Загружает набор данных PTB и помещает его в память"""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

**PyTorch:**

```python
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Загружает набор данных PTB и помещает его в память"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, 
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab
```

**PaddlePaddle:**

```python
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Загружает набор данных PTB и помещает его в память"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(paddle.io.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = paddle.io.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, return_list=True, 
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab
```

Выведем первый минибатч итератора данных.

```python
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Резюме

* Частые слова могут оказаться не очень полезными при обучении. Их можно прореживать, чтобы ускорить обучение.
* Для повышения эффективности вычислений мы загружаем примеры минибатчами. Можно определить дополнительные переменные, которые отличают дополняющие элементы от остальных, а также положительные примеры от отрицательных.

## Упражнения

1. Как изменится время работы кода этого раздела, если не использовать прореживание?
1. Класс `RandomGenerator` кэширует `k` случайных результатов выборки. Задайте для `k` другие значения и посмотрите, как это влияет на скорость загрузки данных.
1. Какие ещё гиперпараметры кода этого раздела могут влиять на скорость загрузки данных?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5734)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5735)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11816)

## Литература

* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. In *Advances in neural information processing systems*, 3111–3119.
