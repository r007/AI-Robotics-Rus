# 8.2 Предобработка текста

В разделе 8.1 мы рассмотрели статистические инструменты, которые нужны для работы с последовательными данными, и трудности, возникающие при предсказании.
Такие данные встречаются во многих формах, и текст — один из самых распространённых примеров.
Например, статью можно рассматривать как последовательность слов или даже как последовательность символов.
В этом разделе мы разберём типичные шаги предобработки текста.
Обычно они включают следующее:

1. загрузку текста в память в виде строк;
1. разбиение строк на токены (например, на слова или символы);
1. построение словаря, который сопоставляет разбитые токены с числовыми индексами;
1. преобразование текста в последовательности числовых индексов, с которыми может работать модель.

**MXNet:**
```python
import collections
from d2l import mxnet as d2l
import re
```

**PyTorch:**
```python
import collections
from d2l import torch as d2l
import re
```

**TensorFlow:**
```python
import collections
from d2l import tensorflow as d2l
import re
```

**PaddlePaddle:**
```python
import collections
from d2l import paddle as d2l
import re
```

## 8.2.1 Чтение набора данных

Для начала загрузим текст книги Герберта Уэллса [«Машина времени»](https://www.gutenberg.org/ebooks/35).
Это довольно небольшой корпус: в нём немногим более 30 000 слов. Для первых экспериментов его достаточно, тогда как реальные коллекции документов могут содержать миллиарды слов.
Функция ниже **считывает набор данных в список строк текста**.
Для простоты мы здесь не учитываем знаки препинания и регистр букв.

```python
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Загружает набор данных «Машина времени» в список текстовых строк"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# Общее число строк текста: {len(lines)}')
print(lines[0])
print(lines[10])
```

## 8.2.2 Токенизация

Функция `tokenize` ниже принимает на вход список текстовых строк `lines`, каждый элемент которого — отдельная текстовая последовательность (например, одна строка текста).
**Каждая текстовая последовательность разбивается на список токенов**. *Токен* (token) — это базовая единица текста.
В итоге функция возвращает список списков токенов, где каждый токен является строкой.

```python
def tokenize(lines, token='word'):  #@save
    """Разбивает строки текста на токены-слова или токены-символы"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ошибка: неизвестный тип токена: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## 8.2.3 Словарь

Токены имеют строковый тип, а модели на вход нужны числа.
Теперь построим **словарь** (*vocabulary*), который сопоставляет строковые токены с числовыми индексами, начиная с $0$.
Для этого мы сначала объединяем все документы обучающего набора, подсчитываем уникальные токены и получаем статистику, которую называют *корпусом* (corpus).
Затем каждому уникальному токену присваиваем числовой индекс в соответствии с частотой его появления.
Очень редкие токены обычно удаляют, чтобы снизить сложность.
Кроме того, любой токен, который отсутствует в корпусе или был удалён, отображается в специальный неизвестный токен «`<unk>`».
При желании можно добавить список зарезервированных токенов, например:
токен заполнения «`<pad>`»,
токен начала последовательности «`<bos>`»,
токен конца последовательности «`<eos>`».

```python
class Vocab:  #@save
    """Словарь текста"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Сортировка по частоте появления
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # Индекс неизвестного токена равен 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Индекс неизвестного токена равен 0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Подсчёт частот токенов"""
    # Здесь tokens — одномерный или двумерный список
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Превращаем список списков токенов в один список
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

Построим **словарь** на основе набора данных «Машина времени» и выведем первые самые частые токены и их индексы.

```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

Теперь мы можем **преобразовать каждую строку текста в список числовых индексов**.

```python
for i in [0, 10]:
    print('текст:', tokens[i])
    print('индексы:', vocab[tokens[i]])
```

## 8.2.4 Собираем всё вместе

Чтобы использовать описанные функции, **объединим все шаги в функции `load_corpus_time_machine`**, которая возвращает `corpus` (список индексов токенов) и `vocab` (словарь корпуса «Машина времени»).
Здесь мы вносим два изменения по сравнению с предыдущими шагами:

1. Чтобы упростить обучение в следующих разделах, мы токенизируем текст по символам, а не по словам;
1. Строка набора данных «Машина времени» не обязательно является предложением или абзацем, а иногда может быть одним словом, поэтому возвращаемый `corpus` — это один плоский список, а не список списков токенов.

```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Возвращает список индексов токенов и словарь набора данных «Машина времени»"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Строки текста из «Машины времени» не обязательно являются предложениями или абзацами,
    # поэтому все строки разворачиваем в один список
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Резюме

* Текст — один из самых распространённых видов последовательных данных.
* Чтобы предобработать текст, обычно разбивают его на токены, строят словарь, который сопоставляет строковые токены с числовыми индексами, и преобразуют текст в индексы токенов, с которыми может работать модель.

## Упражнения

1. Токенизация — ключевой шаг предобработки, и она зависит от языка. Попробуйте найти ещё три распространённых способа токенизации текста.
1. В экспериментах этого раздела разбейте текст на слова и измените параметр `min_freq` экземпляра `Vocab`. Как это влияет на размер словаря?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2093)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2094)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/2095)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11796)
