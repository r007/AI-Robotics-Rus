# 15.4 Логический вывод на естественном языке и набор данных

В разделе 15.1 мы обсуждали задачу анализа тональности. Цель этой задачи — отнести одну текстовую последовательность к одной из заранее определённых категорий, например к одной из полярностей тональности. Однако когда нужно решить, можно ли одно предложение вывести из другого, или устранить избыточность между предложениями, выявляя семантически эквивалентные, одного умения классифицировать отдельную текстовую последовательность недостаточно. Вместо этого нужно уметь делать выводы о парах текстовых последовательностей.

## 15.4.1 Логический вывод на естественном языке

*Логический вывод на естественном языке* (natural language inference) изучает, можно ли *гипотезу* (hypothesis) вывести из *посылки* (premise), причём и то, и другое — текстовые последовательности. Иными словами, логический вывод на естественном языке определяет логическое отношение между парой текстовых последовательностей. Такие отношения обычно делятся на три типа:

* *Следование* (entailment): гипотезу можно вывести из посылки.
* *Противоречие* (contradiction): отрицание гипотезы можно вывести из посылки.
* *Нейтральность* (neutral): все остальные случаи.

Логический вывод на естественном языке также называют задачей распознавания текстового следования. Например, следующая пара получит метку «следование», потому что «проявляют нежность» в гипотезе можно вывести из «обнимают друг друга» в посылке.

> Посылка: Две женщины обнимают друг друга.

> Гипотеза: Две женщины проявляют нежность.

Ниже пример «противоречия», поскольку «запускает пример кода» означает «не спит», а не «спит».

> Посылка: Мужчина запускает пример кода из книги «Погружение в глубокое обучение».

> Гипотеза: Мужчина спит.

Третий пример показывает отношение «нейтральности», потому что из факта «выступают для нас» нельзя вывести ни «известны», ни «неизвестны».

> Посылка: Музыканты выступают для нас.

> Гипотеза: Музыканты известны.

Логический вывод на естественном языке — центральная тема в понимании естественного языка. Он широко применяется — от информационного поиска до ответов на вопросы в открытой предметной области. Чтобы изучить эту задачу, мы начнём с популярного эталонного набора данных для логического вывода на естественном языке.

## 15.4.2 Набор данных Stanford Natural Language Inference (SNLI)

**Корпус Stanford Natural Language Inference (SNLI)** — это коллекция из более чем 500 000 размеченных пар английских предложений (Bowman et al., 2015). Мы скачиваем извлечённый набор данных SNLI и сохраняем его по пути `../data/snli_1.0`.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### 15.4.2.1 Чтение набора данных

Исходный набор данных SNLI содержит гораздо больше информации, чем нам реально нужно в экспериментах. Поэтому мы определяем функцию `read_snli`, которая извлекает лишь часть набора данных и возвращает списки посылок, гипотез и их меток.

```python
#@save
def read_snli(data_dir, is_train):
    """Разбирает набор данных SNLI на посылки, гипотезы и метки"""
    def extract_text(s):
        # Удаляем информацию, которую мы не будем использовать
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Заменяем два и более подряд идущих пробела одним пробелом
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Теперь **выведем первые 3 пары** посылок и гипотез вместе с их метками («0», «1» и «2» соответствуют «следованию», «противоречию» и «нейтральности»).

```python
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('посылка:', x0)
    print('гипотеза:', x1)
    print('метка:', y)
```

В обучающем наборе около 550 000 пар, а в тестовом — около 10 000. Ниже видно, что в обучающем и тестовом наборах три **метки «следование», «противоречие» и «нейтральность» сбалансированы**.

```python
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### 15.4.2.2 Определение класса для загрузки набора данных

Ниже мы определяем класс для загрузки набора данных SNLI. Переменная `num_steps` конструктора класса задаёт длину текстовой последовательности, чтобы все минибатчи последовательностей имели одинаковую форму. Иными словами, в более длинных последовательностях токены после первых `num_steps` отбрасываются, а к более коротким последовательностям добавляется специальный токен `<pad>`, пока их длина не станет равной `num_steps`. Реализовав `__getitem__`, мы можем произвольно обращаться к посылке, гипотезе и метке с индексом `idx`.

**MXNet:**

```python
#@save
class SNLIDataset(gluon.data.Dataset):
    """Пользовательский набор данных для загрузки SNLI"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('прочитано ' + str(len(self.premises)) + ' примеров')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

**PyTorch:**

```python
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """Пользовательский набор данных для загрузки SNLI"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('прочитано ' + str(len(self.premises)) + ' примеров')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

**PaddlePaddle:**

```python
#@save
class SNLIDataset(paddle.io.Dataset):
    """Пользовательский набор данных для загрузки SNLI"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = paddle.to_tensor(dataset[2])
        print('прочитано ' + str(len(self.premises)) + ' примеров')

    def _pad(self, lines):
        return paddle.to_tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]
    
    def __len__(self):
        return len(self.premises)
```

### 15.4.2.3 Сборка кода

Теперь можно вызвать функцию `read_snli` и класс `SNLIDataset`, чтобы скачать набор данных SNLI и вернуть экземпляры `DataLoader` для обучающего и тестового наборов вместе со словарём обучающего набора. Важно отметить, что в качестве словаря тестового набора нужно использовать словарь, построенный по обучающему набору. Поэтому модель, обученная на обучающем наборе, не будет знать никаких новых токенов из тестового набора.

**MXNet:**

```python
#@save
def load_data_snli(batch_size, num_steps=50):
    """Скачивает набор данных SNLI и возвращает итераторы данных и словарь"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

**PyTorch:**

```python
#@save
def load_data_snli(batch_size, num_steps=50):
    """Скачивает набор данных SNLI и возвращает итераторы данных и словарь"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

**PaddlePaddle:**

```python
#@save
def load_data_snli(batch_size, num_steps=50):
    """Скачивает набор данных SNLI и возвращает итераторы данных и словарь"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = paddle.io.DataLoader(train_set,batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      return_list=True)
                                             
    test_iter = paddle.io.DataLoader(test_set, batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     return_list=True)
    return train_iter, test_iter, train_set.vocab
```

Здесь мы задаём размер минибатча 128 и длину последовательности 50 и вызываем функцию `load_data_snli`, чтобы получить итераторы данных и словарь. Затем выводим размер словаря.

```python
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Теперь выведем форму первого минибатча. В отличие от анализа тональности, здесь у нас два входа `X[0]` и `X[1]`, представляющие пары посылок и гипотез.

```python
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Резюме

* Логический вывод на естественном языке изучает, можно ли «гипотезу» вывести из «посылки», причём обе — текстовые последовательности.
* В логическом выводе на естественном языке отношения между посылкой и гипотезой включают следование, противоречие и нейтральность.
* Корпус Stanford Natural Language Inference (SNLI) — популярный эталонный набор данных для логического вывода на естественном языке.

## Упражнения

1. Машинный перевод издавна оценивался по поверхностному совпадению $n$-грамм между переведённым выходом и эталонным переводом. Можно ли разработать метод оценки результатов машинного перевода с помощью логического вывода на естественном языке?
1. Как изменить гиперпараметры, чтобы уменьшить размер словаря?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5721)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5722)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11828)

## Литература

* Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large annotated corpus for learning natural language inference. *arXiv preprint arXiv:1508.05326*.
