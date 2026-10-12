# 14.7 Сходство слов и задачи на аналогии

В разделе 14.4 мы обучили модель word2vec на небольшом наборе данных и применили её, чтобы находить слова, семантически близкие к заданному. На практике векторы слов, предобученные на больших корпусах, можно применять в прикладных задачах обработки естественного языка, о чём пойдёт речь в главе 15. Чтобы наглядно продемонстрировать семантику предобученных на больших корпусах векторов слов, применим их к задачам на сходство слов и аналогии.

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

## 14.7.1 Загрузка предобученных векторов слов

Ниже перечислены предобученные вложения GloVe размерностей 50, 100 и 300; их можно скачать с [сайта GloVe](https://nlp.stanford.edu/projects/glove/). Предобученные вложения fastText есть для многих языков. Здесь мы используем английскую версию размерности 300 («wiki.en»), которую можно скачать с [сайта fastText](https://fasttext.cc/).

```python
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Чтобы загрузить эти предобученные вложения GloVe и fastText, определим следующий класс `TokenEmbedding`.

```python
#@save
class TokenEmbedding:
    """Вложения токенов (GloVe, fastText)"""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # Сайт GloVe: https://nlp.stanford.edu/projects/glove/
        # Сайт fastText: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Пропускаем заголовок, например первую строку в fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Ниже мы загружаем 50-мерные вложения GloVe (предобученные на части «Википедии»). При создании экземпляра `TokenEmbedding` указанный файл вложений нужно скачать, если этого ещё не сделано.

```python
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Выведем размер словаря. Он содержит 400 000 слов (токенов) и один специальный неизвестный токен.

```python
len(glove_6b50d)
```

Можно получить индекс слова в словаре и наоборот.

```python
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 14.7.2 Применение предобученных векторов слов

С помощью загруженных векторов GloVe мы продемонстрируем семантику векторов слов на следующих задачах на сходство слов и аналогии.

### 14.7.2.1 Сходство слов

Подобно разделу 14.4.3, чтобы находить для заданного слова семантически близкие по косинусному сходству их векторов, мы реализуем следующую функцию `knn` ($k$ ближайших соседей).

**MXNet:**

```python
def knn(W, x, k):
    # Добавляем 1e-9 для численной устойчивости
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

**PyTorch:**

```python
def knn(W, x, k):
    # Добавляем 1e-9 для численной устойчивости
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

**PaddlePaddle:**

```python
def knn(W, x, k):
    # Добавляем 1e-9 для численной устойчивости
    cos = paddle.mv(W, x) / (
        paddle.sqrt(paddle.sum(W * W, axis=1) + 1e-9) *
        paddle.sqrt((x * x).sum()))
    _, topk = paddle.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Затем мы ищем похожие слова, используя предобученные векторы слов из экземпляра `embed` класса `TokenEmbedding`.

```python
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Исключаем входное слово
        print(f'{embed.idx_to_token[int(i)]}: косинусное сходство={float(c):.3f}')
```

Словарь предобученных векторов слов `glove_6b50d` содержит 400 000 слов и один специальный неизвестный токен. Исключив входное слово и неизвестный токен, найдём в словаре три слова, семантически наиболее близких к слову «chip».

```python
get_similar_tokens('chip', 3, glove_6b50d)
```

Ниже выводятся слова, близкие к «baby» и «beautiful».

```python
get_similar_tokens('baby', 3, glove_6b50d)
```

```python
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 14.7.2.2 Аналогии слов

Помимо поиска похожих слов, векторы слов можно применять к задаче на аналогии слов. Например, «man» : «woman» :: «son» : «daughter» — аналогия слов: «man» относится к «woman» так же, как «son» к «daughter». Точнее, задачу на аналогию слов можно определить так: для словесной аналогии $a : b :: c : d$ заданы первые три слова $a$, $b$ и $c$, нужно найти $d$. Обозначим вектор слова $w$ через $\text{vec}(w)$. Чтобы решить аналогию, мы найдём слово, вектор которого наиболее похож на результат $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```python
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Исключаем неизвестное слово
```

Проверим на загруженных векторах слов аналогию «мужской — женский род».

```python
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Ниже — аналогия «столица — страна»: «beijing» : «china» :: «tokyo» : «japan». Она демонстрирует семантику предобученных векторов слов.

```python
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Кроме того, для аналогий «прилагательное — превосходная степень», например «bad» : «worst» :: «big» : «biggest», предобученные векторы слов могут улавливать синтаксическую информацию.

```python
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Чтобы показать, что предобученные векторы слов улавливают понятие прошедшего времени, можно проверить синтаксис на аналогии «настоящее время — прошедшее время»: «do» : «did» :: «go» : «went».

```python
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Резюме

* На практике векторы слов, предобученные на больших корпусах, можно применять в прикладных задачах обработки естественного языка.
* Предобученные векторы слов можно применять к задачам на сходство слов и аналогии.

## Упражнения

1. Проверьте результаты fastText с помощью `TokenEmbedding('wiki.en')`.
1. Если словарь очень велик, как быстрее находить похожие слова или решать задачу на аналогию слов?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5745)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5746)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11819)
