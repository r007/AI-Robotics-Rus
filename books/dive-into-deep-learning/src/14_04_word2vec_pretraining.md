# 14.4 Предобучение word2vec

Продолжим реализацию модели skip-gram, определённой в разделе 14.1. Затем мы предобучим word2vec с отрицательной выборкой на наборе данных PTB. Для начала получим итератор данных и словарь для этого набора, вызвав функцию `d2l.load_data_ptb`, которая описана в разделе 14.3.

**MXNet:**

```python
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

**PyTorch:**

```python
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import math
import paddle
from paddle import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Модель skip-gram

Мы реализуем модель skip-gram с помощью слоёв вложений и пакетного матричного умножения. Сначала вспомним, как работают слои вложений.

### Слой вложений

Как описано в разделе 9.7, слой вложений отображает индекс токена в его вектор признаков. Вес этого слоя — матрица, число строк которой равно размеру словаря (`input_dim`), а число столбцов — размерности вектора каждого токена (`output_dim`). После обучения модели вложений слов именно этот вес нам и нужен.

**MXNet:**

```python
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

**PyTorch:**

```python
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Параметр embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

**PaddlePaddle:**

```python
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Параметр embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Входом слоя вложений служит индекс токена (слова). Для любого индекса токена $i$ его векторное представление берётся из $i$-й строки матрицы весов слоя вложений. Поскольку размерность вектора (`output_dim`) равна 4, для мини-батча индексов токенов формы (2, 3) слой вложений вернёт векторы формы (2, 3, 4).

```python
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Определение прямого распространения

При прямом распространении входом модели skip-gram служат индексы центральных слов `center` формы (размер мини-батча, 1) и индексы контекстных и шумовых слов `contexts_and_negatives` формы (размер мини-батча, `max_len`), где `max_len` определено в подразделе «Загрузка обучающих примеров мини-батчами» раздела 14.3. Эти две переменные сначала преобразуются из индексов токенов в векторы слоем вложений, затем их пакетное матричное умножение (описанное в подразделе «Пакетное матричное умножение» раздела 10.2) даёт выход формы (размер мини-батча, 1, `max_len`). Каждый элемент выхода — скалярное произведение вектора центрального слова и вектора контекстного или шумового слова.

**MXNet:**

```python
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

**PyTorch:**

```python
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

**PaddlePaddle:**

```python
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = paddle.bmm(v, u.transpose(perm=[0, 2, 1]))
    return pred
```

Выведем форму результата этой функции `skip_gram` для некоторых тестовых входов.

**MXNet:**

```python
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

**PyTorch:**

```python
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

**PaddlePaddle:**

```python
skip_gram(paddle.ones((2, 1), dtype='int64'),
          paddle.ones((2, 4), dtype='int64'), embed, embed).shape
```

## Обучение

Прежде чем обучать модель skip-gram с отрицательной выборкой, определим для неё функцию потерь.

### Бинарные потери кросс-энтропии

Согласно определению функции потерь отрицательной выборки в подразделе «Отрицательная выборка» раздела 14.2, мы будем использовать бинарные потери кросс-энтропии.

**MXNet:**

```python
loss = gluon.loss.SigmoidBCELoss()
```

**PyTorch:**

```python
class SigmoidBCELoss(nn.Module):
    # Бинарные потери кросс-энтропии с маской
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

**PaddlePaddle:**

```python
class SigmoidBCELoss(nn.Layer):
    # Бинарные потери кросс-энтропии с маской
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            logit=inputs, label=target, weight=mask, reduction="none")
        return out.mean(axis=1)

loss = SigmoidBCELoss()
```

Вспомним описание переменной-маски и переменной-меток в подразделе «Загрузка обучающих примеров мини-батчами» раздела 14.3. Ниже вычисляются бинарные потери кросс-энтропии для заданных переменных.

**MXNet, PyTorch:**

```python
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

**PaddlePaddle:**

```python
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype='float32')
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Ниже показано, как (менее эффективным способом) вычислить тот же результат с помощью сигмоидной функции активации в бинарных потерях кросс-энтропии. Два выхода можно рассматривать как две нормированные потери, усреднённые по немаскированным предсказаниям.

```python
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Инициализация параметров модели

Мы определяем два слоя вложений: один используется для всех слов словаря как центральных, другой — как контекстных. Размерность вектора слова `embed_size` равна 100.

**MXNet:**

```python
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

**PyTorch, PaddlePaddle:**

```python
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Определение цикла обучения

Цикл обучения определён ниже. Из-за наличия дополнения вычисление функции потерь немного отличается от прежних функций обучения.

**MXNet:**

```python
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[1, num_epochs])
    # Сумма нормированных потерь, число нормированных потерь
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'потери {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} токенов/с на {str(device)}')
```

**PyTorch:**

```python
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[1, num_epochs])
    # Сумма нормированных потерь, число нормированных потерь
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'потери {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} токенов/с на {str(device)}')
```

**PaddlePaddle:**

```python
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.initializer.XavierUniform(m.weight)
    net.apply(init_weights)
    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    animator = d2l.Animator(xlabel='эпоха', ylabel='потери',
                            xlim=[1, num_epochs])
    # Сумма нормированных потерь, число нормированных потерь
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.clear_grad()
            center, context_negative, mask, label = [
                paddle.to_tensor(data, place=device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape), paddle.to_tensor(label, dtype='float32'), 
                        paddle.to_tensor(mask, dtype='float32'))
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'потери {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} токенов/с на {str(device)}')
```

Теперь можно обучить модель skip-gram с отрицательной выборкой.

```python
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Применение вложений слов

После обучения модели word2vec мы можем по косинусному сходству векторов слов обученной модели найти в словаре слова, семантически наиболее близкие к заданному.

**MXNet:**

```python
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Вычисляем косинусное сходство. Добавляем 1e-9 для численной устойчивости
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + \
          1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Исключаем входное слово
        print(f'косинусное сходство={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

**PyTorch:**

```python
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Вычисляем косинусное сходство. Добавляем 1e-9 для численной устойчивости
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Исключаем входное слово
        print(f'косинусное сходство={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

**PaddlePaddle:**

```python
def get_similar_tokens(query_token, k, embed):
    W = embed.weight
    x = W[vocab[query_token]]
    # Вычисляем косинусное сходство. Добавляем 1e-9 для численной устойчивости
    cos = paddle.mv(W, x) / paddle.sqrt(paddle.sum(W * W, axis=1) *
                                        paddle.sum(x * x) + 1e-9)
    topk = paddle.topk(cos, k=k+1)[1].numpy().astype('int32')
    for i in topk[1:]:  # Исключаем входное слово
        print(f'косинусное сходство={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Резюме

* Модель skip-gram с отрицательной выборкой можно обучить, используя слои вложений и бинарные потери кросс-энтропии.
* Применения вложений слов включают нахождение семантически близких слов для заданного слова по косинусному сходству их векторов.

## Упражнения

1. Используя обученную модель, найдите семантически близкие слова для других входных слов. Можно ли улучшить результаты, настроив гиперпараметры?
1. Когда обучающий корпус огромен, при обновлении параметров модели контекстные и шумовые слова часто выбирают для *центральных слов в текущем мини-батче*. Иными словами, у одного и того же центрального слова на разных итерациях обучения могут быть разные контекстные или шумовые слова. В чём преимущество такого подхода? Попробуйте реализовать этот метод обучения.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5739)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5740)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11817)
