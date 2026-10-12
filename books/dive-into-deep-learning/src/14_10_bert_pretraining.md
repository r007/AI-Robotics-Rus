# 14.10 Предобучение BERT

Используя модель BERT, реализованную в разделе 14.8, и примеры предобучения, сгенерированные из набора данных WikiText-2 в разделе 14.9, мы предобучим BERT на наборе данных WikiText-2 в этом разделе.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import torch
from torch import nn
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
```

Сначала мы загружаем набор данных WikiText-2 как мини-батчи примеров предобучения для маскированной языковой модели и предсказания следующего предложения. Размер мини-батча равен 512, максимальная длина входной последовательности BERT — 64. Заметим, что в исходной модели BERT максимальная длина равна 512.

**MXNet, PyTorch:**

```python
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

**PaddlePaddle:**

```python
def load_data_wiki(batch_size, max_len):
    """Загружает набор данных WikiText-2

    Определено в подразделе «Генерация данных для задачи маскированной языковой модели» раздела 14.9"""
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = d2l._read_wiki(data_dir)
    train_set = d2l._WikiTextDataset(paragraphs, max_len)
    train_iter = paddle.io.DataLoader(dataset=train_set, batch_size=batch_size, return_list=True,
                                        shuffle=True, num_workers=0)
    return train_iter, train_set.vocab

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
```

## Предобучение BERT

У исходного BERT (Devlin et al., 2018) есть две версии разного размера. Базовая модель ($\text{BERT}_{\text{BASE}}$) использует 12 слоёв (блоков кодировщика Transformer), 768 скрытых элементов (скрытый размер) и 12 голов самовнимания. Большая модель ($\text{BERT}_{\text{LARGE}}$) использует 24 слоя, 1024 скрытых элемента и 16 голов самовнимания. Стоит отметить, что у первой 110 миллионов параметров, а у второй — 340 миллионов. Для простоты демонстрации мы определяем малый BERT с 2 слоями, 128 скрытыми элементами и 2 головами самовнимания.

**MXNet:**

```python
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_layers=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

**PyTorch, PaddlePaddle:**

```python
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

Прежде чем определять цикл обучения, определим вспомогательную функцию `_get_batch_loss_bert`. По обучающим примерам эта функция вычисляет потери двух задач: маскированной языковой модели и предсказания следующего предложения. Обратите внимание: итоговые потери предобучения BERT — это сумма потерь маскированной языковой модели и потерь предсказания следующего предложения.

**MXNet:**

```python
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Прямое распространение
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Вычисляем потери маскированной языковой модели
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Вычисляем потери задачи предсказания следующего предложения
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

**PyTorch:**

```python
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Прямое распространение
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Вычисляем потери маскированной языковой модели
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Вычисляем потери задачи предсказания следующего предложения
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

**PaddlePaddle:**

```python
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Прямое распространение
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape([-1]), 
                                  pred_positions_X)
    # Вычисляем потери маскированной языковой модели
    mlm_l = loss(mlm_Y_hat.reshape([-1, vocab_size]), mlm_Y.reshape([-1])) *\
    mlm_weights_X.reshape([-1, 1])
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Вычисляем потери задачи предсказания следующего предложения
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Вызывая две описанные вспомогательные функции, следующая функция `train_bert` определяет процедуру предобучения BERT (`net`) на наборе данных WikiText-2 (`train_iter`). Обучение BERT может занять очень много времени. Вход `num_steps` следующей функции задаёт число итераций (шагов) обучения, а не число эпох, как в функции `train_ch13` (см. раздел 13.1).

**MXNet:**

```python
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='шаг', ylabel='потери',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Сумма потерь маскированной языковой модели, сумма потерь предсказания
    # следующего предложения, число пар предложений, счётчик
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'потери MLM {metric[0] / metric[3]:.3f}, '
          f'потери NSP {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} пар предложений/с на '
          f'{str(devices)}')
```

**PyTorch:**

```python
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='шаг', ylabel='потери',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Сумма потерь маскированной языковой модели, сумма потерь предсказания
    # следующего предложения, число пар предложений, счётчик
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'потери MLM {metric[0] / metric[3]:.3f}, '
          f'потери NSP {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} пар предложений/с на '
          f'{str(devices)}')
```

**PaddlePaddle:**

```python
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='шаг', ylabel='потери',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Сумма потерь маскированной языковой модели, сумма потерь предсказания
    # следующего предложения, число пар предложений, счётчик
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            trainer.clear_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'потери MLM {metric[0] / metric[3]:.3f}, '
          f'потери NSP {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} пар предложений/с на '
          f'{str(devices)}')
```

Во время предобучения можно построить графики потерь маскированной языковой модели и потерь предсказания следующего предложения.

**MXNet, PyTorch:**

```python
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

**PaddlePaddle:**

```python
train_bert(train_iter, net, loss, len(vocab), devices[:1], 50)
```

## Представление текста с помощью BERT

После предобучения BERT его можно использовать для представления одиночного текста, пар текстов или любого из их токенов. Следующая функция возвращает представления BERT (`net`) для всех токенов из `tokens_a` и `tokens_b`.

**MXNet:**

```python
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

**PyTorch:**

```python
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

**PaddlePaddle:**

```python
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = paddle.to_tensor(vocab[tokens]).unsqueeze(0)
    segments = paddle.to_tensor(segments).unsqueeze(0)
    valid_len = paddle.to_tensor(len(tokens))
    
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

Рассмотрим предложение «a crane is flying». Вспомним входное представление BERT, обсуждавшееся в подразделе «Входное представление» раздела 14.8. После вставки специальных токенов `<cls>` (для классификации) и `<sep>` (для разделения) длина входной последовательности BERT равна 6. Поскольку нулевая позиция — токен `<cls>`, `encoded_text[:, 0, :]` — представление BERT всего входного предложения. Чтобы оценить многозначный токен «crane», мы также выводим первые три элемента представления BERT этого токена.

```python
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Токены: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

Теперь рассмотрим пару предложений «a crane driver came» и «he just left». Аналогично, `encoded_pair[:, 0, :]` — результат кодирования всей пары предложений предобученным BERT. Обратите внимание: первые три элемента многозначного токена «crane» отличаются от тех, что были в другом контексте. Это подтверждает, что представления BERT контекстно-зависимы.

```python
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Токены: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

В главе 15 мы дообучим предобученную модель BERT для прикладных задач обработки естественного языка.

## Резюме

* У исходного BERT две версии: базовая модель имеет 110 миллионов параметров, а большая — 340 миллионов.
* После предобучения BERT его можно использовать для представления одиночного текста, пар текстов или любого из их токенов.
* В эксперименте один и тот же токен в разных контекстах имеет разные представления BERT. Это подтверждает контекстную зависимость представлений BERT.

## Упражнения

1. В эксперименте видно, что потери маскированной языковой модели заметно выше потерь предсказания следующего предложения. Почему?
2. Задайте максимальную длину входной последовательности BERT равной 512 (как в исходной модели BERT). Используйте конфигурацию исходной модели BERT, например $\text{BERT}_{\text{LARGE}}$. Возникают ли ошибки при запуске этого раздела? Почему?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5742)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5743)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11824)

## Литература

* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.
