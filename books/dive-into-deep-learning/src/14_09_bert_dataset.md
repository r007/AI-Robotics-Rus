# 14.9 Набор данных для предобучения BERT

Чтобы предобучить модель BERT, реализованную в разделе 14.8, нужно сгенерировать набор данных в удобном формате, подходящем для двух задач предобучения: маскированной языковой модели и предсказания следующего предложения. С одной стороны, исходная модель BERT была предобучена на объединении двух огромных корпусов — BookCorpus и английской «Википедии» (см. подраздел «Задачи предобучения» раздела 14.8), — но большинству читателей этой книги это не подходит. С другой стороны, готовая предобученная модель BERT может не подойти для приложений в специфических областях, например в медицине. Поэтому предобучение BERT на собственном наборе данных становится всё популярнее. Чтобы упростить демонстрацию предобучения BERT, мы используем небольшой корпус WikiText-2 (Merity et al., 2016).

По сравнению с набором данных PTB из раздела 14.3, который использовался для предобучения word2vec, WikiText-2 (1) сохраняет исходную пунктуацию, что подходит для предсказания следующего предложения; (2) сохраняет исходный регистр и числа; (3) более чем вдвое больше.

**MXNet:**

```python
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

**PyTorch:**

```python
from d2l import torch as d2l
import os
import random
import torch
```

**PaddlePaddle:**

```python
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import os
import random
import paddle
```

В наборе данных WikiText-2 каждая строка представляет абзац, причём между любым знаком препинания и предшествующим ему токеном вставлен пробел. Сохраняются абзацы, содержащие не менее двух предложений. Для простоты мы разбиваем предложения, используя в качестве разделителя только точку. Обсуждение более сложных методов разбиения на предложения мы оставляем для упражнений в конце раздела.

```python
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Переводим заглавные буквы в строчные
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Вспомогательные функции для задач предобучения

Ниже мы сначала реализуем вспомогательные функции для двух задач предобучения BERT. Эти функции будут вызываться позже, когда исходный текстовый корпус будет преобразован в набор данных нужного формата для предобучения BERT.

### Генерация данных для задачи предсказания следующего предложения

Согласно описанию в подразделе «Предсказание следующего предложения (Next Sentence Prediction)» раздела 14.8, функция `_get_next_sentence` генерирует обучающие примеры для задачи бинарной классификации.

```python
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs — трижды вложенный список
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

Следующая функция генерирует обучающие примеры для предсказания следующего предложения из входного `paragraph`, вызывая `_get_next_sentence`. Здесь `paragraph` — список предложений, каждое из которых — список токенов. Аргумент `max_len` задаёт максимальную длину входной последовательности BERT при предобучении.

```python
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Учитываем 1 токен '<cls>' и 2 токена '<sep>'
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### Генерация данных для задачи маскированной языковой модели

Чтобы сгенерировать обучающие примеры маскированной языковой модели из входной последовательности BERT, мы определяем следующую функцию `_replace_mlm_tokens`. На входе `tokens` — список токенов, представляющих входную последовательность BERT, `candidate_pred_positions` — список индексов токенов входной последовательности BERT без специальных токенов (специальные токены в задаче маскированной языковой модели не предсказываются), а `num_mlm_preds` указывает число предсказаний (выбирается 15% случайных токенов для предсказания). Согласно определению задачи маскированной языковой модели в подразделе «Маскированная языковая модель (Masked Language Modeling)» раздела 14.8, в каждой позиции предсказания вход может быть заменён специальным токеном `<mask>` или случайным токеном либо оставлен без изменений. Наконец, функция возвращает входные токены после возможной замены, индексы токенов, в которых происходит предсказание, и метки для этих предсказаний.

```python
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # Создаём копию токенов для входа маскированной языковой модели, куда
    # могут попасть замены '<mask>' или случайные токены
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Перемешиваем, чтобы в задаче маскированной языковой модели получить
    # 15% случайных токенов для предсказания
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # В 80% случаев заменяем слово токеном '<mask>'
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # В 10% случаев оставляем слово без изменений
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # В 10% случаев заменяем слово случайным словом
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

Вызывая описанную выше функцию `_replace_mlm_tokens`, следующая функция принимает на вход входную последовательность BERT (`tokens`) и возвращает индексы входных токенов (после возможной замены токенов, описанной в подразделе «Маскированная языковая модель (Masked Language Modeling)» раздела 14.8), индексы токенов, в которых происходит предсказание, и индексы меток для этих предсказаний.

```python
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens — список строк
    for i, token in enumerate(tokens):
        # Специальные токены в задаче маскированной языковой модели не предсказываются
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # В задаче маскированной языковой модели предсказывается 15% случайных токенов
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Преобразование текста в набор данных для предобучения

Теперь мы почти готовы написать класс `Dataset` для предобучения BERT. Но прежде нужно определить вспомогательную функцию `_pad_bert_inputs`, которая дополняет входы специальными токенами `<pad>`. Её аргумент `examples` содержит выходы вспомогательных функций `_get_nsp_data_from_paragraph` и `_get_mlm_data_from_tokens` для двух задач предобучения.

**MXNet:**

```python
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # valid_lens не включает подсчёт '<pad>'
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Предсказания для дополняющих токенов отфильтровываются в потерях
        # умножением на вес 0
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

**PyTorch:**

```python
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens не включает подсчёт '<pad>'
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Предсказания для дополняющих токенов отфильтровываются в потерях
        # умножением на вес 0
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

**PaddlePaddle:**

```python
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(paddle.to_tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=paddle.int64))
        all_segments.append(paddle.to_tensor(segments + [0] * (
            max_len - len(segments)), dtype=paddle.int64))
        # valid_lens не включает подсчёт '<pad>'
        valid_lens.append(paddle.to_tensor(len(token_ids), dtype=paddle.float32))
        all_pred_positions.append(paddle.to_tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=paddle.int64))
        # Предсказания для дополняющих токенов отфильтровываются в потерях
        # умножением на вес 0
        all_mlm_weights.append(
            paddle.to_tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=paddle.float32))
        all_mlm_labels.append(paddle.to_tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=paddle.int64))
        nsp_labels.append(paddle.to_tensor(is_next, dtype=paddle.int64))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

Объединяя вспомогательные функции для генерации обучающих примеров двух задач предобучения и вспомогательную функцию для дополнения входов, определим следующий класс `_WikiTextDataset` как набор данных WikiText-2 для предобучения BERT. Реализуя функцию `__getitem__`, мы можем произвольно обращаться к примерам предобучения (маскированной языковой модели и предсказания следующего предложения), сгенерированным из пар предложений корпуса WikiText-2.

Исходная модель BERT использует вложения WordPiece с размером словаря 30000 (Wu et al., 2016). Токенизация WordPiece — это небольшая модификация исходного алгоритма кодирования пар байтов из подраздела «Кодирование пар байтов (Byte Pair Encoding)» раздела 14.6. Для простоты мы используем для токенизации функцию `d2l.tokenize`. Редкие токены, встречающиеся менее 5 раз, отфильтровываются.

**MXNet:**

```python
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Вход paragraphs[i] — список строк-предложений, представляющих абзац;
        # выход paragraphs[i] — список предложений, представляющих абзац,
        # где каждое предложение — список токенов
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Получаем данные для задачи предсказания следующего предложения
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Получаем данные для задачи маскированной языковой модели
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Дополняем входы
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

**PyTorch:**

```python
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Вход paragraphs[i] — список строк-предложений, представляющих абзац;
        # выход paragraphs[i] — список предложений, представляющих абзац,
        # где каждое предложение — список токенов
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Получаем данные для задачи предсказания следующего предложения
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Получаем данные для задачи маскированной языковой модели
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Дополняем входы
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

**PaddlePaddle:**

```python
#@save
class _WikiTextDataset(paddle.io.Dataset):
    def __init__(self, paragraphs, max_len):
        # Вход paragraphs[i] — список строк-предложений, представляющих абзац;
        # выход paragraphs[i] — список предложений, представляющих абзац,
        # где каждое предложение — список токенов
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Получаем данные для задачи предсказания следующего предложения
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Получаем данные для задачи маскированной языковой модели
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Дополняем входы
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

С помощью функции `_read_wiki` и класса `_WikiTextDataset` мы определяем следующую функцию `load_data_wiki`, которая скачивает набор данных WikiText-2 и генерирует из него примеры предобучения.

**MXNet:**

```python
#@save
def load_data_wiki(batch_size, max_len):
    """Загружает набор данных WikiText-2"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

**PyTorch:**

```python
#@save
def load_data_wiki(batch_size, max_len):
    """Загружает набор данных WikiText-2"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

**PaddlePaddle:**

```python
#@save
def load_data_wiki(batch_size, max_len):
    """Загружает набор данных WikiText-2"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = paddle.io.DataLoader(dataset=train_set, batch_size=batch_size, return_list=True,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

Зададим размер мини-батча 512 и максимальную длину входной последовательности BERT 64 и выведем формы мини-батча примеров предобучения BERT. Обратите внимание: в каждой входной последовательности BERT для задачи маскированной языковой модели предсказывается $10$ ($64 \times 0.15$) позиций.

```python
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

Наконец, посмотрим на размер словаря. Даже после отфильтровывания редких токенов он более чем вдвое больше, чем у набора данных PTB.

```python
len(vocab)
```

## Резюме

* По сравнению с набором данных PTB, набор данных WikiText-2 сохраняет исходную пунктуацию, регистр и числа и более чем вдвое больше PTB.
* Мы можем произвольно обращаться к примерам предобучения (маскированной языковой модели и предсказания следующего предложения), сгенерированным из пар предложений корпуса WikiText-2.

## Упражнения

1. Для простоты точка используется как единственный разделитель при разбиении на предложения. Попробуйте другие методы разбиения на предложения, например из spaCy и NLTK. Возьмём NLTK: сначала нужно установить пакет: `pip install nltk`. В коде сначала выполните `import nltk`. Затем скачайте токенизатор предложений Punkt: `nltk.download('punkt')`. Чтобы разбить на предложения, например, `sentences = 'This is great ! Why not ?'`, вызов `nltk.tokenize.sent_tokenize(sentences)` вернёт список из двух строк-предложений: `['This is great !', 'Why not ?']`.
1. Каким будет размер словаря, если не отфильтровывать редкие токены?

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/5737)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/5738)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11822)

## Литература

* Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*.
* Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., & others (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*.
