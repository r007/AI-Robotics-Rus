# Библиотека Xget

Чтобы скачать датасеты и модели Hugging Face (HF) через Xget, есть два основных способа.

### Способ 1: прямое преобразование URL (для командной строки)

Этот способ удобен, когда нужно скачать конкретный файл модели или датасета. Правило простое: перед исходным URL Hugging Face добавляется домен Xget и префикс `/hf`.

**Правило преобразования:**

* **Исходный URL**: `https://huggingface.co/...`
* **Ускоренный URL**: `https://xget.xi-xu.me/hf/...`

**Примеры:**

1. **Загрузка файла модели**
   * **Исходный URL**:
     ```
     https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin
     ```
   * **URL через Xget**:
     ```
     https://xget.xi-xu.me/hf/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin
     ```
2. **Загрузка файла датасета**
   * **Исходный URL**:
     ```
     https://huggingface.co/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet
     ```
   * **URL через Xget**:
     ```
     https://xget.xi-xu.me/hf/datasets/rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet
     ```

Преобразованный URL можно открыть в браузере или скачать утилитами командной строки: `wget`, `curl`, `aria2`. Для максимальной скорости в документации рекомендуется `aria2` со многопоточной загрузкой:

**Bash**

```bash
# Загрузка большого файла модели через aria2 в 16 потоков
aria2c -x 16 -s 16 https://xget.xi-xu.me/hf/microsoft/DialoGPT-large/resolve/main/pytorch_model.bin
```

### Способ 2: интеграция в Python-код (рекомендуется)

Если проект использует библиотеки `transformers` или `datasets`, удобнее всего задать переменную окружения. Тогда эти библиотеки будут автоматически загружать ресурсы через Xget.

**Основной шаг:**

Переменная окружения `HF_ENDPOINT` перенаправляет все запросы к Hugging Face Hub на зеркало Xget.

**Пример кода на Python:**

В примере переменная окружения задаётся до загрузки модели. Библиотека `transformers` затем сама использует Xget для скачивания.

**Python**

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ключевой шаг: задаём переменную окружения, чтобы transformers автоматически использовал зеркало Xget
os.environ['HF_ENDPOINT'] = 'https://xget.xi-xu.me/hf'

# Задаём имя модели как обычно
model_name = 'microsoft/DialoGPT-medium'

print(f"Загрузка модели из зеркала Xget: {model_name}")

# Загружаем модель и токенизатор методом from_pretrained
# Благодаря переменной окружения код не требует других изменений
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Модель и токенизатор успешно загружены!")

# Дальше модель используется как обычно
# chat_history_ids = model.generate(...)
```

**Итог:**

* **Разовая загрузка одного файла**: преобразуйте URL вручную и скачайте его утилитой командной строки.
* **Постоянное использование в Python-проекте**: задайте `HF_ENDPOINT` в начале кода. Библиотеки Hugging Face начнут использовать ускорение, и остальной код менять не нужно.
