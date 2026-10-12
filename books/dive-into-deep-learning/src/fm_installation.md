# Установка

Чтобы быстро начать работу и получить практический опыт, нам нужно настроить среду для запуска Python, Jupyter Notebook, необходимых библиотек и кода этой книги.

## Установка Miniconda

Проще всего установить [Miniconda](https://conda.io/en/latest/miniconda.html), которая работает на Python 3.x. Если conda уже установлена, следующие шаги можно пропустить. Зайдите на сайт Miniconda и выберите подходящую версию в соответствии с вашей версией Python 3.x.

Если вы используете macOS и версия Python — 3.9 (на ней мы тестировали код), скачайте bash-скрипт, в имени которого есть строка «MacOSX», и выполните следующее:

```bash
# Пример для процессора Intel; имя файла может отличаться
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```

Если вы используете Linux и версия Python — 3.9 (на ней мы тестировали код), скачайте bash-скрипт, в имени которого есть строка «Linux», и выполните следующее:

```bash
# Имя файла может отличаться
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```

Затем инициализируйте оболочку терминала, чтобы можно было запускать `conda` напрямую.

```bash
~/miniconda3/bin/conda init
```

Теперь закройте и заново откройте текущую оболочку. Создайте новое окружение командой:

```bash
conda create --name d2l python=3.9 -y
```

Теперь активируйте окружение `d2l`:

```bash
conda activate d2l
```

## Установка фреймворка глубокого обучения и пакета `d2l`

Перед установкой фреймворка глубокого обучения проверьте, есть ли на компьютере доступный GPU. Например, посмотрите, установлен ли на компьютере GPU NVIDIA и [CUDA](https://developer.nvidia.com/cuda-downloads). Если на машине нет GPU, беспокоиться не стоит: для первых глав вполне достаточно CPU. Но если вы хотите комфортно пройти все главы, заранее обзаведитесь GPU и установите GPU-версию фреймворка глубокого обучения.

**MXNet:** чтобы установить GPU-версию MXNet, сначала нужно узнать установленную версию CUDA (это можно проверить командой `nvcc --version` или `cat /usr/local/cuda/version.txt`). Предположим, что установлена CUDA 10.1; выполните следующие команды:

```bash
# Для пользователей Linux и macOS
pip install mxnet-cu101==1.7.0

# Для пользователей Windows
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python
```

Число в конце `mxnet-cu101` можно менять в соответствии с версией CUDA, например: для CUDA 10.0 — `cu100`, для CUDA 9.0 — `cu90`.

Если на машине нет GPU NVIDIA или CUDA, CPU-версию MXNet можно установить так:

```bash
pip install mxnet==1.7.0.post1
```

**PyTorch:** CPU- или GPU-версию PyTorch можно установить так:

```bash
pip install torch==1.12.0
pip install torchvision==0.13.0
```

**TensorFlow:** CPU- или GPU-версию TensorFlow можно установить так:

```bash
pip install tensorflow==2.8.0
pip install tensorflow-probability==0.16.0
```

**PaddlePaddle:** чтобы установить GPU-версию PaddlePaddle, сначала нужно узнать установленную версию CUDA (это можно проверить командой `nvcc --version` или `cat /usr/local/cuda/version.txt`). Предположим, что установлена CUDA 11.2; выполните следующую команду:

```bash
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

Если на машине нет GPU NVIDIA или CUDA, CPU-версию PaddlePaddle можно установить так:

```bash
# Параметр -i задаёт зеркало PyPI университета Цинхуа; его можно опустить
python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Следующий шаг — установить пакет `d2l`, чтобы было удобно вызывать функции и классы, которые часто используются в книге:

```bash
pip install d2l==0.17.6
```

## Скачивание блокнотов D2L

Далее нужно скачать код книги. Можно скачать и распаковать его, выбрав пункт с блокнотами Jupyter вверху HTML-страницы книги на сайте оригинала [zh.d2l.ai](https://zh.d2l.ai), либо загрузить следующим образом:

**MXNet:**

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd mxnet
```

**PyTorch:**

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd pytorch
```

**TensorFlow:**

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd tensorflow
```

**PaddlePaddle:**

```bash
mkdir d2l-zh && cd d2l-zh
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
cd paddle
```

Примечание: если `unzip` не установлен, его можно установить командой `sudo apt install unzip`.

После установки откройте Jupyter Notebook командой (в командной строке Windows перед запуском этой команды сначала перейдите в каталог с только что распакованным кодом книги):

```bash
jupyter notebook
```

Теперь откройте в браузере <http://localhost:8888> (обычно он открывается автоматически). После этого можно запускать код каждого раздела книги. Перед запуском кода книги, обновлением фреймворка глубокого обучения или пакета `d2l` всегда выполняйте `conda activate d2l`, чтобы активировать среду выполнения. Чтобы выйти из окружения, выполните `conda deactivate`.

**MXNet:**
[Обсуждение](https://discuss.d2l.ai/t/2082)

**PyTorch:**
[Обсуждение](https://discuss.d2l.ai/t/2083)

**TensorFlow:**
[Обсуждение](https://discuss.d2l.ai/t/2084)

**PaddlePaddle:**
[Обсуждение](https://discuss.d2l.ai/t/11679)
