# Руководство по воспроизведению ETPNav

## 🌟 О проекте (About)

> **Официальный репозиторий**: [MarSaKi/ETPNav](https://github.com/MarSaKi/ETPNav)
> **Статья**: [ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments](https://arxiv.org/abs/2304.03047v2) (статья принята в 2024 году в ведущий журнал **IEEE TPAMI**)

**Замечание: ETPNav — отличная работа Dong An и соавторов; сейчас это сильная базовая модель (Baseline) в области визуально-языковой навигации в непрерывной среде (VLN-CE). Этот проект и документ в основном представляют собой подробные заметки о подводных камнях и инструкцию по действиям, которые я собрал, изучая и воспроизводя официальный открытый проект ETPNav. Надеюсь, они помогут исследователям, которые хотят воспроизвести эту работу, — это ясное и замкнутое пошаговое руководство.**

<p align="center">
  <img width="1044" height="287" alt="Демо ETPNav 1" src="https://github.com/user-attachments/assets/fad76d29-53fb-4f8f-aa66-8eb704f748de" />
  <br>
  <img width="1023" height="294" alt="Демо ETPNav 2" src="https://github.com/user-attachments/assets/c06c0b09-d99e-4cd9-90e5-05833dca93b4" />
  <br>
  <img width="523" height="287" alt="Демо ETPNav 3" src="https://github.com/user-attachments/assets/1c4b2b30-a174-4830-aeb2-478c4718ee65" />
</p>

Что касается самого алгоритма **ETPNav** (Evolving Topological Planning), он решает ограничения существующих методов в дальнем планировании и избегании препятствий и прорывается в двух основных возможностях:

1. **Онлайн-топологическое картографирование и дальнее планирование**: модели не нужно заранее исследовать среду — она сама строит топологическую карту онлайн, самоорганизуя путевые точки (waypoints), которые предсказываются по ходу движения. Это позволяет разделить навигационную задачу на высокоуровневое планирование и низкоуровневое управление. На основе этой топологической карты кросс-модальный планировщик (на основе Transformer) лучше понимает инструкцию и формирует дальний навигационный маршрут.
2. **Управление избеганием препятствий в непрерывной среде**: с учётом сложности непрерывной среды ETPNav предлагает устойчивый контроллер избегания препятствий (Tryout), основанный на эвристике «проб и ошибок» (trial-and-error). Он эффективно предотвращает тупики, в которые агент попадает из-за столкновений с препятствиями в непрерывной среде.

## 1. Настройка окружения

Для этого воспроизведения используется окружение Python 3.8. Настройте виртуальное окружение и зависимости по шагам ниже:

### 1.1 Создание виртуального окружения и установка PyTorch

```bash
# Создать и активировать conda-окружение
conda create -n vlnce38 python=3.8
conda activate vlnce38

# Установить PyTorch (1.9.1+cu111)
(Способ 1)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
-f https://download.pytorch.org/whl/torch_stable.html \
-i https://mirrors.cloud.tencent.com/pypi/simple

(Способ 2)
## Скачать torch 1.9.1+cu111 (Python 3.8 / Linux x86_64)
wget https://mirrors.aliyun.com/pytorch-wheels/cu111/torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl

## Скачать torchvision 0.10.1+cu111 (соответствует версии torch выше)
wget https://mirrors.aliyun.com/pytorch-wheels/cu111/torchvision-0.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install torch-1.9.1+cu111-cp38-cp38-linux_x86_64.whl torchvision-0.10.1+cu111-cp38-cp38-linux_x86_64.whl
```

### 1.2 Установка зависимостей проекта

Скачайте `requirements.txt`: [Baidu Netdisk](https://pan.baidu.com/s/1PU1pm596QtL8ZNG08602Dg) (код: `8je8`)

```bash
pip install "pip<24.1" setuptools==65.5.0 wheel==0.38.4
python -m pip install -r requirements.txt
```

### 1.3 Установка симулятора Habitat

(1) Скачайте предварительно собранный пакет Habitat-sim v0.1.7 без графического интерфейса (headless): [скачать](https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2)
(2) Скачайте habitat-lab v0.1.7: [скачать](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7)

```bash
# Установить habitat-sim
conda install habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2

# Установить habitat-lab
cd habitat-lab-0.1.7
pip install -e .
```

### 1.4 Исправление зависимостей на облачном сервере (необязательно)

При запуске симулятора Habitat в облачном контейнере или в урезанной версии Linux часто возникают проблемы: не хватает библиотек графического рендеринга (ошибки `OpenGL/EGL`) и конфликтуют версии низкоуровневых библиотек C++ (ошибки `CXXABI`). Чтобы разом их решить, выполните шаги по порядку:

**1. Установка системных зависимостей рендеринга (исправляет ошибку «отсутствует libEGL»)**

Для инициализации трёхмерного движка Magnum, на котором работает Habitat, нужны такие библиотеки:

```bash
apt-get update
apt-get install -y libopengl0 libgl1-mesa-glx libglib2.0-0 libegl1
```

**2. Обновление системной стандартной библиотеки C++ (исправляет ошибку CXXABI)**

```bash
# Установить инструмент для управления PPA-репозиториями
apt-get install -y software-properties-common

# Добавить тестовый репозиторий ubuntu-toolchain-r
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update

# Обновить только стандартную библиотеку C++ системы
apt-get install --only-upgrade libstdc++6 -y
```

### 1.5 Загрузка исходного кода ETPnav-main

[Оффлайн-загрузка](https://github.com/MarSaKi/ETPNav)
```bash
# Клонировать репозиторий
git clone https://gh-proxy.org/https://github.com/MarSaKi/ETPNav.git
```

## 2. Загрузка наборов данных

### 2.1 Сцены (Scenes): Matterport3D (MP3D)

Нужно скачать реконструкцию сцен Matterport3D: всего 90 сцен, объём набора около 22 ГБ. Итоговый путь хранения: `data/scene_datasets/mp3d/{scene}/{scene}.glb`

* **Способ 1: официальный скрипт** (нужен Python 2.7)
  Перейдите на [официальную страницу проекта Matterport3D](https://niessner.github.io/Matterport/) и следуйте инструкциям, чтобы получить официальный скрипт загрузки (`download_mp.py`).
  ```bash
  python download_mp.py --task habitat -o data/scene_datasets/mp3d/
  ```

* **Способ 2: быстрая загрузка с облачного диска**
  * Ссылка: [Baidu Netdisk](https://pan.baidu.com/s/1XRXDsRhg4j09nHxXBe9boA)
  * Код: `4kz6`

### 2.2 Задачные данные (Episodes): R2R и RxR

Поместите загруженные данные эпизодов в каталог `data/datasets`.

| Набор данных | Ссылка для загрузки | Путь хранения |
| --- | --- | --- |
| **R2R_VLNCE_v1-2_preprocessed** | [Google Drive](https://drive.google.com/file/d/1j9sQ0w4wFYSafh42U8VCuKTwMrnrsV6z/view) | `data/datasets` |
| **R2R_VLNCE_v1-2_preprocessed_BERTidx** | [Baidu Netdisk](https://pan.baidu.com/s/1Sz7Q7iXcLV7ToQ0FeZmHIg) (код: `88yy`) | `data/datasets` |
| **RxR** | [Baidu Netdisk](https://pan.baidu.com/s/1WZwKwdBt0ofdmuTKmWjHSQ) (код: `g317`) | `data/datasets` |

### 2.3 Связанные графы (Connectivity Graphs)

Файл связанного графа для визуализации:
* **Ссылка для загрузки**: [connectivity_graphs.pkl](https://github.com/jacobkrantz/VLN-CE/blob/master/data/connectivity_graphs.pkl)
* **Путь хранения**: `data/connectivity_graphs.pkl`

---

## 3. Веса моделей и предобучающие данные

Скачайте соответствующие веса энкодеров, предсказателя и предвычисленные файлы признаков и поместите их в указанные каталоги.

### 3.1 Веса энкодеров и компонентов

| Компонент модели | Ссылка для загрузки | Целевой путь хранения |
| --- | --- | --- |
| **Предсказатель путевых точек (R2R-CE)** | [[ссылка на исходный проект](https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view)] | `data/wp_pred/check_cwp_bestdist_hfov90` |
| **Предсказатель путевых точек (RxR-CE)** | [[ссылка на исходный проект](https://drive.google.com/file/d/1LxhXkise-H96yMMrTPIT6b2AGjSjqqg0/view)] | `data/wp_pred/check_cwp_bestdist_hfov63` |
| **Веса BERT** | [Huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main) | `bert_config/bert-base-uncased` |
| **Энкодер RGB (ViT-B32)** | [Huggingface](https://huggingface.co/jinaai/clip-models/blob/main/ViT-B-32.pt) | `.cache/clip/ViT-B-32.pt` |
| **Энкодер глубины (ResNet50)** | [Gibson Pretrained](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth) | `data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth` |

### 3.2 Предобучающие данные (Pretraining Data)

* **Предобучающие данные R2R**: [ссылка для загрузки](https://www.dropbox.com/scl/fo/4iaw2ii2z2iupu0yn4tqh/AP2waOdlwdbJE5sUti2557U/R2R?dl=0&rlkey=88khaszmvhybxleyv0a9bulyn&subfolder_nav_tracking=1) $\rightarrow$ сохранить в `pretrain_src/datasets/R2R`
* **Предвычисленные визуальные признаки**: [ссылка для загрузки](https://drive.google.com/file/d/1D3Gd9jqRfF-NjlxDAQG_qwxTIakZlrWd/view) $\rightarrow$ сохранить в `pretrain_src/datasets/img_features`
* **Веса предобучения LXMERT**: [ссылка для загрузки](https://nlp.cs.unc.edu/data/model_LXRT.pth) $\rightarrow$ сохранить в `pretrain_src/datasets/pretrained/LXMERT`

### 3.3 Итоговые веса предобучения (Pretrained Weights)

Если вы хотите пропустить этап предобучения, можно сразу скачать готовые веса:
* **Ссылка для загрузки**: [Baidu Netdisk](https://pan.baidu.com/s/1oTmRkuj6syTmI6kE78k0JQ) (код: `vfsh`)
* **Путь хранения**: `pretrained/ETP/model_step_82500.pt`

**Итоговая структура папок:**

```text
ETPNav/
├── assets/
├── bert_config/
│   ├── bert-base-uncased/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── xlm-roberta-base/
├── data/
│   ├── datasets/
│   │   ├── R2R_VLNCE_v1-2_preprocessed/
│   │   ├── R2R_VLNCE_v1-2_preprocessed_BERTidx/
│   │   └── RxR_VLNCE_v0_enc_xlmr/
│   ├── ddppo-models/
│   │   └── gibson-2plus-resnet50.pth
│   ├── scene_datasets/
│   │   └── mp3d/
│   ├── wp_pred/
│   │   ├── check_cwp_bestdist_hfov79
│   │   └── check_cwp_bestdist_hfov90
│   └── connectivity_graphs.pkl
├── habitat_extensions/
├── precompute_img_features/
├── pretrain_src/
│   ├── datasets/
│   │   ├── pretrained/
│   │   │   └── LXMERT/
│   │   │       └── model_LXRT.pth
│   │   └── R2R/
│   ├── img_features/
│   │   ├── CLIP-ViT-B-32-views-habitat.hdf5
│   │   └── ddppo_resnet50_depth_features.hdf5
│   ├── pretrain_src/
│   └── run_pt/
├── pretrained/
│   ├── ETP/
│   │   └── model_step_82500.pt
├── run_r2r/
├── run_rxr/
├── vlnce_baselines/
├── .gitignore
├── environment.yaml
├── LICENSE
├── README.md
└── run.py
```

---

## 4. Запуск кода

### 4.1 Предобучение (Pretraining)

(Если хотите пропустить этот шаг, можно использовать ранее скачанные готовые веса и сразу перейти к дообучению или оценке.)

1. **Исправить пути к файлам**: если нет доступа к внешней сети, замените адрес весов BERT на локальный.
   <p align="center">
     <img width="1566" height="444" alt="Изменение пути импорта" src="https://github.com/user-attachments/assets/47b99ba4-f96a-441b-82d4-0b117a2ba452" />
   </p>

2. **Изменить число GPU**: настройте число GPU для распределённого обучения в соответствии с реальным оборудованием.
   <p align="center">
     <img width="1148" height="349" alt="Изменение числа GPU" src="https://github.com/user-attachments/assets/0336bc2b-9ae6-4294-9f73-105646ff49b9" />
   </p>

3. **Запустить предобучение**: на этом шаге выполняются две задачи предобучения — MLM (Masked Language Modeling) и SAP.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash pretrain_src/run_pt/run_r2r.bash 233
   ```
   Журнал обучения сохраняется в `/ETPNav-main/pretrained/r2r_ce/mlm.sap_habitat_depth/logs/log.txt`; по реальным метрикам оценки выберите лучшие веса предобучения:
   <p align="center">
     <img width="395" height="260" alt="Журнал предобучения" src="https://github.com/user-attachments/assets/321a4a31-4fb1-420a-b40c-a1540ab5d82a" />
   </p>

### 4.2 Дообучение (Finetuning)

1. **Настроить путь к весам предобучения**: укажите в скрипте путь загрузки `pretrained/ETP/model_step_82500.pt`.
   <p align="center">
     <img width="1019" height="499" alt="Настройка пути к предобученным весам 1" src="https://github.com/user-attachments/assets/5b51f709-6b75-439e-993f-f4d714e8a879" />
     <br>
     <img width="1123" height="497" alt="Настройка пути к предобученным весам 2" src="https://github.com/user-attachments/assets/0c85ccfa-e76c-4518-8919-bd1246734127" />
   </p>

2. **Изменить число GPU**: настройте число GPU для распределённого обучения в соответствии с реальным оборудованием.
   <p align="center">
     <img width="1060" height="434" alt="Изменение числа GPU 1" src="https://github.com/user-attachments/assets/1bf9004e-4991-411c-8b7c-92150f8ec5c4" />
     <br>
     <img width="1337" height="333" alt="Изменение числа GPU 2" src="https://github.com/user-attachments/assets/e45b0d7e-7b7b-431b-a760-37830c0dee95" />
   </p>

3. **Запустить дообучение**:
   **Примечание**: дообучение на одной видеокарте RTX 4090 занимает примерно 1,5 суток.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash train 2333  # обучение
   ```
4. **Отключить предупреждения TensorFlow (необязательно)**:
  export TF_ENABLE_ONEDNN_OPTS=0
  export TF_CPP_MIN_LOG_LEVEL=2


### 4.3 Тестирование и оценка (Eval & Testing)

#### Оценка

Запустите оценку командой:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash eval  2333  # оценка
```

**Метрики, полученные в режиме eval:**

<p align="center">
  <img width="1175" height="266" alt="Метрики оценки" src="https://github.com/user-attachments/assets/fd25cf3f-efed-4265-b46c-0eeaf5f2c810" />
</p>

**Дополнительная визуализация (необязательно)**

Здесь можно сохранить тестовые видео на диск или использовать TensorBoard для визуализации.

<p align="center">
  <img width="1309" height="438" alt="Настройка визуализации" src="https://github.com/user-attachments/assets/d083cf80-8e06-4c31-882f-6b5fe259eff7" />
</p>

Если выбран режим `disk` и вы заново запустите код оценки, навигационные видео будут сохранены в:
`ETPNav-main/data/logs/video/release_r2r/*`

**Демонстрация видео для отдельных сцен:**

https://github.com/user-attachments/assets/e5797cce-265b-4e95-9ba4-f732c1a2813c

<p align="center">
  <img width="686" height="1440" alt="oLBMNvg9in8-478-spl1 00" src="https://github.com/user-attachments/assets/bf874fef-0de3-4405-b644-23cbf065a976" />
</p>

#### Инференс

Запустите инференс командой:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash inference  2333  # инференс
```

---

## 5. Ссылки и благодарности (Acknowledgements)

Это руководство и код в основном опираются на следующие замечательные открытые проекты. Выражаем искреннюю благодарность их авторам за открытость:

* **Официальный репозиторий ETPNav**: [MarSaKi/ETPNav](https://github.com/MarSaKi/ETPNav) — основная цель воспроизведения в этом проекте.
* **Официальный фреймворк VLN-CE**: [jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE) — предоставляет инфраструктуру навигации в непрерывной среде и стандарты оценки.
* **Discrete-Continuous-VLN**: [YicongHong/Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN) — важный источник для модулей предобучения и предсказания путевых точек.
