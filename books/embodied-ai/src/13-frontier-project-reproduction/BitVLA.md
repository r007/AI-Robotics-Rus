# BitVLA: воспроизведение и дообучение на LIBERO

## 1. Установка окружения

```bash
conda create -n bitvla python=3.10 -y
conda activate bitvla
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
# Или используйте готовый Docker-образ
# docker run --name nvidia_24_07  --privileged --net=host --ipc=host --gpus=all -v /mnt:/mnt -v /tmp:/tmp -d nvcr.io/nvidia/pytorch:24.07-py3 sleep infinity

cd BitVLA
pip install -e openvla-oft/
pip install -e transformers

cd openvla-oft/
# Установка LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO/
pip install -r experiments/robot/libero/libero_requirements.txt

# Установка BitVLA
pip install -e bitvla/
```

## 2. Загрузка модели и конвертация чекпоинта

```bash
# Загрузка модели
git clone https://huggingface.co/hongyuw/bitvla-bitsiglipL-224px-bf16
python convert_ckpt.py /data/models/bitvla-bitsiglipL-224px-bf16
```

В исходном материале упоминается, что для этой модели скачивались два чекпоинта. Один из них не нужен: прямое использование оказалось невозможно.

## 3. Ошибка NumPy при запуске

При запуске оценки может появиться такая ошибка (приведён фрагмент журнала):

```text
RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
...
  File "/.../experiments/robot/libero/libero_utils.py", line 8, in <module>
    import tensorflow as tf
...
ImportError / CheckpointReader: initialization of _pywrap_checkpoint_reader raised unreported exception
```

Решение: установить совместимую версию NumPy.

```bash
pip install "numpy>=1.23.5,<2.0"
```

## 4. Дообучение на LIBERO-Spatial

```bash
bash ft_bitvla_libero_spatial.sh
```

Параметры обучения в исходном материале:

- `batch_size` = 1; обучение занимает около 36 ГБ видеопамяти;
- обучение заняло примерно 1 час 27 минут (10 001 шаг, около 1.94 it/s);
- скрипт сохраняет чекпоинт на 10 000 шаге.

Ключевые строки журнала:

```text
Using LIBERO constants:
  NUM_ACTIONS_CHUNK = 8
  ACTION_DIM = 7
  PROPRIO_DIM = 8
  ACTION_PROPRIO_NORMALIZATION_TYPE = bounds_q99
Total parameters: 2819549344
Trainable parameters: 2819549344
# trainable params in proprio_projector: 6579200
# trainable params in action_head: 59059207
# total trainable params: 2885187751
Max step 10001 reached! Stopping training...
```

*Журнал сокращён: из исходного текста удалены повторяющиеся предупреждения TensorFlow и подробные выводы CUDA.*

Итоги запуска по журналу:

- модель: `/data/models/bitvla-bitsiglipL-224px-bf16`, датасет `libero_spatial_no_noops`, для обучения включена аугментация изображений (`image_aug`);
- batch size 1, learning rate 0.0001;
- выход на 10 000 шаге, чекпоинт сохранён в каталоге:

```text
/data/ckpt/libero_spatial_no_noops/bitvla-bitsiglipL-224px-bf16+libero_spatial_no_noops+b1+lr-0.0001--image_aug--single_gpu_training--10000_chkpt/
```

## 5. Оценка

Перед оценкой нужно настроить ссылку (link) на чекпоинт. Порядок показан на скриншоте в исходном материале, который здесь не перенесён.

```bash
python experiments/robot/libero/run_libero_eval_bitnet.py \
    --pretrained_checkpoint  /data/ckpt/libero_spatial_no_noops/bitvla-bitsiglipL-224px-bf16+libero_spatial_no_noops+b1+lr-0.0001--image_aug--single_gpu_training--10000_chkpt/ \
    --task_suite_name libero_spatial \
    --info_in_path "information you want to show in path" \
    --model_family "bitnet"
```

Во время оценки успешность выполнения задач была очень низкой. Для сравнения можно посмотреть официальные результаты OpenVLA-OFT: <https://github.com/moojink/openvla-oft/blob/main/LIBERO.md>.
