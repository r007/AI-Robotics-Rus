# Облачная платформа AMD ROCm и удалённая разработка

Эта глава помогает выбрать платформу запуска, проверить постоянный каталог и настроить удалённый доступ для разработки. Основные действия в веб-интерфейсе кнопок, JupyterHub и Code Server описаны в [приложении о работе с платформой](./appendices/AUP_JUPYTER_CODE_SERVER.md).

## Выбор платформы

| Платформа | Подходящие задачи | Хранилище и сеть | Рекомендация |
| --- | --- | --- | --- |
| AUP Learning Cloud | Загрузка открытых данных, установка зависимостей, изучение Notebook, обучение и оценка | Есть доступ к GitHub и Hugging Face; ориентируйтесь на постоянный каталог, указанный платформой | Основной вход для этой темы |
| Облако разработчика AMD | Предварительная проверка шаблона ROCm, обучение на GPU и тесты производительности | Сохраняется только PVC в `/workspace` после перезапуска экземпляра; остальные каталоги считаются временными | Размещайте исходный код, модели и результаты в `/workspace` |
| Локальная рабочая станция AMD | Длительное обучение, пакетная оценка и генерация видео | Диск данных и кэш настраиваете сами | Подходит для постоянных экспериментов |

Квоты, модели GPU и срок жизни экземпляров могут меняться; ориентируйтесь на текущую информацию на странице платформы.

## Первичная проверка

Войдите в терминал и выполните:

```bash
id
pwd
df -h
python --version
which python
```

Затем проверьте GPU AMD и ROCm:

```bash
rocminfo | sed -n '1,80p'
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
assert torch.version.hip is not None
PY
```

PyTorch на ROCm использует интерфейс `torch.cuda`, поэтому строка устройства по-прежнему записывается как `cuda`. Атрибут `torch.version.hip` подтверждает, что установлена сборка ROCm.

## Постоянный каталог

Для крупных файлов рекомендуется единая переменная `WORK_ROOT`:

```bash
export WORK_ROOT=${WORK_ROOT:-$HOME/physical-ai}

# Для постоянного PVC в облаке AMD:
# export WORK_ROOT=/workspace/physical-ai

# Если образ AUP сохраняет /home/jovyan:
# export WORK_ROOT=/home/jovyan/physical-ai

export SRC_ROOT=$WORK_ROOT/src
export DATA_ROOT=$WORK_ROOT/datasets
export MODEL_ROOT=$WORK_ROOT/checkpoints
export RUN_ROOT=$WORK_ROOT/runs
export CACHE_ROOT=$WORK_ROOT/cache
export HF_HOME=$CACHE_ROOT/huggingface
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export UV_CACHE_DIR=$CACHE_ROOT/uv

mkdir -p "$SRC_ROOT" "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT" \
  "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Перед перезапуском экземпляра копируйте модели, данные и результаты в постоянный каталог. Временные виртуальные окружения можно пересоздать; не храните их внутри Git-репозитория.

## Загрузка исходного кода и моделей

```bash
cd "$SRC_ROOT"
git clone https://github.com/datawhalechina/every-embodied.git
cd every-embodied

hf auth login
hf auth whoami
```

Перед загрузкой закрытых моделей подайте заявку на доступ на странице Hugging Face. Учётные данные для входа сохраняются в пользовательском каталоге конфигурации и не записываются в Notebook, скрипты или репозиторий.

## Служба SSH

В некоторых образах SSH не установлен. Если у вас есть права администратора, выполните:

```bash
sudo apt-get update
sudo apt-get install -y openssh-server iproute2
sudo mkdir -p /run/sshd
sudo ssh-keygen -A
sudo service ssh restart
ss -lntp | grep ':22'
```

Если платформа использует проброс портов, ориентируйтесь на хост и порт, которые показаны на её странице:

```bash
ssh -p <PORT> <USER>@<HOST>
```

SSH отвечает только за удалённый терминал. Запуск, остановка и удаление экземпляров выполняются в консоли платформы.

## Запуск темы

```bash
cd "$SRC_ROOT/every-embodied/<topic-folder>"
jupyter lab --ip=0.0.0.0 --no-browser
```

Здесь `<topic-folder>` — каталог этой темы в исходном репозитории `every-embodied` (раздел тем для совместного обучения, тема AMD ROCm).

Сначала выполните [проверку устройства и окружения](./01-amd-rocm-device-and-environment-confirmation.md), а затем по [единому каталогу](./10-simulation-benchmark-download-and-unified-directory.md) загрузите данные бенчмарков и модели.

## Перенос на другую платформу

Чтобы продолжить эксперимент на другом устройстве, достаточно заново задать переменные каталогов и проверить четыре группы материалов:

1. репозиторий с исходным кодом и версия вышестоящего проекта;
2. каталог наборов данных и их метаданные;
3. каталог моделей и конфигурации обучения;
4. результаты оценки, видео и конфигурации запусков.

Скрипты и Notebook используют `$SRC_ROOT`, `$DATA_ROOT`, `$MODEL_ROOT` и `$RUN_ROOT`, поэтому в них не встречаются абсолютные пути конкретного сервера.
