# Полное руководство по развёртыванию Isaac Lab + GR00T в Alibaba Cloud

> Руководство по созданию среды разработки воплощённого ИИ для инженеров и исследователей с базовыми навыками Linux и GPU
>
> Окружение автора, на котором всё проверялось: облачный прерываемый экземпляр Alibaba Cloud с A10 GPU

⚠️ **Заявление о версиях.** Руководство написано на версиях программного обеспечения от января 2026 года и жёстко зависит от следующей комбинации:
- Isaac Sim 4.2.0 + Isaac Lab v1.4.1
- GR00T N1.6 + transformers==4.51.3
- PyTorch 2.5.1 + CUDA 12.1

**Гарантируется воспроизводимость только на этих версиях.** При использовании более новых версий, возможно, придётся самостоятельно менять зависимости и конфигурацию.

⚠️ **Заявление о безопасности.** Руководство рассчитано на быстрый эксперимент: в нём используются root-пользователь, `--network host` и `xhost +`. **Не рекомендуется для продакшена и для серверов, которые работают постоянно.**

## Содержание

1. [Обзор окружения](#1-обзор-окружения)
2. [Создание экземпляра в Alibaba Cloud](#2-создание-экземпляра-в-alibaba-cloud)
3. [Базовая настройка окружения](#3-базовая-настройка-окружения)
4. [Настройка удалённого рабочего стола VNC](#4-настройка-удалённого-рабочего-стола-vnc)
5. [Развёртывание Isaac Sim и Isaac Lab](#5-развёртывание-isaac-sim-и-isaac-lab)
6. [Настройка окружения GR00T](#6-настройка-окружения-groot)
7. [Оценка GR00T + MuJoCo](#7-оценка-groot--mujoco)
8. [Частые проблемы и журнал ошибок](#8-частые-проблемы-и-журнал-ошибок)

---

## 1. Обзор окружения

### Итоговая архитектура

```
Хост (Ubuntu 22.04 + A10 GPU)
├── Docker: isaac-sim-gui
│   ├── Isaac Sim 4.2.0
│   ├── Isaac Lab v1.4.1
│   └── Вывод VNC (порт 6080)
└── Conda: окружение groot
    ├── Код Isaac-GR00T
    ├── Модель GR00T-N1.6-3B (универсальная)
    └── Модель GR00T-N1.6-G1-PnPAppleToPlate (для G1)
```

### Почему так устроено

- **Isaac Lab внутри Docker:** зависимости Isaac Sim сложны, официальный Docker-образ — самый удобный вариант.
- **GR00T в Conda на хосте:** GR00T требует PyTorch 2.5+, а Isaac Sim использует PyTorch 2.4, и они конфликтуют.
- **Удалённый рабочий стол VNC:** WebRTC Livestream на облачном сервере работает нестабильно, VNC надёжнее.

#### Ключевые проектные решения

| Решение | Почему так | Что будет, если сделать иначе |
|------|-------------|---------------|
| GR00T в Conda, а не в Docker | PyTorch 2.5+ конфликтует с Isaac Sim | Ад зависимостей, окружение ломается |
| Пропуск flash-attn | Сборка долгая и не обязательна | OOM или зависание, перезагрузка сервера |
| Фиксация transformers==4.51.3 | В новых версиях изменился API | Ошибки загрузки модели |
| Установка GR00T с `--no-deps` | Чтобы не собирать flash-attn заново | pip зависает |
| Ручное клонирование подмодулей | curl не даёт `.git` | Скрипт setup падает с ошибкой |

### Порядок развёртывания (выполнять строго по порядку)

```
базовая настройка (3) → VNC (4) → Isaac Docker (5) → GR00T Conda (6) → оценка MuJoCo (7)
     │              │              │                │
     └── драйвер NVIDIA → Docker → Container Toolkit ─┘
```

> ⚠️ **Не пропускайте шаги и не меняйте порядок:** каждый шаг опирается на предыдущие.

### Требования к оборудованию

- GPU: NVIDIA A10 24 ГБ или выше (подойдут и RTX 3090/4090)
- Оперативная память: от 32 ГБ
- Диск: **от 150 ГБ SSD** (фактически занято около 80–100 ГБ)
  - система Ubuntu: около 5 ГБ;
  - образ Docker Isaac Sim: около 25 ГБ;
  - кэш рантайма Docker: около 10 ГБ;
  - окружение Conda (PyTorch + CUDA): около 10 ГБ;
  - модели GR00T (3B + G1): около 12 ГБ;
  - журналы и временные файлы: около 10 ГБ;
  - резерв: около 30 ГБ.

---

## 2. Создание экземпляра в Alibaba Cloud

### 2.0 Подготовка

#### Регистрация учётной записи NGC (нужна для загрузки образа Isaac Sim)

1. Перейдите на https://ngc.nvidia.com.
2. Нажмите «Sign Up» и зарегистрируйтесь по e-mail (или войдите через Google/GitHub).
3. После входа нажмите на аватар в правом верхнем углу → «Setup» → «Generate API Key».
4. Нажмите «Generate API Key» и сохраните ключ.

> ⚠️ API-ключ показывается только один раз — сохраните его. Он понадобится для `docker login nvcr.io`.

### 2.1 Стоимость

- **Фактическая стоимость:** прохождение всего руководства обошлось примерно в **30 юаней** (прерываемый экземпляр и трафик).
- **Требование к аккаунту:** чтобы купить прерываемый экземпляр, на аккаунте Alibaba Cloud нужно не менее **100 юаней**.
- Цены на прерываемые экземпляры колеблются, поэтому фактическая стоимость может отличаться.

### 2.2 Выбор конфигурации экземпляра

1. Войдите в консоль Alibaba Cloud → ECS (облачные серверы) → создать экземпляр.
2. Выберите регион: рекомендуется **Западная 1 (Чэнду)** — там самая низкая цена.
3. Тип экземпляра: найдите `ecs.gn7i` и выберите конфигурацию A10 24 ГБ.
   - **ecs.gn7i-c16g1.4xlarge** (рекомендуется): 16 vCPU + 60 ГБ памяти;
   - ecs.gn7i-c8g1.2xlarge: 8 vCPU + 32 ГБ памяти.

> 💡 **Как сэкономить:** цены в разных регионах заметно различаются.
> - Чэнду, c16g1.4xlarge: около **2,4 юаня/час** (высокая конфигурация по низкой цене);
> - Пекин/Ханчжоу, c8g1.2xlarge: около 3,1 юаня/час.
>
> Рекомендуется выбрать Чэнду: конфигурация выше, а цена ниже.

4. **Модель оплаты: прерываемый экземпляр** (дешевле оплаты по требованию на 80–90 %).
   - Максимальная цена — 50–70 % от цены оплаты по требованию.
   - Включите «защиту от освобождения экземпляра».

### 2.2 Выбор образа

- Операционная система: **Ubuntu 22.04 64-bit**.
- Или образ с предустановленными драйверами NVIDIA для GPU-облака.

### 2.3 Сеть

- Публичный IP (тарификация по трафику).
- Полоса 5–10 Мбит/с достаточно.

### 2.4 Группа безопасности

Создайте или измените группу безопасности и откройте порты:

| Порт | Назначение |
|------|------|
| 22 | SSH |

> ⚠️ **Рекомендация по безопасности:** не открывайте порты VNC (5901/6080) для интернета, для доступа лучше использовать SSH-туннель (см. ниже).

---

## 3. Базовая настройка окружения

### 3.1 Подключение по SSH

```bash
ssh root@<ваш публичный IP>
```

> ⚠️ **Важно:** при подключении через Workbench Alibaba Cloud не выбирайте «беспарольное подключение»!
>
> Беспарольное подключение использует контейнеризованный веб-терминал с ограниченными правами; через него Docker не запускается.
>
> Выберите «пароль» или «ключ» либо подключайтесь обычным SSH-клиентом с локального компьютера.

### 3.2 Установка драйвера NVIDIA (если его нет в образе)

```bash
# Проверка драйвера
nvidia-smi

# Если драйвера нет, установите его
apt update
apt install -y nvidia-driver-535
reboot
```

### 3.3 Установка Docker

```bash
# Проверка, установлен ли Docker
docker --version

# Если нет, установите Docker
apt update
apt install -y docker.io
systemctl enable docker
systemctl start docker

# Проверка установки
docker --version
```

### 3.4 Настройка зеркал Docker для Китая

Docker Hub в Китае работает нестабильно, поэтому настраиваем ускоряющие зеркала:

```bash
cat > /etc/docker/daemon.json << 'EOF'
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
EOF

systemctl restart docker
```

### 3.5 Установка NVIDIA Container Toolkit

Устанавливаем из deb-пакетов напрямую (так стабильнее воспроизводится в Китае):

```bash
cd /tmp

# Скачиваем 4 необходимых пакета
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container1_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-container-tools_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit-base_1.17.4-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit_1.17.4-1_amd64.deb

# Устанавливаем по порядку
dpkg -i libnvidia-container1_1.17.4-1_amd64.deb
dpkg -i libnvidia-container-tools_1.17.4-1_amd64.deb
dpkg -i nvidia-container-toolkit-base_1.17.4-1_amd64.deb
dpkg -i nvidia-container-toolkit_1.17.4-1_amd64.deb

# Настраиваем Docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Проверка (должен вывестись nvidia-smi)
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

> 💡 **Подсказка:** ссылки для загрузки автоматически перенаправляются на зеркала в Китае (nvidia.cn), скорость хорошая.

---

## 4. Настройка удалённого рабочего стола VNC

### 4.1 Установка рабочего стола и VNC

```bash
apt update
apt install -y xfce4 xfce4-goodies tigervnc-standalone-server tigervnc-common novnc websockify
```

> 💡 **Подсказки при установке:**
> - Если появится окно «Daemons using outdated libraries», нажмите Tab, выберите `<Ok>` и подтвердите Enter. Это обычное сообщение после обновления системы с просьбой перезапустить службы.

### 4.2 Настройка VNC

```bash
# Задаём пароль VNC
vncpasswd
# Вводим пароль (не короче 6 символов); на вопрос о view-only отвечаем n

# Создаём конфигурацию VNC
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# Запускаем службу VNC
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

# Проверяем, что VNC запустился
netstat -tlnp | grep 5901
# Должен быть слушающий порт 5901 у Xtigervnc, примерно такой вывод:
# tcp        0      0 0.0.0.0:5901            0.0.0.0:*               LISTEN      26811/Xtigervnc

# Запускаем noVNC (доступ через веб)
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &
```

### 4.3 Доступ к VNC

#### Способ 1: SSH-туннель (рекомендуется, безопаснее)

Открывать порт VNC в группе безопасности не нужно. На локальном компьютере выполните:

```bash
# Windows PowerShell / Mac Terminal / Linux
ssh -L 6080:localhost:6080 root@<ваш публичный IP>
```

Не закрывайте SSH-соединение и откройте в браузере: `http://localhost:6080/vnc.html`.

#### Способ 2: прямой доступ (нужно открыть порт)

Если выбираете прямой доступ, откройте порт 6080 в группе безопасности (только для своего IP):

Откройте в браузере: `http://<ваш публичный IP>:6080/vnc.html`.

Введите пароль VNC — и вы попадёте на рабочий стол.

> ⚠️ **Предупреждение о безопасности:** протокол VNC слабо защищён, открытый порт легко подбирается. Настоятельно рекомендуем SSH-туннель.

### 4.4 Автозапуск VNC (необязательно)

```bash
cat > /etc/systemd/system/vncserver.service << 'EOF'
[Unit]
Description=VNC Server
After=network.target

[Service]
Type=forking
User=root
ExecStart=/usr/bin/vncserver :1 -geometry 1920x1080 -depth 24
ExecStop=/usr/bin/vncserver -kill :1

[Install]
WantedBy=multi-user.target
EOF

systemctl enable vncserver
```

---

## 5. Развёртывание Isaac Sim и Isaac Lab

### 5.1 Вход в NGC и загрузка образа

Образ Isaac Sim хранится в NVIDIA NGC (GPU Cloud), сначала нужно войти:

```bash
# Вход в NGC
docker login nvcr.io
```

Система спросит имя пользователя и пароль:
- **Username:** введите `$oauthtoken` (фиксированное значение, скопируйте как есть);
- **Password:** введите API-ключ NGC, полученный в разделе 2.0.

Надпись `Login Succeeded` означает успешный вход.

```bash
# Загружаем образ Isaac Sim 4.2.0 (около 15 ГБ, займёт время)
docker pull nvcr.io/nvidia/isaac-sim:4.2.0
```

После успешной загрузки появится:
```
Status: Downloaded newer image for nvcr.io/nvidia/isaac-sim:4.2.0
```

Проверка, что образ скачан:
```bash
docker images | grep isaac-sim
# Должно быть примерно такое:
# nvcr.io/nvidia/isaac-sim   4.2.0   <IMAGE_ID>   <SIZE>
```

### 5.2 Запуск контейнера

```bash
# На хосте настраиваем права X11
export DISPLAY=:1
xhost +local:docker

# Запускаем контейнер
# Важно: --network host и xhost + упрощают настройку, их стоит использовать только для коротких экспериментов
docker run -it --name isaac-sim-gui --gpus all --network host \
  -e DISPLAY=:1 \
  -e ACCEPT_EULA=Y \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --entrypoint bash \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

### 5.3 Подготовка: загрузка зависимостей на хосте

Из-за сетевых ограничений внутри контейнера лучше скачать Isaac Lab и robomimic на хосте:

```bash
# Выходим из контейнера, возвращаемся на хост
exit

# Выполняем на хосте (в контейнер не заходим)
cd ~

# Устанавливаем утилиты загрузки (если их нет)
apt update
apt install -y git wget unzip

# Скачиваем Isaac Lab через зеркало в Китае
git clone --depth 1 --branch v1.4.1 https://gitclone.com/github.com/isaac-sim/IsaacLab.git

# Скачиваем robomimic (для задач имитационного обучения)
wget https://github.com/ARISE-Initiative/robomimic/archive/refs/heads/master.zip
unzip master.zip
mv robomimic-master robomimic

# Запускаем контейнер заново
docker start isaac-sim-gui

# Копируем внутрь контейнера
docker cp IsaacLab isaac-sim-gui:/isaac-sim/
docker cp robomimic isaac-sim-gui:/tmp/
```

### 5.4 Установка зависимостей внутри контейнера

```bash
# Заходим в контейнер
docker exec -it isaac-sim-gui bash

# Устанавливаем инструменты сборки и библиотеки EGL (нужны robomimic)
apt update
apt install -y build-essential cmake pkg-config git
apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev

# Устанавливаем robomimic
cd /tmp/robomimic
/isaac-sim/python.sh -m pip install -e .
```

### 5.5 Исправление конфигурации Isaac Lab

В файле зависимостей Isaac Lab есть две проблемы, которые нужно исправить:

```bash
cd /isaac-sim/IsaacLab

# Сохраняем исходный файл
cp source/extensions/omni.isaac.lab_tasks/setup.py source/extensions/omni.isaac.lab_tasks/setup.py.bak

# Исправление 1: несовпадение имени rsl-rl
# Причина: в setup.py указано имя репозитория GitHub "rsl-rl", а на PyPI пакет называется "rsl-rl-lib".
# Это несогласованность именования в upstream, на работу не влияет.
sed -i '46s/"rsl-rl@git+https:\/\/github.com\/leggedrobotics\/rsl_rl.git"/"rsl-rl-lib==2.3.0"/' source/extensions/omni.isaac.lab_tasks/setup.py

# Исправление 2: robomimic уже установлен вручную, удаляем git-клонирование
# Причина: setup.py пытается клонировать robomimic из GitHub, а сеть внутри контейнера нестабильна.
# Мы уже скачали его на хост и скопировали внутрь, поэтому эту строку удаляем.
sed -i '/robomimic@git/d' source/extensions/omni.isaac.lab_tasks/setup.py

# Проверка исправлений
echo "=== Строка 46 (rsl-rl) ==="
sed -n '46p' source/extensions/omni.isaac.lab_tasks/setup.py
echo "=== Строки 50–56 (robomimic должен быть удалён) ==="
sed -n '50,56p' source/extensions/omni.isaac.lab_tasks/setup.py
```

Ожидаемый результат:
```
=== Строка 46 (rsl-rl) ===
"rsl-rl": ["rsl-rl-lib==2.3.0"],
=== Строки 52–56 (robomimic) ===
# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))
```

Если в строке 46 всё ещё `rsl-rl@git+https://...`, значит sed не сработал: проверьте номер строки.

### 5.6 Установка Isaac Lab

```bash
cd /isaac-sim/IsaacLab

# Создаём символическую ссылку
ln -s /isaac-sim _isaac_sim

# Чиним pip (в контейнере Isaac Sim pip может быть повреждён)
/isaac-sim/python.sh -m ensurepip --upgrade
/isaac-sim/python.sh -m pip install --upgrade pip setuptools

# Устанавливаем Isaac Lab (около 5–10 минут)
./isaaclab.sh --install
```

Во время установки будет много загрузок и установок зависимостей; в конце должно появиться:
```
Successfully installed omni-isaac-lab_tasks-0.10.18 ...
```

### 5.7 Проверка Isaac Lab

```bash
# Запускаем демо внутри контейнера
export DISPLAY=:1
cd /isaac-sim/IsaacLab

# Демо с манипулятором
./isaaclab.sh -p source/standalone/demos/arms.py

# Демо с двуногим роботом
./isaaclab.sh -p source/standalone/demos/bipeds.py

# Обучение гуманоидного робота с подкреплением
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Humanoid-Direct-v0 \
  --num_envs 64 \
  --max_iterations 100
```

На рабочем столе VNC должна отображаться симуляция. Чтобы остановить программу, нажмите `Ctrl+C`.

---

## 6. Настройка окружения GR00T

### 6.1 Установка Miniconda

```bash
# Выходим из контейнера Docker, возвращаемся на хост
exit

# Устанавливаем Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init
source ~/.bashrc

# Принимаем условия обслуживания Conda (требуется в новых версиях)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 6.2 Создание окружения GR00T

```bash
# Создаём окружение Python 3.10
conda create -n groot python=3.10 -y
conda activate groot

# Устанавливаем PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Установка flash-attn (необязательно, рекомендуем пропустить)

> ⚠️ **Важно:** сборка flash-attn занимает очень много времени (15–60 минут) и сильно нагружает систему: из-за этого может отваливаться SSH и VNC.
> GR00T работает и без flash-attn, только инференс немного медленнее.
> **Рекомендуем пропустить этот шаг** и перейти к разделу 6.3.

Если flash-attn всё же нужен (для максимальной скорости):

```bash
# Устанавливаем зависимости сборки
pip install numpy psutil packaging ninja

# Обязательно задаём переменные CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Ограничиваем число параллельных заданий сборки, чтобы не закончилась память
# При 32 ГБ памяти — MAX_JOBS=4
# При 58 ГБ и выше — MAX_JOBS=6
MAX_JOBS=6 pip install flash-attn --no-build-isolation -v
```

> ⏳ **Что происходит при сборке:**
> - Фаза `running bdist_wheel` может висеть 1–3 минуты — это нормально (идёт конфигурация cmake);
> - затем появится `[1/73] ...` — началась собственно компиляция (всего 73 файла);
> - вся сборка занимает 15–60 минут в зависимости от числа ядер CPU и значения MAX_JOBS.
>
> ⚠️ **Нормальные странности во время сборки:**
> - **Новое SSH-подключение может не подключиться:** сборка съедает ресурсы, sshd отвечает медленно;
> - **VNC может отключиться:** по той же причине;
> - **Открытые терминалы могут зависнуть:** высокая нагрузка на систему;
> - **Пока загрузка CPU выше 50 %, сборка идёт — не перезагружайте сервер принудительно!**
> - Когда загрузка CPU упадёт ниже 5 %, сборка завершена.
>
> 💡 Загрузку CPU можно смотреть на странице мониторинга экземпляра в консоли Alibaba Cloud, SSH для этого не нужен.
>
> ⚠️ **После сборки может потребоваться перезагрузка:**
> - после завершения сборки SSH может по-прежнему не подключаться;
> - перезапустите экземпляр в консоли Alibaba Cloud;
> - после перезагрузки снова выполните `MAX_JOBS=6 pip install flash-attn --no-build-isolation -v`;
> - в этот раз используется кэшированное колесо, установка займёт несколько секунд.

**Проверка установки:**

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
# Должна вывестись версия 2.8.3 или близкая
```

### 6.3 Установка Isaac-GR00T

```bash
cd ~

# Скачиваем Isaac-GR00T (обратите внимание: репозиторий в организации NVIDIA, а не NVIDIA-Omniverse)
# Способ 1: git clone (из Китая может быть медленно)
git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git

# Способ 2: зеркало gitclone
git clone --depth 1 https://gitclone.com/github.com/NVIDIA/Isaac-GR00T.git

# Способ 3 (рекомендуется): ускорение через ghproxy
curl -L -o isaac-groot.tar.gz https://ghproxy.cn/https://github.com/NVIDIA/Isaac-GR00T/archive/refs/heads/main.tar.gz
tar -xzf isaac-groot.tar.gz
mv Isaac-GR00T-main Isaac-GR00T

cd Isaac-GR00T
```

#### Установка GR00T

> ⚠️ **Важно:** не используйте `pip install -e .` напрямую: он заново собирает flash-attn и зависает.
>
> **Причина:** setup.py GR00T объявляет flash-attn зависимостью, и даже если он уже установлен, pip пытается его пересобрать.
> Поэтому используем `--no-deps`, а основные пакеты ставим вручную.

```bash
# Устанавливаем без автоматической подгрузки зависимостей
pip install -e . --no-build-isolation --no-deps

# Вручную ставим основные зависимости
# transformers обязательно фиксируем на 4.51.3: более новые версии вызывают ошибки загрузки модели
pip install transformers==4.51.3 safetensors einops peft diffusers tyro omegaconf pandas dm-tree termcolor av albumentations huggingface_hub deepspeed accelerate
pip install click datasets gymnasium lmdb matplotlib msgpack-numpy pyzmq wandb torchcodec

# Проверка установки
python -c 'from gr00t.policy.gr00t_policy import Gr00tPolicy; print("GR00T loaded!")'
```

> 💡 **Про предупреждения о версиях зависимостей:** после установки может появиться много предупреждений `pip's dependency resolver` о несовместимых версиях.
> Это потому, что setup.py GR00T жёстко фиксирует версии, но немного более новые версии на практике работают нормально.
> **Если проверочная команда выводит `GR00T loaded!`, установка прошла успешно, предупреждения можно игнорировать.**

> 💡 Так мы избегаем повторной сборки flash-attn через pip. Если flash-attn уже установлен, GR00T будет использовать его, если нет — переключится на стандартное внимание.

### 6.4 Загрузка моделей GR00T

**Внимание:** в Китае для HuggingFace нужно использовать зеркало.

```bash
conda activate groot
cd ~/Isaac-GR00T

# Загружаем универсальную модель GR00T-N1.6-3B
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-3B', local_dir='/path/to/groot_n16_model')"

# Загружаем модель, специализированную для робота G1 (для оценки)
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-G1-PnPAppleToPlate', local_dir='/path/to/groot_g1_model')"
```

### 6.5 Проверка инференса GR00T

```bash
cd ~/Isaac-GR00T

# Создаём тестовый скрипт
cat > test_groot.py << 'EOF'
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

print('Loading GR00T N1.6...')
policy = Gr00tPolicy(
    model_path='/path/to/groot_n16_model',
    embodiment_tag=EmbodimentTag('gr1'),
    device='cuda',
)

obs = {
    'video': {
        'ego_view_bg_crop_pad_res256_freq20': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
    },
    'state': {
        'left_arm': np.random.rand(1, 1, 7).astype(np.float32),
        'right_arm': np.random.rand(1, 1, 7).astype(np.float32),
        'left_hand': np.random.rand(1, 1, 6).astype(np.float32),
        'right_hand': np.random.rand(1, 1, 6).astype(np.float32),
        'waist': np.random.rand(1, 1, 3).astype(np.float32),
    },
    'language': {
        'task': [['pick up the red apple']],
    },
}

action = policy.get_action(obs)
print('Action output:')
for k, v in action[0].items():
    print(f'  {k}: shape={v.shape}')
print('GR00T Inference Success!')
EOF

# Запускаем тест
python test_groot.py
```

---

## 7. Оценка GR00T + MuJoCo

### 7.1 Подготовка

```bash
conda activate groot
cd ~/Isaac-GR00T

# Устанавливаем зависимости
pip install uv
apt-get update && apt-get install -y libegl1-mesa-dev libglu1-mesa git-lfs
git lfs install

# Настраиваем git на использование ghproxy (ключевой шаг!)
git config --global http.version HTTP/1.1
git config --global url."https://ghproxy.cn/https://github.com/".insteadOf "https://github.com/"

# Клонируем подмодуль GR00T-WholeBodyControl (обязательно через git clone, а не tar.gz)
mkdir -p external_dependencies
cd external_dependencies
rm -rf GR00T-WholeBodyControl
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git

cd ~/Isaac-GR00T

# Проверяем, что LFS-файлы загружены (должно быть 100 КБ+, а не 131 байт)
ls -la ~/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/robot_model/model_data/g1/meshes/left_hip_pitch_link.STL
```

> ⚠️ **Важно:**
> - обязательно используйте `git clone`, а не `curl` для tar.gz, потому что репозиторий содержит большие файлы Git LFS (STL-модели роботов);
> - tar.gz даёт только LFS-указатели размером 131 байт, а не настоящие модели;
> - если STL-файл весит 131 байт, MuJoCo выдаст ошибку `Failed to determine STL storage representation`.

### 7.2 Изменение скрипта установки и его запуск

```bash
cd ~/Isaac-GR00T

# Комментируем команды git submodule и git lfs pull (мы их уже выполнили вручную)
sed -i 's/^git submodule update/#git submodule update/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
sed -i 's/^git -C/#git -C/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# Комментируем удаление robosuite и его git clone (скачаем вручную)
sed -i '26s/^rm/#rm/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
sed -i '27s/^git clone/#git clone/' gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# Скачиваем robosuite вручную (git clone из скрипта setup часто падает)
cd ~/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg
curl -L -o robosuite.tar.gz https://ghproxy.cn/https://github.com/xieleo5/robosuite/archive/refs/heads/leo/support_g1_locomanip.tar.gz
tar -xzf robosuite.tar.gz
mv robosuite-leo-support_g1_locomanip gr00trobosuite
rm robosuite.tar.gz

# Запускаем скрипт установки
cd ~/Isaac-GR00T
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

Скрипт сделает следующее:
1. создаст виртуальное окружение uv;
2. установит robosuite, robocasa, lerobot и другие зависимости (загрузка torch и других больших пакетов займёт несколько минут);
3. проверит окружение (должны появиться `Imports OK` и `Env OK`).

> 💡 **Диагностика сетевых проблем:**
> - ошибка `HTTP2 framing layer`: убедитесь, что выполнено `git config --global http.version HTTP/1.1`;
> - ошибка `Connection timed out`: убедитесь, что настроено ускорение через ghproxy;
> - разрыв SSH: переподключитесь и снова запустите скрипт setup, уже скачанные пакеты берутся из кэша.

### 7.3 Изменение кода для визуализации в реальном времени

```bash
# Сохраняем исходный файл
cp ~/Isaac-GR00T/gr00t/eval/rollout_policy.py ~/Isaac-GR00T/gr00t/eval/rollout_policy.py.bak

# Меняем режим рендеринга
sed -i 's/os.environ\["MUJOCO_GL"\] = "egl"/os.environ["MUJOCO_GL"] = "glx"/' ~/Isaac-GR00T/gr00t/eval/rollout_policy.py
sed -i 's/onscreen=False/onscreen=True/' ~/Isaac-GR00T/gr00t/eval/rollout_policy.py
```

### 7.4 Запуск оценки

Нужны два терминала.

**Терминал 1: запуск сервера GR00T:**
```bash
cd ~/Isaac-GR00T
conda activate groot
python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/groot_g1_model \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper
```

Дождитесь надписи `Server is ready and listening on tcp://127.0.0.1:5555`.

**Терминал 2: запуск клиента оценки (выполняется в VNC):**

> 💡 **Как вставлять команды через VNC:**
> 1. В левом меню noVNC откройте «Буфер обмена» (Clipboard).
> 2. Вставьте команду ниже в текстовое поле буфера.
> 3. На рабочем столе VNC откройте терминал (правый клик по рабочему столу → Terminal).
> 4. В терминале вставьте команду комбинацией `Ctrl+Shift+V`.
> 5. Нажмите Enter.
>
> ⚠️ **Внимание:** команда записана в одну строку, чтобы при копировании многострочных команд не попали скрытые символы, которые вызывают ошибки вроде `invalid int value: '5555~'`.

```bash
cd ~/Isaac-GR00T && export DISPLAY=:1 && gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py --n_episodes 3 --max_episode_steps 500 --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc --n_action_steps 20 --n_envs 1 --policy_client_host 127.0.0.1 --policy_client_port 5555
```

На рабочем столе VNC появится окно MuJoCo, где в реальном времени видно, как робот G1 выполняет задачу «взять яблоко и положить на тарелку».

---

## 8. Частые проблемы и журнал ошибок

### Q1: Как установить NVIDIA Container Toolkit?

**Рекомендуемый способ:** установить deb-пакеты напрямую, см. раздел 3.4. Он самый стабильный и воспроизводимый.

**Запасной способ** (если хочется использовать репозиторий apt):
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
```

Внимание: nvidia.github.io в Китае может быть недоступен.

### Q2: WebRTC Livestream показывает чёрный экран

**Решение:** перейти на VNC, он стабильнее.

### Q3: Не клонируется Isaac Lab

**Симптом:** `fatal: unable to access 'https://github.com/isaac-sim/IsaacLab.git/'`

**Решение:** скачать через зеркало на хосте и скопировать в контейнер:
```bash
# На хосте
cd ~
git clone --depth 1 --branch v1.4.1 https://gitclone.com/github.com/isaac-sim/IsaacLab.git
docker cp IsaacLab isaac-sim-gui:/isaac-sim/
```

### Q4: Не устанавливается robomimic

**Симптом:** `Failed to build 'robomimic'` или ошибка компиляции `egl_probe`.

**Решение:**
```bash
# Скачиваем на хосте
cd ~
wget https://github.com/ARISE-Initiative/robomimic/archive/refs/heads/master.zip
unzip master.zip
mv robomimic-master robomimic
docker cp robomimic isaac-sim-gui:/tmp/

# Внутри контейнера ставим библиотеки EGL и инструменты сборки
docker exec -it isaac-sim-gui bash
apt update
apt install -y build-essential cmake pkg-config
apt install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev

# Устанавливаем robomimic
cd /tmp/robomimic
/isaac-sim/python.sh -m pip install -e .
```

### Q5: Ошибка в конфигурации setup.py

**Симптом:** `IndentationError` или `Could not find a version that satisfies the requirement rsl-rl`.

**Решение:** исправить файл конфигурации по разделу 5.5:
```bash
cd /isaac-sim/IsaacLab
sed -i '46s/"rsl-rl@git+https:\/\/github.com\/leggedrobotics\/rsl_rl.git"/"rsl-rl-lib==2.3.0"/' source/extensions/omni.isaac.lab_tasks/setup.py
sed -i '53,54d' source/extensions/omni.isaac.lab_tasks/setup.py
```

### Q6: Новейшая версия Isaac Lab выдаёт ошибку

**Причина:** новейшая версия требует Isaac Sim 4.5 и выше.
**Решение:** клонировать конкретную версию: `git clone --depth 1 --branch v1.4.1`.

### Q7: Конфликт версий rsl-rl

**Решение:** уже исправлено в разделе 5.5 через `rsl-rl-lib==2.3.0`.

### Q8: HuggingFace недоступен

**Решение:** использовать зеркало `HF_ENDPOINT=https://hf-mirror.com`.

### Q9: Ошибка shape mismatch при загрузке модели GR00T

**Причина:** загружена старая модель N1-2B, а код соответствует N1.6.
**Решение:** загрузить `nvidia/GR00T-N1.6-3B`.

### Q10: Проблемы со сборкой flash-attn

#### Симптом 1: во время сборки разрывается SSH/VNC, но загрузка CPU высокая

**Это нормально!** Сборка съедает ресурсы и замедляет систему.
- Смотрите загрузку CPU в консоли Alibaba Cloud.
- Когда CPU упадёт ниже 5 %, сборка завершена.
- **Не перезагружайте сервер принудительно!** Иначе сборка пойдёт впустую.

#### Симптом 2: загрузка CPU почти 0 %, сборка зависла

Процесс сборки аварийно завершился, нужно повторить:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
MAX_JOBS=4 pip install flash-attn --no-build-isolation -v
```

#### Симптом 3: после перезагрузки flash-attn пропал

Сборка прервалась, её нужно повторить. Если не хотите ждать, можно пропустить flash-attn:
```bash
cd ~/Isaac-GR00T
pip install -e . --no-deps
pip install transformers==4.51.3 safetensors einops peft diffusers tyro omegaconf pandas dm-tree termcolor av albumentations huggingface_hub deepspeed accelerate
pip install click datasets gymnasium lmdb matplotlib msgpack-numpy pyzmq wandb torchcodec
```

#### Симптом 4: установка прошла успешно, но выводятся предупреждения о версиях

```
gr00t 0.1.0 requires flash-attn==2.7.4.post1, but you have flash-attn 2.8.3 which is incompatible.
...
```

**Можно игнорировать.** GR00T жёстко фиксирует версии в setup.py, но немного более новые версии работают нормально. Важно только, чтобы `python -c "import flash_attn"` не выдавал ошибку.

> 💡 flash-attn — это компонент оптимизации производительности, а не обязательная часть. Без него GR00T работает.

### Q11: MuJoCo не показывает окно

**Решение:**
1. Убедитесь, что VNC запущен: `netstat -tlnp | grep 5901`;
2. задайте `export DISPLAY=:1`;
3. в коде поменяйте `MUJOCO_GL` на `glx`, а `onscreen` на `True`.

### Q12: VNC не подключается после запуска

**Симптом:** в журнале websockify появляется `Connection refused`.

**Решение:**
```bash
# Убиваем старые процессы
vncserver -kill :1
pkill -9 websockify

# Исправляем конфигурацию xstartup
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# Перезапускаем VNC (обратите внимание на параметр -localhost no)
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &
```

### Q13: Прерываемый экземпляр освободили

**Рекомендации:**
- задавайте разумную максимальную цену;
- регулярно делайте резервные копии важных данных;
- храните модели и код на диске данных.

### Q14: Не хватает места на диске

**Решение:**
- очистить кэш Docker: `docker system prune -a`;
- очистить кэш pip: `pip cache purge`;
- очистить кэш conda: `conda clean -a`;
- лучше использовать системный диск от 150 ГБ или подключить отдельный диск данных.

### Q15: VNC показывает рывки картинки

**Пояснение:** передача 3D-симуляции по VNC даёт задержку и потерю кадров, это нормально. VNC нужен, чтобы «проверить, что всё запустилось», а не для плавной работы.

**Продвинутые варианты:**
- использовать NoMachine (быстрее);
- записать видео и скачать его на локальный компьютер.

### Q16: Не клонируется GR00T-WholeBodyControl

**Симптомы:**
- `fatal: repository 'https://github.com/NVIDIA/GR00T-WholeBodyControl.git/' not found`
- `error: RPC failed; curl 16 Error in the HTTP2 framing layer`

**Причина:** подмодуль находится в организации `NVlabs`, а не `NVIDIA`; либо проблемы с сетью.

**Что посмотреть:** раздел 7.1.

**Решение:**
```bash
cd ~/Isaac-GR00T/external_dependencies

# Способ 1: правильный адрес
git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git

# Способ 2 (рекомендуется): ускорение через ghproxy
curl -L -o wbc.tar.gz https://ghproxy.cn/https://github.com/NVlabs/GR00T-WholeBodyControl/archive/refs/heads/main.tar.gz
tar -xzf wbc.tar.gz
mv GR00T-WholeBodyControl-main GR00T-WholeBodyControl
rm wbc.tar.gz
```

### Q17: Несовместимость версий transformers

**Симптом:** `AttributeError: 'Eagle3_VLConfig' object has no attribute '_attn_implementation_autoset'`

**Причина:** версия transformers слишком новая.

**Что посмотреть:** раздел 6.3, убедитесь, что установлен `transformers==4.51.3`.

**Решение:**
```bash
pip install transformers==4.51.3
```

---

## Приложение: шпаргалка по командам

```bash
# Запуск VNC
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
nohup websockify --web=/usr/share/novnc/ 6080 localhost:5901 > /dev/null 2>&1 &

# Вход в контейнер Isaac Sim
docker start isaac-sim-gui
docker exec -it isaac-sim-gui bash

# Запуск демо Isaac Lab внутри контейнера
export DISPLAY=:1
cd /isaac-sim/IsaacLab
./isaaclab.sh -p source/standalone/demos/arms.py

# Активация окружения GR00T
conda activate groot

# Запуск сервера GR00T
python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/groot_g1_model \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper
```

---

## Ссылки

- [Документация Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Isaac Lab на GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac-GR00T на GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T-WholeBodyControl на GitHub](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [Модель GR00T на HuggingFace](https://huggingface.co/nvidia/GR00T-N1.6-3B)

---

*Руководство завершено: январь 2026 года.*
*Окружение проверки: Alibaba Cloud ecs.gn7i-c16g1.4xlarge (A10 24 ГБ).*
