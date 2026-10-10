# Руководство по локальной и облачной настройке Isaac Sim

После этой главы вы сможете определить, подходит ли ваш компьютер или облачный сервер для Isaac Sim, и выполнить три типичные настройки: установку на локальной рабочей станции Windows, установку на рабочей станции Linux или облачном сервере и установку Python-окружения через micromamba/venv. В тексте также объясняется, на каком этапе подключается Isaac Lab и когда стоит переходить к воспроизведению GR00T.

Главный вывод такой: если вы просто изучаете GUI Isaac Sim, импорт роботов, датчики и базовые сцены, Windows-рабочей станции достаточно как входной среды. Для обучения с подкреплением в Isaac Lab, Docker, удалённого рабочего стола или работы нескольких человек надёжнее рабочая станция Linux или облачный GPU-сервер. Команды на облачном сервере и локальной машине почти одинаковы; главное отличие облака — удалённый вывод изображения, порты, группы безопасности и сохранение данных.

## 1. Сначала выберите путь установки

| Сценарий | Рекомендуемый путь | Комментарий |
| --- | --- | --- |
| Компьютер с Windows, нужно просто открыть GUI Isaac Sim | Установочный пакет Workstation | Проще всего объяснить новичкам: запуск через App Selector. |
| Компьютер с Windows, нужно управлять Isaac Sim через Python-окружение | micromamba/venv + pip | Подходит для дальнейшей работы с Isaac Lab, но пакет крупный, первый запуск расширений идёт медленно. |
| Рабочая станция Ubuntu с монитором | Пакет Workstation или pip | Workstation нагляднее, pip лучше сочетается с Isaac Lab. |
| Облачный сервер Ubuntu без монитора | Docker или pip + удалённый рабочий стол / потоковая трансляция | Нужно отдельно настроить VNC, WebRTC, NICE DCV или X11. |
| Цель — обучение в Isaac Lab | pip-вариант Isaac Sim + исходный код Isaac Lab | Официально рекомендуется сначала установить pip-пакет Isaac Sim, затем Isaac Lab. |
| Цель — воспроизведение GR00T | Сначала прочитать объявление версий, затем перейти к руководству по GR00T | Зависимости GR00T и Isaac очень тяжёлые; лучше отдельное окружение, не смешивать с базовой средой. |

Isaac Sim — высокоточная симуляционная платформа, а не обычная маленькая Python-библиотека. Он зависит от видеокарты NVIDIA RTX, драйвера, Omniverse Kit, PhysX, ассетов USD, рендеринга и большого числа расширений. При установке важно не только то, что `pip install` завершился без ошибок: нужно убедиться, что драйвер GPU, видеопамять, графический интерфейс и кэш расширений работают корректно.

## 2. Общая таблица загрузок

Чтобы вы могли пройти всё по шагам, здесь собраны все источники, которые понадобятся в главе. Версии Isaac Sim обновляются часто; если вы будете использовать версию 5.2 или новее, сначала посмотрите имена файлов и номера версий на официальной странице загрузок и замените URL в командах.

| Что | Получение | Сценарий применения |
| --- | --- | --- |
| Isaac Sim 5.1, Windows Workstation | [isaac-sim-standalone-5.1.0-windows-x86_64.zip](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip) | Знакомство с GUI на Windows |
| Isaac Sim 5.1, Linux Workstation | [isaac-sim-standalone-5.1.0-linux-x86_64.zip](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip) | Знакомство с GUI на Ubuntu |
| Пакет Isaac Sim 5.1 для Python | `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` | micromamba / venv / Isaac Lab |
| Docker-образ Isaac Sim 5.1 | `docker pull nvcr.io/nvidia/isaac-sim:5.1.0` | Облачный Linux-сервер, изоляция контейнера |
| Клиент WebRTC Streaming для Windows | [isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe) | Удалённое подключение к облачному Isaac Sim |
| Клиент WebRTC Streaming для Linux | [isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage](https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage) | Клиент Linux для удалённого Isaac Sim |
| Полный пакет ассетов Isaac Sim | [complete 001](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.001.zip), [complete 002](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.002.zip), [complete 003](https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.003.zip) | Офлайн-ассеты, локальный кэш, закрытая сеть |
| NVIDIA Container Toolkit | [официальная инструкция по установке](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Доступ Docker к GPU |
| Исходный код Isaac Lab | `git clone https://github.com/isaac-sim/IsaacLab.git` | Обучение и построение задач в Isaac Lab |

Самый надёжный путь для новичка — сначала поставить пакет Workstation и запустить GUI Isaac Sim; если потом понадобится Isaac Lab, перейти на окружение pip. Путь с облачным сервером начинается с настройки драйвера NVIDIA и GPU-runtime для Docker, а затем загрузки `nvcr.io/nvidia/isaac-sim:5.1.0`.

## 3. Требования к железу и системе

По официальным требованиям NVIDIA для Isaac Sim 5.1 платформа x86_64 поддерживает Ubuntu 22.04/24.04 и Windows 10/11. В минимальной конфигурации указано 32 ГБ оперативной памяти и не менее 50 ГБ SSD; для видеокарты важны RT-ядра NVIDIA RTX. Официальная документация также предупреждает, что сложные сцены, датчики и крупномасштабное обучение увеличивают требования к оперативной памяти и видеопамяти, а GPU с менее чем 16 ГБ памяти может не справиться со сложными сценами.

Это означает, что ноутбук RTX 3060 с 6 ГБ подходит для лёгкого обучения и проверки совместимости, но не как основная среда для сложного Isaac Lab, высокоразрешающих датчиков, многороботных сцен или больших моделей GR00T. Если вы не хотите арендовать сервер, можно сделать GUI и базовые сцены на локальной машине, а переход к обучению, синтетическим данным или инференсу больших моделей выполнить на облачном сервере.

## 4. Проверки перед установкой

В Windows PowerShell сначала проверьте драйвер NVIDIA, Python и свободное место на диске. Для pip-пакетов Isaac Sim 5.x нужен Python 3.11; если по умолчанию у вас Python 3.12 или 3.10, создайте отдельное окружение, не ставьте всё в системный Python.

```powershell
nvidia-smi
python --version
python -m pip --version
Get-PSDrive C
```

Если вы собираетесь ставить через pip, можно заранее проверить, видит ли PyPI и индекс NVIDIA пакет Isaac Sim. Эта команда только запрашивает список версий и не скачивает полный пакет.

```powershell
python -m pip index versions isaacsim --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
```

На Linux дополнительно проверьте GLIBC. Пакеты Isaac Sim 5.x через pip требуют GLIBC 2.35 или новее: Ubuntu 22.04 обычно подходит, Ubuntu 20.04 с GLIBC 2.31 для pip-пути не рекомендуется.

```bash
nvidia-smi
python3 --version
ldd --version
df -h
```

Контрольная точка 1: если `nvidia-smi` не видит GPU NVIDIA, не продолжайте установку Isaac Sim. Тогда причина обычно в драйвере видеокарты, типе GPU-инстанса в облаке, GPU-runtime Docker или среде удалённого рабочего стола, а не в самом Isaac Sim.

## 5. Установка Workstation на Windows

Новичкам в Windows лучше сначала поставить пакет Workstation: он ближе всего к привычной работе с настольным приложением. Скачать можно в браузере или прямо в PowerShell. Пакет занимает несколько десятков гигабайт, поэтому сначала проверьте свободное место и распакуйте в короткий путь вроде `C:\isaacsim`, чтобы избежать проблем с длинными путями Windows при установке и кэшировании.

Первый шаг — загрузить standalone-пакет для Windows:

```powershell
New-Item -ItemType Directory -Force C:\isaacsim-downloads
cd C:\isaacsim-downloads
Invoke-WebRequest `
  -Uri "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip" `
  -OutFile "isaac-sim-standalone-5.1.0-windows-x86_64.zip"
```

Если `Invoke-WebRequest` загружает медленно, скопируйте URL в браузер, IDM, aria2 или сетевой загрузчик учебного заведения и сохраните файл под тем же именем.

Второй шаг — распаковать и выполнить постустановочные действия:

```powershell
mkdir C:\isaacsim
cd C:\isaacsim-downloads
tar -xvzf "isaac-sim-standalone-5.1.0-windows-x86_64.zip" -C C:\isaacsim
cd C:\isaacsim
.\post_install.bat
.\isaac-sim.selector.bat
```

После запуска App Selector выберите Isaac Sim Full и нажмите START. Первый запуск прогревает кэш шейдеров и загружает расширения; пустое окно в течение нескольких минут — нормальное явление. Когда откроется основной интерфейс, сначала сделайте минимальную проверку графики: `Create > Environment > Simple Room`, затем `Create > Robots > Franka Emika Panda Arm`, после этого нажмите кнопку воспроизведения слева. Если сцена отображается и запускается, значит, работают локальный GUI, рендеринг, базовые ассеты и физическая симуляция.

Контрольная точка 2: эта проверка доказывает, что Isaac Sim открывается на вашей машине и запускает базовую сцену. Она не доказывает, что видеопамяти хватит для большого обучения и что ROS2, Isaac Lab или GR00T уже настроены.

## 6. Установка Windows через micromamba / venv

Если позже понадобится Isaac Lab или хочется управлять Isaac Sim как пакетом Python, используйте micromamba или venv. Рекомендуется создать отдельное окружение `env_isaacsim`, не ставьте его в окружение, в котором уже работают LeRobot, OpenVLA или эксперименты с PyTorch.

```powershell
micromamba create -n env_isaacsim python=3.11 -y
micromamba activate env_isaacsim
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

При первом запуске Isaac Sim нужно принять лицензионное соглашение NVIDIA Omniverse EULA. В PowerShell это можно сделать временной переменной окружения:

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"
isaacsim isaacsim.exp.compatibility_check
isaacsim
```

Если сначала нужно только проверить, подходит ли машина, можно установить меньший пакет для проверки совместимости:

```powershell
python -m pip install "isaacsim[compatibility-check]==5.1.0" --extra-index-url https://pypi.nvidia.com
$env:OMNI_KIT_ACCEPT_EULA="YES"
isaacsim isaacsim.exp.compatibility_check
```

Контрольная точка 3: `isaacsim isaacsim.exp.compatibility_check` запускает приложение проверки совместимости Isaac Sim. Оно подходит как предустановочная проверка состояния, но не заменяет полноценную проверку сцены.

## 7. Рабочая станция Linux и облачный сервер

Локальная рабочая станция Linux может использовать пакет Workstation напрямую. Первый шаг — загрузить пакет:

```bash
mkdir -p ~/isaacsim-downloads
cd ~/isaacsim-downloads
wget -c https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

Второй шаг — распаковать и запустить App Selector:

```bash
mkdir -p ~/isaacsim
cd ~/isaacsim-downloads
unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
./isaac-sim.selector.sh
```

На облачном сервере обычно нет физического монитора, поэтому GUI Workstation требует VNC, NICE DCV, пересылки X11 или Isaac Sim Livestream. Рекомендуется считать «облачный сервер» и «локальную рабочую станцию Linux» одной и той же установкой Isaac, только в облаке добавляется удалённый вывод изображения и открытые порты. Не открывайте порты напрямую в интернет; лучше использовать SSH-туннель или ограничивать источники IP в группах безопасности облака.

Если вы используете путь pip, команды для Linux такие:

```bash
micromamba create -n env_isaacsim python=3.11 -y
micromamba activate env_isaacsim
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
isaacsim isaacsim.exp.compatibility_check
isaacsim
```

Если `ldd --version` показывает GLIBC ниже 2.35, не устанавливайте pip-вариант на старой системе силой. Надёжнее перейти на Ubuntu 22.04/24.04 или использовать официальный бинарный пакет или контейнерный путь.

## 8. Установка Docker на Linux

Docker лучше подходит для облачных Linux-серверов, совместной работы и изоляции окружений. NVIDIA официально указывает, что контейнер Isaac Sim поддерживается только на Linux. Если вы хотите работать с GUI Isaac Sim на Windows, предпочтительны Workstation или pip; Docker под Windows не стоит делать основным учебным путём.

Первый шаг — установить Docker. Если на облачном образе Docker уже установлен, этот раздел можно пропустить:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
docker --version
```

Второй шаг — установить NVIDIA Container Toolkit. Ниже приведён официальный способ для Ubuntu; для других дистрибутивов ориентируйтесь на официальную документацию NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Третий шаг — проверить, что контейнер видит GPU:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Четвёртый шаг — загрузить образ Isaac Sim 5.1. Образ берётся из NVIDIA NGC; при первой загрузке может понадобиться вход в NGC. Если появилась ошибка аутентификации, зарегистрируйтесь на [NGC](https://ngc.nvidia.com/), создайте API-ключ и выполните `docker login nvcr.io`.

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/isaac-sim:5.1.0
```

Рекомендуется монтировать каталоги кэша, журналов и пользовательских данных Isaac Sim отдельно, чтобы кэш не пропадал при удалении контейнера:

```bash
mkdir -p ~/docker/isaac-sim/cache/main
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/config
mkdir -p ~/docker/isaac-sim/data
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/pkg
sudo chown -R 1234:1234 ~/docker/isaac-sim
```

Запуск интерактивного контейнера:

```bash
docker run --name isaac-sim --entrypoint bash -it --gpus all --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -u 1234:1234 \
  nvcr.io/nvidia/isaac-sim:5.1.0
```

Внутри контейнера сначала запустите проверку совместимости:

```bash
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

Контрольная точка 4: если проверка системы проходит, значит, Docker, NVIDIA Container Toolkit, драйвер GPU и контейнер Isaac Sim хотя бы работают в режиме без окна (headless). Это не означает, что удалённый GUI или Livestream уже настроены.

## 9. Как вывести изображение с удалённого облачного сервера

Исполняемые файлы Isaac Sim на облачном сервере и локальной машине одинаковые, но в облаке нужно решить, как увидеть изображение. Для учебных целей проще всего VNC или NICE DCV; если используется встроенная трансляция Isaac Sim (livestream), скачайте клиент WebRTC Streaming.

Чтобы подключиться с локального Windows-компьютера к облачному Isaac Sim, скачайте клиент:

```powershell
New-Item -ItemType Directory -Force C:\isaacsim-downloads
Invoke-WebRequest `
  -Uri "https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe" `
  -OutFile "C:\isaacsim-downloads\isaacsim-webrtc-streaming-client-1.1.5-windows-x64.exe"
```

Для Linux-клиента скачайте AppImage:

```bash
mkdir -p ~/isaacsim-downloads
cd ~/isaacsim-downloads
wget -c https://downloads.isaacsim.nvidia.com/isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
chmod +x isaacsim-webrtc-streaming-client-1.1.5-linux-x64.AppImage
```

Настройка удалённого вывода сильно зависит от групп безопасности облачного провайдера. В принципе не стоит открывать порты VNC или livestream для всего интернета: используйте SSH-туннель или разрешите доступ только с вашего фиксированного IP.

## 10. Настройка Isaac Lab

Isaac Lab — фреймворк для обучения роботов, построенный поверх Isaac Sim; он подходит для обучения с подкреплением, имитационного обучения, построения задач и параллельной симуляции. Рекомендуемый порядок такой: сначала убедитесь, что Isaac Sim запускается, и только потом устанавливайте Isaac Lab. Не ставьте Isaac Lab, если Isaac Sim ещё не открывается: тогда ошибки будет трудно локализовать.

Дальше всё построено на Isaac Lab 2.3.x и Isaac Sim 5.1. Исходный код Isaac Lab берётся с GitHub; сначала скачайте его и переключитесь на нужный релиз. Перед обновлением лучше открыть страницу релизов Isaac Lab и проверить, какая версия Isaac Sim ему соответствует.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
```

На Linux сначала установите базовые инструменты сборки, затем расширения Isaac Lab:

```bash
sudo apt update
sudo apt install -y cmake build-essential
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

На Windows в активированном окружении Isaac Python выполните:

```powershell
isaaclab.bat --install
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

Официальный скрипт проверки открывает пустую сцену с чёрным окном просмотра. Результат выглядит простым, но он доказывает, что Isaac Lab может вызвать текущее окружение Python, найти Isaac Sim, загрузить базовые расширения и открыть симуляционное приложение.

## 11. Как выбрать интеграцию с ROS2

Isaac Sim 5.1 поддерживает ROS2 Humble и Jazzy. Для Ubuntu 22.04 рекомендуется Humble, для Ubuntu 24.04 — Jazzy; для Windows 10/11 официальная документация советует использовать ROS2 Humble через WSL.

Новичкам не стоит в первый день ставить одновременно Isaac Sim и ROS2. Надёжнее порядок: сначала открыть Isaac Sim, затем установить ROS2, затем включить ROS2 Bridge. Стандартные типы сообщений вроде `std_msgs`, `geometry_msgs`, `nav_msgs` обычно запускаются без проблем; если нужны собственные интерфейсы ROS2, учитывайте совместимость Python 3.11 из Isaac Sim и версии Python системного ROS.

## 12. Когда переходить к GR00T

GR00T — не вводная настройка Isaac Sim, а более поздний этап практики с базовыми моделями робототехники. В репозитории есть [полное руководство по развёртыванию Isaac Lab + GR00T в Alibaba Cloud](02-complete-tutorial-on-deploying-isaac-lab-gr00t-in-alibaba-cloud.md), в котором зафиксирован исторически воспроизводимый набор версий: Isaac Sim 4.2.0, Isaac Lab v1.4.1, GR00T N1.6, PyTorch 2.5.1 и CUDA 12.1. Эта комбинация отличается от основной линии Isaac Sim 5.1 в данной главе.

Если вы только учитесь работать с локальным Isaac Sim, устанавливать GR00T сразу не нужно. Если цель — воспроизвести GR00T, лучше создать отдельный облачный сервер или отдельное окружение и следовать версиям исходного руководства; не устанавливайте зависимости GR00T в `env_isaacsim`. Так вы не допустите конфликтов PyTorch, CUDA, transformers, flash-attn и расширений Isaac.

## 13. Частые проблемы

Если установка Isaac Sim не удалась, сначала проверяйте по уровням. Первый уровень — железо и драйвер: работает ли `nvidia-smi`, достаточно ли нов драйвер, есть ли у GPU RT-ядра, не слишком ли мала видеопамять. Второй уровень — система: включена ли поддержка длинных путей в Windows, соответствует ли GLIBC требованиям pip-пакета в Linux, достаточно ли места на диске (хотя бы несколько десятков гигабайт). Третий уровень — Python: используется ли Python 3.11, установлен ли Isaac в чистое окружение, используется ли индекс NVIDIA PyPI.

Если первый запуск идёт очень долго, не закрывайте окно сразу. Isaac Sim при первом запуске загружает расширения, компилирует кэш шейдеров и инициализирует пользовательскую конфигурацию; 5–10 минут могут быть нормой. Действовать стоит только тогда, когда в журнале явно видна ошибка драйвера, загрузки расширения, непринятое EULA или аварийное завершение процесса.

Если на Windows pip-установка жалуется на слишком длинный путь, сначала включите поддержку длинных путей или выберите более короткий путь окружения. При установке Workstation тоже лучше использовать короткий путь вроде `C:\isaacsim`, а не глубоко вложенный каталог в личных документах.

Если на облачном сервере не открывается GUI, сначала проверьте, запускается ли Isaac в режиме без окна. Если проверка совместимости проходит, проблема, скорее всего, в VNC, WebRTC, DISPLAY, Xauthority или портах в группе безопасности, а не в самом Isaac Sim. В таком случае сначала проверьте headless-команду в контейнере, а затем разбирайтесь с удалённым выводом.

## 14. Лёгкая проверка для этого руководства

В этой главе не выполнялась полная загрузка установочного пакета Isaac Sim объёмом около 40 ГБ на машине репозитория, потому что это требует много времени и места на диске. Выполненные лёгкие проверки: `nvidia-smi` на локальной машине видит GPU NVIDIA GeForce RTX 3060 Laptop, версия драйвера 581.95, видеопамять 6144 МиБ; текущая версия Python — 3.11.15; версия micromamba — 2.3.0; через индексы PyPI и NVIDIA PyPI видны версии пакета `isaacsim` 5.1.0.0 и 5.0.0.0.

Эти проверки подтверждают, что на машине есть базовый драйвер NVIDIA, Python 3.11 и доступ к индексам пакетов, но не доказывают, что 6 ГБ видеопамяти хватит для сложных сцен Isaac. Передавая руководство учащимся, всё же стоит пройти контрольные точки по шагам.

## Источники

- [Установка Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)
- [Загрузки Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- [Требования Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Установка Workstation Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
- [Установка Python-окружения Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Установка в контейнер Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
- [Установка ROS 2 для Isaac Sim 5.1 (NVIDIA)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [Установка NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Isaac Lab 2.3: установка через pip-пакет Isaac Sim](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
