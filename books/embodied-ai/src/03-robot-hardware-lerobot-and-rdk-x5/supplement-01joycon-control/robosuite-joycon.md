# Дополнение 01. Управление Joy-Con в Robosuite (Linux)

```bash
# Создать окружение conda
conda create -n robosuite python=3.10
conda activate robosuite

# Установить robosuite
git clone https://github.com/box2ai-robotics/robosuite-joycon.git
cd robosuite-joycon
pip3 install -r requirements.txt
pip3 install -r requirements-extra.txt

# Установить joycon-robotics
cd ..
git clone https://github.com/box2ai-robotics/joycon-robotics.git
cd joycon-robotics
pip install -e .
sudo apt install dkms
make install

---

cd ..
cd robosuite-joycon

# По умолчанию — управление с клавиатуры
python robosuite/demos/demo_device_control.py

sudo apt install libhidapi-dev
# Один Joy-Con (R)
python robosuite/demos/demo_device_control_joycon_single.py

# Два Joy-Con (двуручный режим)
python robosuite/demos/demo_device_control_joycon_bimanual.py
```

```bash
sudo snap install bun-js
```

## Поиск и устранение неисправностей

### 1. Не хватает зависимостей

Python-пакет `hid` (он же `hidapi`) не может найти системную библиотеку, которую он вызывает.

Проблема состоит из двух уровней:

1. **Пакет Python (`hid`)**: установлен в окружении `micromamba`. Сам по себе он не общается с устройством, а служит «переводчиком» или «пультом».
2. **Системная библиотека (`libhidapi-*.so`)**: установлена в операционной системе как библиотека на языке C. Именно она отвечает за связь с USB HID-устройствами, например Joy-Con.

Сообщение `Unable to load any of the following libraries: libhidapi-hidraw.so ...` означает, что Python не нашёл «двигатель», с которым может работать.

### **Решение** 🛠️

Нужно установить недостающую библиотеку `hidapi` для вашей операционной системы.

**1. Установите библиотеку разработки `hidapi`**

Имя пакета отличается в разных дистрибутивах Linux. Откройте терминал и выполните:

- **Debian / Ubuntu** (в том числе Ubuntu в WSL):

  ```bash
  sudo apt update
  sudo apt install libhidapi-dev
  ```

  *Версия `-dev` обычно тянет за собой и библиотеки времени выполнения (`libhidapi-hidraw0` и др.), а также заголовочные файлы для сборки. Это самый надёжный вариант.*

- **Arch Linux**:

  ```bash
  sudo pacman -S hidapi
  ```

- **Fedora / CentOS**:

  ```bash
  sudo dnf install hidapi-devel
  ```

**2. (Важно) Настройте правила `udev`**

После установки библиотеки обычно нужно выдать права на устройство Joy-Con. Иначе обычный пользователь не сможет к нему обратиться, и программа не подключится к контроллеру, даже если библиотека найдена.

- **Создайте файл правил udev**: команда даёт всем пользователям права на чтение и запись для всех устройств Nintendo (Vendor ID `057e`):

  ```bash
  echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", MODE="0666"' | sudo tee /etc/udev/rules.d/99-joycon.rules
  ```

- **Перезагрузите правила udev**, чтобы они сразу начали работать:

  ```bash
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  ```

**3. Переподключите Joy-Con** 🎮

Отключите Bluetooth-соединение Joy-Con и подключите его заново, чтобы новые правила udev применились к устройству.

**4. Запустите Python-программу ещё раз**

После этих шагов вернитесь в окружение `micromamba` и снова запустите скрипт. Теперь модуль `hid` должен импортироваться и находить устройство.

```bash
python /home/user/17robo/robosuite-joycon/robosuite/demos/demo_device_control_joycon_bimanual.py
```
