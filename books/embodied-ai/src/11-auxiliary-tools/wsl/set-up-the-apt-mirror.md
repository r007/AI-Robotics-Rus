# Настройка зеркала apt

Сначала замените источники apt на зеркало Tsinghua:

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak && sudo tee /etc/apt/sources.list > /dev/null <<'EOF'
# Источники исходного кода закомментированы, чтобы ускорить apt update. При необходимости раскомментируйте их
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
```

Затем установите рабочий стол и xrdp:

```bash
sudo apt update

sudo apt install -y xfce4 xfce4-goodies
sudo apt install -y xrdp


echo "xfce4-session" > ~/.xsession

# Редактируем скрипт запуска xrdp
sudo sed -i 's/^test -x \/etc\/X11\/Xsession && exec \/etc\/X11\/Xsession$/startxfce4/' /etc/xrdp/startwm.sh

# Или вручную откройте /etc/xrdp/startwm.sh и добавьте в конец файла:
# startxfce4

# Порт по умолчанию — 3389. При необходимости отредактируйте конфигурацию
sudo nano /etc/xrdp/xrdp.ini
# Найдите строку port=3389 и при необходимости замените порт

sudo service xrdp start

# Проверка состояния службы
sudo service xrdp status

# В WSL2 можно добавить в конец ~/.bashrc или ~/.zshrc:
# sudo service xrdp start
```

Подробные шаги:

# Запуск графического интерфейса WSL через xrdp

## 1. Установка графической среды

Сначала выберите и установите рабочий стол. Ниже два распространённых варианта.

### Вариант A: Xfce (лёгкий, рекомендуется)

```bash
sudo apt update
sudo apt install -y xfce4 xfce4-goodies
```

### Вариант B: Ubuntu Desktop (полный рабочий стол)

```bash
sudo apt update
sudo apt install -y ubuntu-desktop
```

## 2. Установка xrdp

```bash
sudo apt install -y xrdp
```

## 3. Настройка xrdp

```bash
# Настраиваем xrdp на использование Xfce
echo "xfce4-session" > ~/.xsession

# Редактируем скрипт запуска xrdp
sudo sed -i 's/^test -x \/etc\/X11\/Xsession && exec \/etc\/X11\/Xsession$/startxfce4/' /etc/xrdp/startwm.sh

# Или вручную откройте /etc/xrdp/startwm.sh и добавьте в конец файла:
# startxfce4
```

## 4. Настройка порта xrdp (необязательно)

```bash
# Порт по умолчанию — 3389. При необходимости отредактируйте конфигурацию
sudo nano /etc/xrdp/xrdp.ini
# Найдите строку port=3389 и при необходимости замените порт
```

## 5. Запуск службы xrdp

```bash
sudo service xrdp start

# Проверка состояния службы
sudo service xrdp status
```

## 6. Автозапуск при старте системы (необязательно)

```bash
# В WSL2 можно добавить в конец ~/.bashrc или ~/.zshrc:
# sudo service xrdp start
```

## 7. Подключение к графическому интерфейсу

**В Windows:**

1. Нажмите `Win + R`, чтобы открыть окно «Выполнить».
2. Введите `mstsc` и нажмите Enter, чтобы открыть «Подключение к удалённому рабочему столу».
3. В поле «Компьютер» введите `localhost:3389` или `127.0.0.1:3389`.
4. Нажмите «Подключить».
5. Введите имя пользователя и пароль WSL.

## 8. Устранение неполадок

### Чёрный экран или нет подключения

```bash
# Перезапускаем службу xrdp
sudo service xrdp restart

# Проверяем, не занят ли порт
netstat -an | grep 3389

# Смотрим журналы xrdp
cat /var/log/xrdp.log
cat /var/log/xrdp-sesman.log
```

### Оптимизация настроек

В `/etc/xrdp/xrdp.ini` можно изменить:

```ini
# Глубина цвета (для повышения производительности)
max_bpp=16

# Настройки сжатия
compression_level=9
```

## 9. Дополнительные советы

**Производительность:**
- Если возникают проблемы с производительностью, используйте Xfce, а не GNOME или KDE.
- Снижение глубины цвета ускоряет отклик.

**Безопасность:**
```bash
# Разрешаем подключения только с локального компьютера
sudo ufw allow from 127.0.0.1 to any port 3389
```

**Скрипт запуска (в Windows PowerShell):**
```powershell
# Создаём скрипт запуска
wsl -d Ubuntu sudo service xrdp start
```

## 10. Альтернативные варианты

Если xrdp не работает, можно попробовать:

- **VcXsrv** или **X410**: X-сервер для Windows;
- **WSLg**: встроенная в WSL2 поддержка графики (нужна Windows 11 или последняя версия Windows 10).
