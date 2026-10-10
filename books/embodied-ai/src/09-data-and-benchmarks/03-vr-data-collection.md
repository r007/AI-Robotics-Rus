# Руководство по использованию Python-библиотеки VR-телеуправления (проверено на Quest 3)

```bash
git clone https://github.com/qrafty-ai/teleop_xr.git
cd teleop_xr

pip install -e .

# Для поддержки IK установите дополнительные зависимости:
pip install spatialmath-python>=1.1.15 gitpython>=3.1.46 xacro>=2.1.1 \
    filelock>=3.20.3 viser>=1.0.21
pip install git+https://github.com/chungmin99/pyroki.git
pip install git+https://github.com/chungmin99/ballpark.git

cd webxr
npm install
npm run build

python -m teleop_xr.demo # Эта команда нужна только для проверки передачи данных.
```

```bash
mamba activate py310
cd C:\tbxr_real # Корень проекта должен быть неглубоким, иначе будут ошибки. Желательно, чтобы путь состоял только из латинских символов.

python -m teleop_xr.demo --mode ik --robot-class so101 --input-mode CONTROLLER --host 0.0.0.0 --port 4444 --no-tui

python -m teleop_xr.demo --mode ik --robot-class franka --input-mode CONTROLLER --host 0.0.0.0 --port 4444 --no-tui
```

Наденьте VR-очки.
Кнопка Grip рядом с большим пальцем: одновременное нажатие на обоих контроллерах имитирует движение правой руки.
Вход в режим происходит только при одновременном нажатии; если отпустить один из контроллеров, управление ставится на паузу.
Двойное нажатие два раза подряд — сброс.

Завершить процесс:

```powershell
Get-NetTCPConnection -LocalPort 4444 -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess -Unique | % { Stop-Process -Id $_ -Force }
```
