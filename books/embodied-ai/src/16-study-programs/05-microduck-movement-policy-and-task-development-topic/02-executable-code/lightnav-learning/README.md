# LightNav-0: многосценная навигация

Используется официальная предобученная модель LightNav-0; визуально-языковая навигация работает на GPU. Модель навигации не обучается и плата не подключается.

Сопровождающий Notebook 8 содержит пути по умолчанию, работу с сервисом модели, три сцены, реальные предсказания, видео и сопоставление траекторий. Это [8-й день плана](../../learning-plan.md): изображение, языковая команда, траектория, MPC и управление ходьбой.

## Подготовка среды

Модель и симулятор работают в двух окружениях; веса и среда хранятся вне репозитория. Пример для Ubuntu:

```bash
git clone https://github.com/lightorigins/LightNav-0.git ~/projects/LightNav-0
uv venv --python 3.11 /path/to/envs/lightnav-0
uv pip install --python /path/to/envs/lightnav-0/bin/python -e "$HOME/projects/LightNav-0[vllm,video]"
/path/to/envs/lightnav-0/bin/hf download LightOriginsHQ/LightNav-0 \
  --local-dir /path/to/checkpoints/LightNav-0

uv venv --python 3.11 /path/to/envs/lightnav-mujoco
uv pip install --python /path/to/envs/lightnav-mujoco/bin/python \
  -e "$HOME/projects/LightNav-0/mujoco_demo[microduck]" \
  jupyterlab nbclient nbformat nbconvert ipykernel matplotlib imageio imageio-ffmpeg websocket-client
```

Официальные ресурсы робота MicroDuck и стратегия ходьбы берутся из исходного проекта:

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git ~/workspaces/microduck_rl
mkdir -p /path/to/checkpoints/microduck-navigation
curl -fL https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/alpha_walking.onnx \
  -o /path/to/checkpoints/microduck-navigation/alpha_walking.onnx
```

Уже существующие окружения, код и веса можно использовать повторно. Вывод модели требует зафиксированной версии `vllm==0.19.1` из исходного проекта; симулятор требует MuJoCo и ресурсы сцены. Используйте ускорение, поддерживаемое официальной инструкцией; конкретное устройство в тексте не указывается.

## Открытие и запуск

```bash
/path/to/envs/lightnav-mujoco/bin/python -m ipykernel install --user \
  --name microduck-lightnav --display-name 'MicroDuck LightNav'
cd '<путь к теме>'
/path/to/envs/lightnav-mujoco/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8890
```

Откройте `03-Notebook/08-lightnav0-vision-language-navigation-gpu-multi-scene.ipynb`, выберите указанное ядро и запускайте ячейки сверху вниз. Параметры по умолчанию уже заполнены. На другой машине можно задать переменные `LIGHTNAV_ROOT`, `LIGHTNAV_MODEL_ENV`, `LIGHTNAV_MODEL_PATH`, `LIGHTNAV_ROBOT_MODEL`, `LIGHTNAV_WALKING_POLICY` и `LIGHTNAV_OUTPUT`.

По умолчанию записи сохраняются в `/path/to/lightnav-recordings/teaching/notebook`, а уменьшенные копии видео — в `03-Notebook/outputs/lightnav_gpu` для просмотра в вебе. Этот каталог в репозиторий не коммитится. При запуске корнем веб-сервера выбирается каталог темы, поэтому относительные ссылки в учебнике и в пакете кода работают корректно.

Полный пакетный запуск выполняется так:

```bash
cd '<путь к теме>/03-Notebook'
/path/to/envs/lightnav-mujoco/bin/jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=microduck-lightnav \
  '08-lightnav0-vision-language-navigation-gpu-multi-scene.ipynb'
/path/to/envs/lightnav-mujoco/bin/jupyter trust '08-lightnav0-vision-language-navigation-gpu-multi-scene.ipynb'
```

## Сцены и управление

| Сцена | Что изучается |
| :-- | :-- |
| Официальная квартира | Официальная многокомнатная сцена ProcTHOR `val_2` и интерфейс модели |
| Собственная гостиная | Перестановка комнаты, кресло, диван и цветочный горшок; наблюдаем подход к цели |
| Собственный коридор | Узкое пространство и боковая мебель; наблюдаем локальную траекторию и поворот |

Модель навигации получает изображение от первого лица и команду и возвращает плоскую траекторию размером 10×3. MPC [модельно-предиктивное управление] по траектории модели вычисляет скорость; официальная стратегия ходьбы через CPU-интерфейс ONNX Runtime выдаёт 14 мерных действий суставов. Гравитация, контакт и позиционные исполнительные механизмы берутся из официальной реализации.

Во время вывода модели симуляция приостанавливается. Видео кодируется по времени симуляции, а реальная задержка инференса записывается в отдельный отчёт. Координаты цели используются только для записи расстояния; в модель они не подаются и в планирование управления не входят.

Каждый запуск перезаписывает `*_latest.mp4` для каждой сцены и сохраняет соответствующие текстовые предсказания, траектории, флаги остановки и отчёт эксперимента. Большие видео в интерактивный учебник не встраиваются; сохраняются небольшие изображения и текстовый вывод, чтобы их можно было читать сразу.

## Файлы и источники

- `lightnav_lesson.py`: построение новых сцен, протокол модели, физическая замкнутая петля и запись видео.
- `build_notebook.py`: генерация учебника из структурированного формата; при повторной генерации старый вывод выполнения удаляется, и Notebook затем нужно выполнить полностью.
- Исходный код: [LightNav-0](https://github.com/lightorigins/LightNav-0).
- Сцены и интерфейс робота: [официальный симулятор](https://github.com/lightorigins/LightNav-0/blob/main/mujoco_demo/README.md).
- Веса навигации: [LightOriginsHQ/LightNav-0](https://huggingface.co/LightOriginsHQ/LightNav-0).
- Робот: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl).

Официальные сетки сцен распространяются по лицензии CC BY 4.0; собственные сцены изменены в стенах, полу и расстановке мебели, а записи изменений сохраняются вместе с выводом. Сетки робота подчиняются условиям вышестоящего проекта CC BY-NC-SA; код и стратегии — лицензиям своих вышестоящих проектов.
