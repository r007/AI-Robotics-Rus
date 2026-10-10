# LingBot-Map Web Demo

Это лёгкая локальная веб-обёртка для LingBot-Map, которая используется для smoke test в учебнике. Она принимает непрерывное видео, вызывает локальный `demo_render/batch_demo.py` из LingBot-Map и показывает результат рендеринга облака точек.

## Использование

```bash
export LINGBOT_MAP_ROOT=/path/to/lingbot-map
export LINGBOT_MAP_CKPT=/path/to/lingbot-map/checkpoints/lingbot-map-long.pt
export LINGBOT_MAP_PYTHON=/path/to/mamba/envs/lingbot-map/bin/python
export LINGBOT_MAP_RUN_ROOT=/path/to/lingbot_map_runs

pip install -r requirements.txt
python app.py
```

Откройте в браузере:

```text
http://127.0.0.1:7860
```

Результаты по умолчанию записываются в `LINGBOT_MAP_RUN_ROOT`. Не добавляйте в репозиторий учебника веса моделей, исходные видео, файлы предсказаний `.npz` и длинные выходные видео.
