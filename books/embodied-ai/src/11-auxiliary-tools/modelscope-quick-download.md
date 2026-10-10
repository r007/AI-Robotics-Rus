# Быстрая загрузка с ModelScope

```bash
pip install modelscope
pip install "datasets>=2.16.0,<3.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Скрипт `download.py`:

```python
from modelscope.hub.snapshot_download import snapshot_download
import os

# 1. Параметры загрузки
repo_id = 'agibot_world/GenieSim3.0-Dataset'
# Путь для сохранения на локальном диске
local_dir = "/path/to/data/openpi/checkpoints/organize_items"

# 2. Загрузка
# allow_patterns задаёт нужные подкаталоги или файлы
# Будет загружено только содержимое папки organize_items: так экономится время и место
print(f"Загрузка checkpoints из {repo_id}...")
snapshot_download(
    repo_id,
    local_dir=local_dir,
    allow_patterns=[
        'checkpoints/organize_items/**',  # Все файлы в этом каталоге
    ],
    repo_type='dataset'  # Явно указываем, что это репозиторий датасета
)

print(f"Загрузка завершена! Файлы сохранены в: {local_dir}")
```
