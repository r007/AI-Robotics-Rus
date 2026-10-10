# Пояснение к публичному коду

Скрипты здесь — публичные инструменты, отделённые от рабочего пространства соревнования. Они не заменяют официальные ассеты RoboTwin/Tron2.

## Строгий аудит обработанных данных ACT

```bash
python code/audit_processed_t4.py \
  --dataset-dir /path/to/processed_data \
  --expected 550 \
  --raw-manifest-sha256 eeca954a7e64dda818a0cca86534e34f6ffa1a8055bd37810f9781a264199901 \
  --output /tmp/processed_audit.json
```

Если аудит не прошёл, остановитесь: скрипт не должен автоматически удалять, дополнять кадры, менять порядок камер или править action.

## Создание манифеста SHA для открытых материалов

```bash
python code/build_public_manifest.py 15-challenges/TRONCamp-Mani \
  --output /tmp/troncamp_public_manifest.json
```

Скрипт отказывается включать в открытый манифест checkpoint, HDF5, pickle, видео и символические ссылки. Он не заменяет проверку удалённой копии после загрузки на Hugging Face.

## Проверка перед публикацией

```bash
bash code/scan_public_release.sh 15-challenges/TRONCamp-Mani
```

После успешного сканирования всё равно проверьте вручную Markdown, CSV и JSON: автоматическая проверка не распознаёт все личные данные, проприетарные ассеты или ошибочные выводы эксперимента.
