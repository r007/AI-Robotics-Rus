# Запуск сайта в Docker

Образ собирает каждую книгу из `books/` как отдельный mdBook, добавляет страницу каталога через `scripts/build_site.py` и отдаёт готовый сайт через nginx.

## Быстрый запуск

Из корня репозитория:

```bash
docker compose -f docker/compose.yaml up --build
```

После сборки каталог доступен по адресу `http://localhost:8080`, а книги — по адресам `/embodied-ai/`, `/dive-into-deep-learning/` и `/physical-ai-systems/`.

## Только сборка образа

```bash
docker build -f docker/Dockerfile -t ai-robotics-rus:local .
docker run --rm -p 8080:80 ai-robotics-rus:local
```

Первая сборка долгая: плагины mdBook компилируются из исходников. Последующие сборки используют кэш слоёв.

## Версии инструментов

| Компонент | Версия |
|---|---|
| mdBook | 0.5.4 |
| mdbook-mermaid | 0.17.1 |
| mdbook-katex | 0.10.0 |
| nginx | 1 (alpine) |

Версии закреплены в `Dockerfile` и в CI. При обновлении меняйте их в обоих местах.
