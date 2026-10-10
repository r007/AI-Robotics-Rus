# Запуск сайта в Docker

Образ собирает книги из `books/` через `scripts/build_site.py` и отдаёт готовый сайт через nginx.

## Быстрый запуск

Из корня репозитория:

```bash
docker compose -f docker/compose.yaml up --build
```

После сборки сайт доступен по адресу `http://localhost:8080`.

## Только сборка образа

```bash
docker build -f docker/Dockerfile -t embodied-ai-book:local .
docker run --rm -p 8080:80 embodied-ai-book:local
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
