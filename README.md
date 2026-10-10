# AI-Robotics-Rus

Библиотека русскоязычных переводов по воплощённому ИИ и робототехнике.

Каждая книга лежит в своём каталоге `books/<книга>/` и содержит файл `book.toml` и исходные страницы в `src/`. Скрипт `scripts/build_site.py` объединяет книги в один сайт mdBook.

## Книги

| Книга | Статус |
|---|---|
| [Воплощённый ИИ и робототехника](books/embodied-ai/) | перевод в работе |
| [Введение в робототехнику](books/robotics-introduction/) | конспект |

## Сборка

Нужны Python 3.11 или новее и mdBook 0.5.4 с плагинами mdbook-mermaid 0.17.1 и mdbook-katex 0.10.0.

```bash
python3 scripts/make_summary.py books/embodied-ai   # обновить оглавление книги
python3 scripts/check_links.py books/embodied-ai/src # проверить относительные ссылки
python3 scripts/build_site.py                        # собрать сайт в site/
```

Сборка в Docker описана в [docker/README.md](docker/README.md).

## Документы

- [LICENSE](LICENSE) — условия использования (CC BY 4.0).
- [NOTICE.md](NOTICE.md) — авторство и источник.
- [PROVENANCE.md](PROVENANCE.md) — происхождение материалов: что переведено и что исключено.
- [CONTRIBUTING.md](CONTRIBUTING.md) — как вносить правки.

## Благодарности

Исходный материал создан Datawhale и соавторами (проект every-embodied). Перевод выполнил Сергей Монин.
