# AI-Robotics-Rus

Библиотека русскоязычных переводов и конспектов по воплощённому ИИ и робототехнике.

Каждая книга лежит в своей папке `books/<книга>/`. Книга описывает себя сама: настройки mdBook, метаданные для каталога, уведомление об авторстве и страницы в `src/`. Скрипт `scripts/build_site.py` объединяет все книги в один сайт mdBook.

## Книги

| Книга | Статус | Лицензия |
|---|---|---|
| [Воплощённый ИИ и робототехника](books/embodied-ai/) | перевод в работе | CC BY 4.0 |
| [Введение в робототехнику](books/robotics-introduction/) | конспект | не указана в источнике |

Страница каталога будет строиться из `meta.toml` каждой книги.

## Структура репозитория

```text
books/<книга>/          одна папка на книгу
  book.toml             настройки mdBook: название, описание, авторы, язык
  meta.toml             данные для каталога: порядок, статус, лицензия, источник
  NOTICE.md             авторство и происхождение текста
  PROVENANCE.md         что переведено и что исключено (если нужно)
  src/                  страницы книги; SUMMARY.md — оглавление
  examples/, tools/     код, на который ссылается книга (если есть)
assets/vendor/          общие JavaScript-библиотеки сайта (mermaid)
directory/              исходник страницы каталога книг
scripts/                сборка сайта и проверки
docker/                 сборка сайта в контейнере
book.toml               корневая конфигурация объединённого сайта
LICENSE                 лицензия по умолчанию (CC BY 4.0)
```

## Сборка и проверки

Нужны Python 3.11 или новее и mdBook 0.5.4 с плагинами mdbook-mermaid 0.17.1 и mdbook-katex 0.10.0.

```bash
python3 scripts/check_books.py                       # структура книг и метаданные
python3 scripts/make_summary.py books/embodied-ai    # обновить оглавление книги, если summary = "generated"
python3 scripts/check_links.py books/embodied-ai/src # проверить относительные ссылки
python3 scripts/build_site.py                        # собрать сайт в site/
```

Сборка в Docker описана в [docker/README.md](docker/README.md).

## Лицензии и авторство

Материалы репозитория публикуются по лицензии [CC BY 4.0](LICENSE), если в `meta.toml` книги не указана другая. Авторство и условия каждой книги описаны в её `NOTICE.md`.

- [embodied-ai/NOTICE.md](books/embodied-ai/NOTICE.md) и [PROVENANCE.md](books/embodied-ai/PROVENANCE.md) — происхождение и исключения.
- [robotics-introduction/NOTICE.md](books/robotics-introduction/NOTICE.md) — лицензия оригинала не указана.
- [CONTRIBUTING.md](CONTRIBUTING.md) — как вносить правки и добавлять новую книгу.

## Благодарности

Исходный материал «Воплощённого ИИ и робототехники» создан Datawhale и соавторами (проект every-embodied). Перевод выполнил Сергей Монин.
