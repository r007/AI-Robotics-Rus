# Источники ресурсов и границы открытости

Собственный код и тексты этой главы распространяются по лицензии CC BY 4.0 корневого репозитория. Сторонние модели, ресурсы роботов, тембры и шрифты подчиняются своим лицензиям.

| Ресурс | Источник и способ использования |
| --- | --- |
| R1 Pro | [GalaxeaManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim/tree/abe7f5161eeaa150e6eaffdf443af5df7f23f356); в пакете воспроизведения сохранены LICENSE и NOTICE по Apache-2.0 |
| Shadow Hand | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand), уведомление Apache-2.0 сохранено в пакете воспроизведения |
| Расширение G1 | [фиксированный коммит источника](https://github.com/google-deepmind/mujoco_menagerie/tree/4d038b3feae26ec82b46a4d586379114012a8ac7/unitree_g1), сохранена лицензия Unitree BSD-3-Clause |
| Фортепианная среда | [RoboPianist](https://github.com/google-research/robopianist), устанавливается по requirements-sim.txt |
| G0.5 | [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) и [страница выпуска модели](https://huggingface.co/OpenGalaxea/G05); пользователь скачивает их самостоятельно по условиям вышестоящего проекта, веса в репозиторий не входят |
| Расширение PianoMime | [PianoMime](https://github.com/sNiper-Qian/pianomime); глава даёт только точку адаптации, веса и данные берутся из вышестоящего проекта |
| Тембр фортепиано | TimGM6mb.sf2, используемый RoboPianist; получается по инструкциям исходного проекта и не дублируется в этой главе |

## Прилагаемые ресурсы

- `assets/galaxea-replay.zip` — вводное воспроизведение двуручной игры R1 Pro: сцена, сетки моделей, начальное состояние, управление приводами, план и отчёт.
- `assets/whole-robot-replay.zip` — расширенное воспроизведение G1; пример, использованный и для существующей проверки физического воспроизведения.
- `assets/relocation-before.png`, `relocation-after.png`, `relocation-recovery.mp4` — кадры эксперимента постановки после перемещения на 6 сантиметров; видео без звука.
- `assets/relocation-static-results.json` — исторический сравнительный результат статической постановки.
- `assets/relocation-session-report.json` — исторический результат непрерывного перемещения с продолжением игры.
- `assets/public-relocation-demo.mp4`, `public-relocation-final.png`, `public-relocation-report.json` — видео, скриншот и полный результат текущего публичного упражнения.
- `assets/public-physical-report.json` — отчёт игры упражнения при фиксированном положении инструмента.
- `assets/piano-xiaohongshu.mp4`, `piano-relocation-performance.mp4` — смонтированные видео для публикации: музыкальная комната и продолжение игры после сдвига инструмента, со звуком от фактических событий нажатия клавиш. Атрибуция, монтаж и описание сжатия — в [VIDEO_CREDITS.md](VIDEO_CREDITS.md).
- `relocation-exercise.json` — публичное учебное упражнение с аккордами и гаммами этой главы.

Изменения в симуляции робота: фиксированное основание, идеальная компенсация гравитации, адаптация установки двух Shadow Hand, управление контактом, демпфирование клавиш и освещение. Лицензии пакета воспроизведения лежат в `scene/LICENSE-*.txt`, уведомления об источниках — в `scene/NOTICE*.txt`.

В главе публикуются два уже смонтированных видео исполнения, но не записи исходных песен, полные ноты песен, исходные текстуры Isaac Sim, сторонние видеоматериалы и локальные ключи. Код учебника можно использовать с собственными лицензированными материалами; музыка, изображения, модели и шрифты в видео подчиняются своим правам и указаниям об атрибуции, а не лицензии этого репозитория.

## Навигация по скриптам

| Категория | Точки входа |
| --- | --- |
| Основной процесс главы | polyphonic_piano.py, g05_relocation_session.py |
| Вывод модели и постановка рук | g05_piano_client.py, g05_piano_worker.py, g05_recovery.py |
| Робот и воспроизведение | prepare_galaxea.py, galaxea_piano.py, replay_windows.py |
| Обработка нот | midi_score.py, bimanual_fingering.py, simplify_performance.py |
| Проверка контактов и звука | audit_polyphonic_contacts.py, piano_events.py |
| Необязательные расширения | physical_piano.py, whole_robot_piano.py, pianomime_runner.py |
| Проверки регрессии | test_*.py |

Каталог `runs/` содержит выходные данные каждой машины; он игнорируется Git. Для каждого запуска используйте новый каталог и сохраняйте результаты в нём.
