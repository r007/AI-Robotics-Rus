# Описание ресурсов

Этот каталог хранит небольшие учебные изображения, которые используются в теме AMD ROCm. Графики и последовательности ключевых кадров создаются скриптом `../code/generate_tutorial_assets.py`; он нужен, чтобы сверять успешность и поведение rollout, приведённые в учебнике. Скриншоты в `aup_cloud_guide/` взяты из архива руководства по облачной платформе AMD / AUP.

Чтобы заново построить графики, сначала подготовьте собственный каталог результатов и выполните:

```bash
python code/generate_tutorial_assets.py --source-root "$OUTPUT_ROOT"
```

В `$OUTPUT_ROOT` должны лежать пакетные результаты оценки в формате JSONL/TSV и представительные видео rollout. Если не передавать `--source-root`, скрипт построит графики по встроенным учебным метрикам и для отсутствующих видео сделает сводное изображение с пояснением источника.

| Файл | Назначение |
| --- | --- |
| `model_status_summary.png` | Общая картина текущего статуса воспроизведения темы |
| `smolvla_red_blue_success.png` | Сравнение успешности SmolVLA при фиксированных инструкциях для красной и синей кружки |
| `act_dagger_progress_curve.png` | Изменение успешности ACT от базового варианта до коррекции DAgger |
| `smolvla_blue_failure_sequence.jpg` | Ключевые кадры неудачи SmolVLA baseline с синей кружкой |
| `smolvla_blue_success_sequence.jpg` | Ключевые кадры успеха SmolVLA с синей кружкой после взвешенной выборки |
| `act_failure_sequence.jpg` | Ключевые кадры типичной неудачи ACT с DAgger |
| `act_success_sequence.jpg` | Ключевые кадры физического успеха ACT с DAgger |
| `pi0_raw_vs_finisher_diagnostic.png` | График диагностических метрик хвостовой фазы pi_0: raw и скриптовый финишер |
| `pi0_ep2_raw_vs_finisher_side_by_side.mp4` | Видео сравнения episode 2 для pi_0: raw и гибридный вариант |
| `pi0_ep2_raw_vs_finisher_frame.png` | Ключевой кадр видео сравнения episode 2 для pi_0 |
| `pi0_ep2_raw_vs_finisher_metrics.md` | Небольшая таблица метрик сравнения episode 2 для pi_0: raw и гибрид |
| `metrics_snapshot.json` | Небольшой снимок метрик: диагностика ACT, SmolVLA и pi_0 на малой выборке, open-loop pi_0 на full20 и строгие closed-loop числа |
| `collection_dataset_snapshot.json` | Сводка измеренного набора данных на устройстве AMD: 20 эпизодов, 2621 кадр, 20 Гц |
| `training_progress_snapshot.json` | Сводка исторических точек обучения ACT, SmolVLA, pi0 и вспомогательного блока |
| `training_progress_overview.png` | Общий график исторических шагов обучения и результатов замкнутого цикла; позволяет не принимать завершённость обучения за успешность |
| `pi0_strict_input_results.json` | Итоговые строгие результаты: raw/head, фиксированная и случайная среда, ошибка seed |
| `pi0_strict_input_progress.png` | График сравнения pi0 strict-input: фиксированная сцена и случайная среда |
| `pnp_four_view_strict_success.mp4` | Реально записанное строго успешное видео с четырёх видов: Agent, Egocentric, Top, Side |
| `pnp_four_view_strict_success_sequence.jpg` | Ключевые кадры из пяти моментов четырёхвидового видео |
| `aup_cloud_guide/` | Скриншоты руководства по JupyterHub и Code Server для AUP Learning Cloud (основной вход) |
| `amd_radeon_cloud/` | Иллюстрации официального руководства AMD Radeon Cloud для разработчиков и скриншоты рабочей области AMD ROCm Embodied AI Policy Replication (резервный вход) |
