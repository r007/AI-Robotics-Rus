# Обзор демонстраций задач MicroDuck

Здесь демонстрации собраны по задачам: сначала вывод на GPU, затем вывод на BPU. Раскрывающиеся разделы содержат дополнительные версии; обучение, источники моделей и шаги реализации описаны в учебниках по задачам.

GPU вычисляет действия на рабочей станции; BPU вычисляет действия на RDK X5, а рабочая станция отвечает за симуляцию и рендеринг. На странице показаны сохранённые видео задач; порядок повторного запуска описан в [пояснении к Notebook](../../03-Notebook/README.md).

## Notebook: по одной группе на каждую задачу

| Notebook и задача | Вывод GPU | Вывод BPU |
| --- | --- | --- |
| `01` Балансировка на мяче | ![Балансировка на мяче, вывод GPU](./assets/notebook-replays/basketball_gpu.gif) | ![Балансировка на мяче, вывод BPU](./assets/notebook-replays/basketball_bpu_cached.gif) |
| `02` Физические возмущения в браузере | ![Возмущения, вывод GPU](./assets/notebook-replays/perturbation_gpu.gif) | ![Возмущения, вывод BPU](./assets/notebook-replays/perturbation_bpu_cached.gif) |
| `03` Ходьба на ходулях | ![Ходьба на ходулях, вывод GPU](./assets/notebook-replays/stilt_gpu.gif) | ![Ходьба на ходулях, вывод BPU](./assets/notebook-replays/stilt_bpu_cached.gif) |
| `04` Качание и вращение | ![Качание и вращение, вывод GPU](./assets/notebook-replays/swing_gpu.gif) | ![Качание и вращение, вывод BPU](./assets/notebook-replays/swing_bpu_cached.gif) |
| `05` Балансировка на шаре FastSAC | ![Балансировка на шаре, вывод GPU](./assets/notebook-replays/ball_balance_gpu.gif) | ![Балансировка на шаре, вывод BPU](./assets/notebook-replays/ball_balance_bpu_cached.gif) |
| `06` Подъём по лестнице | ![Подъём по лестнице и выход на стол, вывод GPU](./assets/notebook-replays/ladder_gpu.gif)<br>Поэтапный подъём, выход на стол и устойчивое стояние | ![Контакт с лестницей, вывод BPU](./assets/notebook-replays/ladder_bpu_cached.gif)<br>Контакт с перекладиной |
| `07` Подключение стратегии ходьбы | ![Ходьба, вывод GPU](./assets/notebook-replays/walking_gpu.gif) | ![Ходьба, вывод BPU](./assets/notebook-replays/walking_bpu_cached.gif) |

Первые 7 Notebook сохранили 64 выполненные ячейки кода с номерами выполнения и выводом; 8-й Notebook добавляет 12 выполненных ячеек. Всего сохранены выводы 76 ячеек кода. Состояние запуска и необязательные шаги описаны в [пояснении к Notebook](../../03-Notebook/README.md). Исходные MP4 не публикуются; на странице показаны сжатые GIF.

## LightNav-0: многосценная навигация

[Восьмой Notebook](../../03-Notebook/08-lightnav0-vision-language-navigation-gpu-multi-scene.ipynb) подключает предобученную модель навигации на GPU, читает изображение от первого лица и языковую команду. Для трёх сцен создаются отдельные видео; шаги запуска, интерфейс управления и источники модели см. в [руководстве по навигации](../../02-executable-code/lightnav-learning/README.md).

| Официальная квартира | Собственная гостиная | Собственный коридор |
| --- | --- | --- |
| ![Языковая навигация в официальной квартире](./assets/lightnav/official-apartment.gif)<br>Поиск стола слева в квартире | ![Языковая навигация в гостиной](./assets/lightnav/living-room.gif)<br>Подход к креслу в гостиной | ![Языковая навигация в коридоре](./assets/lightnav/corridor.gif)<br>Поиск цветочного горшка в коридоре |

## Подъём по лестнице и выход на стол

![MicroDuck: поэтапный подъём по лестнице, выход на стол и устойчивое стояние](./assets/notebook-replays/ladder_gpu.gif)

Робот поэтапно ступает на перекладины, затем выходит на стол и встаёт. Переключение двух стратегий, методы обучения и источники моделей описаны в [учебнике по подъёму по лестнице](../06-microduck-tread-climbing-reinforcement-learning-guide/README.md).

## Ходьба на ходулях: 25 см и 200 см

![Ходьба на ходулях 25 см](./assets/task-demos/stilts-25cm.gif)

![Ходьба на ходулях 200 см](./assets/task-demos/stilts-200cm.gif)

Попеременная опорная ходьба на разной высоте. Учебный план по форме, источники моделей и конфигурации размеров описаны в [учебнике по ходьбе на ходулях](../03-microduck-stilt-walker-reinforcement-learning-reproduction/README.md).

## Другие версии задач

Здесь собраны дополнительные движения и этапы обучения для каждой задачи.

<details>
<summary>Ходьба, обход препятствий и командная хореография</summary>

![Ходьба по ровной поверхности](../07-rdk-on-side-and-web-deployment/assets/local_videos/microduck_4096env_6000iter_walk.gif)

Команды скорости и направления:

![12-секундная хореография команд для той же стратегии ходьбы](../07-rdk-on-side-and-web-deployment/assets/local_videos/microduck_command_dance.gif)

Сцена с препятствиями:

![Ходьба с обходом препятствий](./assets/task-demos/walking-obstacle-policy.gif)
</details>

<details>
<summary>Балансировка на баскетбольном мяче</summary>

![Балансировка на баскетбольном мяче](../01-microduck-basketball-balance-reinforcement-learning/assets/preview.gif)
</details>

<details>
<summary>Физические возмущения в браузере</summary>

![Демонстрация физических возмущений в браузере](../02-mjswan-microduck-browser-physical-disturbance/assets/microduck_official.gif)

![Сжатая версия 94-секундной ручной демонстрации с перетаскиванием](./assets/task-demos/browser-manual-drag-full.gif)

Внешнюю силу задают перетаскиванием утёнка мышью; наблюдаются силы и изменения движения в симуляции.
</details>

<details>
<summary>Качание и вращение</summary>

![Качание и вращение](./assets/task-demos/swing-public-alpha050.gif)

![Демонстрация обучения качанию и вращению](./assets/task-demos/swing-local-alpha050.gif)

Движения качания и вращения.
</details>

<details>
<summary>Балансировка на шаре и обучение FastSAC</summary>

![Балансировка на шаре](./assets/task-demos/ball-balance-official.gif)

![Обучение FastSAC балансировке на шаре](./assets/task-demos/ball-balance-local-5000iter.gif)

Обучение FastSAC балансировке на сферической поверхности.
</details>

<details>
<summary>Подъём по лестнице: анализ действий и этапы обучения</summary>

Полный подъём и выход на стол показаны в демонстрации выше; ниже — эталон действий и этапы обучения V1/V2/V3. Ход реализации описан в [учебнике по подъёму по лестнице](../06-microduck-tread-climbing-reinforcement-learning-guide/README.md).

![Эталонное видео: подъём MicroDuck по лестнице](./assets/task-demos/ladder-reference.gif)

![V1: частичный подъём на локальном стенде](./assets/task-demos/ladder-v1-partial.gif)

![V2: подход стопы к перекладине](./assets/task-demos/ladder-v2-bootstrap-preview.gif)

![V3: контакт с перекладиной](./assets/task-demos/ladder-v3-model-1100.gif)

![V3: начальная поза](./assets/task-demos/ladder-v3-model-1238.gif)

![Фаза контакта с перекладиной в сборном фрагменте](./assets/task-demos/ladder-stage-contact-suite.gif)

Контакт с перекладиной, подход стопы и начальная поза.
</details>

<details>
<summary>Обзор связки навыков</summary>

![Сжатый обзор связки навыков MicroDuck](./assets/task-demos/microduck-skill-suite-overview.gif)

Монтаж нескольких задач; полные демонстрации каждой задачи — в соответствующих разделах выше.
</details>

## Материалы и воспроизведение

- Все опубликованные предпросмотры видео — в формате GIF. Первые 7 Notebook показывают видео через относительные ссылки на GIF; 8-й Notebook сохраняет небольшие изображения и графики траекторий, а после запуска новое MP4 проигрывается в ячейке.
- Исходные MP4 остаются на рабочей станции симуляции, чтобы уменьшить объём клонирования и просмотра. Окружения, модели, метрики и шаги воспроизведения каждой задачи — в README соответствующей задачи.
- Повторный запуск Notebook создаёт новые MP4 задач и отчёты об оценке.
