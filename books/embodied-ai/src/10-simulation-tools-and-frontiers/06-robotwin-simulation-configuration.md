https://github.com/TianxingChen/RoboTwin/tree/main?tab=readme-ov-file

Следуя readme, сначала настраиваем окружение; я использовал уже имеющееся окружение dl.

Затем смотрим конфигурацию по readme.

![](https://icndr2yneehy.feishu.cn/space/api/box/stream/download/asynccode/?code=NGE2ZTgyZDQ3MjI3ZDBhN2FlN2ZhNDJiOTA1NzY3ZjdfMXZrNDNDaGxmZEp2NzMydkVmTjNvV1lXck1CSllCRnBfVG9rZW46TjU1Q2JOME1Jb3N5cm54ZDBsYmNYMURObkVwXzE3NTA4MjI0NDY6MTc1MDgyNjA0Nl9WNA)

Для лучшей визуализации возьмём значения 20 и true и посмотрим на результат.

В этой конфигурации `fovy` означает **вертикальный угол обзора (Vertical Field of View)**.

Это диапазон углов сцены, который камера способна захватить по вертикали, обычно в градусах (°).

- Для камеры `L515` значение `fovy: 45` означает, что её вертикальный угол обзора составляет 45 градусов.

- Для камеры `D435` значение `fovy: 37` означает, что её вертикальный угол обзора составляет 37 градусов.

Этот параметр связан с фокусным расстоянием камеры и размером сенсора и определяет, насколько широко камера видит «вверх и вниз». Ему соответствует горизонтальный угол обзора (fovx или hfov), то есть «влево и вправо».

Но при evaluation оказалось, что эта штука даже молоток поднять не может, смешно.

![](https://icndr2yneehy.feishu.cn/space/api/box/stream/download/asynccode/?code=MzIwNjQzMGFlY2IzYjdjMmJiNzhiYWQ4NjdkYTIxMTlfMzZVNVlGRng2TTZLU05oZXY1RjlSVUFPcnN6SE9oRTZfVG9rZW46Q05ubGJoSWtob0dZSXh4V0ZUdWMyQXg4bmllXzE3NTA4MjI0NDY6MTc1MDgyNjA0Nl9WNA)

Сначала захватывает клешню, затем кружит по соседству — это показывает, что обучение с подкреплением на agile.x всё ещё имеет проблемы.
