# Учебный проект RDK X5 Magicbox: от нуля до демо

Этот каталог — практический учебник на русском языке для **RDK X5 Magicbox** (платформы Horizon/D-Robotics). Цель — провести читателя от распаковки, прошивки и подключения к сети до базовых периферийных устройств, демонстраций компьютерного зрения, голосовой цепочки с LLM и понимания того, как подключить **OpenClaw** к возможностям платы.

1. Распаковка и подготовка системы
2. Быстрый старт и базовая периферия
3. Демонстрации компьютерного зрения (бинокулярная глубина, жесты)
4. Локальная голосовая цепочка с LLM
5. Развёртывание OpenClaw на Magicbox и подключение
6. FAQ, известные проблемы и поиск неисправностей

## 1. Общее оглавление

- [01 Подготовка окружения и прошивка](./01-environment-preparation-and-flashing.md)
- [02 Подключение по проводной сети и вход](./02-cable-network-port-connection-and-login.md)
- [03 Быстрый старт и базовая периферия](./03-quick-start-and-basic-peripherals.md)
- [03 Быстрый старт и базовая периферия: дополнительные материалы](./03-quick-start-and-basic-peripherals-supplement-materials.md)
- [04 Демо алгоритмов: бинокулярное зрение и жесты](./04-algorithm-demo-stereo-vision-and-gesture.md)
- [05 Демо алгоритмов: голос и LLM](./05-algorithm-demo-voice-and-llm.md)
- [06 Развёртывание OpenClaw и подключение](./06-openclaw-deployment-and-access.md)
- [07 Поиск неисправностей и рекомендации по публикации](./07-troubleshooting-and-release-recommendations.md)
- [08 Развёртывание OpenClaw и подключение: резервная копия](./08-openclaw-deployment-and-access-backup.md)
- [08 Выполнение команд по SSH и вставка целым блоком](./08-ssh-command-execution-and-full-paste-instructions.md)
- [09 Точка управления через Feishu](./09-feishu-control-entry.md)
- [10 Голосовой мост OpenClaw на Windows-хосте и помощник по домашним запасам](./10-windows-openclaw-voice-bridge-and-household-inventory-assistant.md)

## 2. Справочные реализации

В учебнике многократно упоминаются возможности официальных и открытых реализаций. В публичном каталоге не размещены полные репозитории зависимостей — читатель сам получает или устанавливает нужные компоненты по инструкциям из соответствующих глав.

- `hobot_stereonet`
- `magicbox_gesture_interaction`
- `magicbox_audio_io`
- `magicbox_qwen_llm`
- `openclaw`
- `sherpa-onnx`

Модель ASR для голосовой цепочки может требовать отдельной загрузки: одного лишь клонирования репозитория недостаточно, чтобы все крупные файлы оказались на месте. Основной путь установки OpenClaw в Linux — сначала `openclaw@latest`, затем `openclaw onboard --install-daemon`.

## 3. Официальные материалы (проверены)

- Главная страница онлайн-документации: https://d-robotics.github.io/magicbox_doc/magicbox
- Разделы:
  - обзор продукта: https://d-robotics.github.io/magicbox_doc/magicbox
  - быстрый старт: https://d-robotics.github.io/magicbox_doc/quickstart
  - базовая периферия: https://d-robotics.github.io/magicbox_doc/basic-peripherals
  - загрузка ресурсов: https://d-robotics.github.io/magicbox_doc/resource-download
  - разработка алгоритмов: https://d-robotics.github.io/magicbox_doc/algorithm-development
  - FAQ: https://d-robotics.github.io/magicbox_doc/faq
  - известные проблемы: https://d-robotics.github.io/magicbox_doc/known_issues
  - журнал изменений: https://d-robotics.github.io/magicbox_doc/changelog
  - техническая поддержка: https://d-robotics.github.io/magicbox_doc/technical-support

## 4. Переработанный основной путь

Чтобы разумно встроить **OpenClaw** в учебник, текущий документ разделён на две линии возможностей:

1. **Линия собственных возможностей Magicbox**
   - прошивка, сеть, базовая периферия, бинокулярное зрение, жесты, голос, LLM
2. **Линия расширения платформы**
   - развёртывание **OpenClaw Gateway** на Magicbox
   - подключение удалённых входов WebChat / Telegram / Feishu и др.
   - точки стыковки с локальными ROS-действиями, подсветкой и голосовой цепочкой

Так не смешиваются «проверка возможностей от производителя» и «подключение через расширение платформы», и читателю проще понимать, где заканчивается заводская функция.

## 5. Что вы сможете сделать после прохождения

После прохождения и выполнения этого учебника вы должны уметь:

- прошить Magicbox, подключить его к сети и войти по SSH;
- проверить работу кнопок, подсветки, сервоприводов, бинокулярной камеры, жестов, голоса и LLM;
- понимать ожидаемые результаты каждой демонстрации, типичные сбои и базовые способы отката;
- понимать место стыковки OpenClaw Gateway с локальным ROS и сценарными действиями;
- определять, какие файлы можно публиковать, а какие модели, кеши, журналы и временные файлы в репозиторий не попадают.

## 6. Локальные материалы

- Исходный PDF учебника (китайская версия) в русское издание не входит
- Локальная копия списка образов: `assets/official_pages/rdk_x5_magicbox_images_index.html` (остальные официальные страницы даны ссылками в главах)
- Архив официальных изображений: `assets/official_images/`
- Снимки реального устройства: `assets/image-*.png`
