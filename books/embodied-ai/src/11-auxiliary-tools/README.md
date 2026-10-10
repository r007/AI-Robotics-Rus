# Вспомогательные инструменты

Этот раздел собирает вспомогательные инструменты, которые часто нужны в экспериментах с воплощённым ИИ. Это не всегда алгоритмы робототехники, но от них зависят настройка окружения, загрузка моделей, обработка данных, производительность GPU-операторов и повседневная разработка.

## Высокоскоростные вычисления и библиотеки операторов

- [FlashLib: высокоскоростная библиотека классического машинного обучения](flashlib-high-speed-machine-learning-library.md): библиотека операторов для современных GPU. Охватывает KMeans, KNN, ANN, PCA, SVD, UMAP, t-SNE, HDBSCAN, GEMM и другие алгоритмы. Подходит для больших embedding, признаков траекторий и задач поиска и кластеризации.
- [Быстрая установка flash-attn](flash-attn-quick-installation.md): подбор wheel FlashAttention под версии PyTorch, CUDA, Python и ABI, чтобы не компилировать пакет из исходников долгое время.

## Загрузка моделей и данных

- [Быстрая загрузка с Hugging Face без лишних веток файлов](hf-fast-download-avoids-downloading-unnecessary-other-branch-files-simultaneously.md)
- [Быстрая загрузка с ModelScope](modelscope-quick-download.md)
- [Библиотека Xget](xget-library.md)
- [Установка Go и загрузка содержимого из S3](install-go-language-and-install-content-downloaded-from-s3.md)

## Окружение и системные инструменты

- [conda / mamba / pip / uv и другие менеджеры пакетов](package-management-tools-such-as-conda-mamba-pip-uv)
- [Docker](docker-related)
- [Использование Git](git-related-usage)
- [Использование VS Code](vscode-usage-related)
- WSL (`wsl/`)
- WSA (`wsa/`)
- [Быстрая установка micromamba в Linux](quickly-install-micromamba-in-linux.md)
- [Установка китайского метода ввода в Ubuntu](install-chinese-input-method-for-ubuntu.md)
- [Важная настройка окружения Vulkan](vulkan-important-environment-configuration.md)

## Диагностика устройств и сети

- [Подключение JoyCon к X5](joycon-connects-to-x5.md)
- [Инструменты проверки сети](network-testing-tools.md)
- [Проблемы с доступом в интернет через виртуальную сетевую карту](related-to-internet-issues-with-virtual-network-card.md)
- [Изменение системы на удалённом сервере](resolve-changes-in-the-remote-server-content-system.md)
- [Копирование командной строки Linux в Markdown с цветами и шрифтами](copying-content-from-linux-command-line-into-markdown-with-font-color.md)
