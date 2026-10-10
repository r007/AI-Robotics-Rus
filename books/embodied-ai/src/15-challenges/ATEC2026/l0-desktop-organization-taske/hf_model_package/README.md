# Инструкция по загрузке модели на Hugging Face

Финальные защищённые веса ATEC2026 L0 Task E загружены на Hugging Face, а не в GitHub.

## Репозиторий модели

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best
```

Коммит загрузки:

```text
https://huggingface.co/Datawhale/atec2026-task-e-act-seed1-best/commit/4583bda9e3118e75987a0f91de62c155a70a2bed
```

## Файлы для загрузки

С локальной промежуточной папки загружено:

```text
$DATA_ROOT/ATEC2026/hf_task_e_act_seed1_best
```

Исходные файлы:

```text
$PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload/policy_act.pt
$PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload.zip
```

Можно также загрузить весь каталог целиком:

```text
$PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload/
```

## SHA256

```text
policy_act.pt
282614de9673dc01e229557448e380b6a02c196b5c3953cec77b2aebd2305a8e
```

SHA256 всего zip-архива — в файле:

```text
task_e_act_seed1_best_20260608_upload.zip.sha256
```

## Загрузка через веб-интерфейс

Если вы уже вошли в аккаунт Datawhale на Hugging Face в браузере, можно открыть страницу репозитория модели и обновить файлы без токена CLI.

## Загрузка через CLI

Обновление через командную строку требует токен. Пример:

```bash
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli upload Datawhale/atec2026-task-e-act-seed1-best \
  $PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload/policy_act.pt \
  policy_act.pt \
  --repo-type model

huggingface-cli upload Datawhale/atec2026-task-e-act-seed1-best \
  $PROJECT_ROOT/submissions/task_e_act_seed1_best_20260608_upload.zip \
  task_e_act_seed1_best_20260608_upload.zip \
  --repo-type model
```

Перед повторной загрузкой сверьте ссылки в README и SHA256:

```text
../README.md
```
