# Обучение на демонстрациях

ManiSkill поддерживает различные методы обучения на демонстрациях и имитационного обучения через единый API и предоставляет несколько готовых, проверенных базовых решений для использования и сравнения. На этих страницах описано, как [подготовить наборы данных и среды для обучения на демонстрациях](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) и как [пользоваться базовыми решениями](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/baselines.html). Все результаты базовых решений опубликованы на нашей [открытой странице wandb](https://wandb.ai/stonet2000/ManiSkill). На ней можно фильтровать результаты по алгоритму, типу среды и т. д. Эксперименты ещё продолжаются, поэтому загружены не все результаты.

# Настройка

На этой странице собраны ключевые сведения, которые нужно знать при обучении на данных демонстраций в средах ManiSkill:

- как [скачать и воспроизвести траектории в стандартных наборах данных](#downloading-and-replaying-trajectories--standard-datasets), которые используются для сравнения имитационного обучения на основе состояния и на основе зрения;
- как [честно и правильно оценивать обученные модели](#evaluation);
- несколько распространённых [замечаний, которых стоит избегать](#common-pitfalls-to-avoid).

## Скачивание и воспроизведение траекторий / стандартные наборы данных

По умолчанию, чтобы загрузка была быстрой, а файлы — небольшими, демонстрации ManiSkill хранятся в сильно упрощённом и сжатом формате, который не содержит никаких наблюдений. Чтобы скачать исходные минимальные демонстрации, выполните:

```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

Чтобы у всех был один и тот же предобработанный и воспроизведённый набор данных, обязательно запустите скрипт: [ManiSkill/scripts/data_generation/replay_for_il_baselines.sh](https://github.com/haosulab/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh). Учтите, что некоторые скрипты воспроизводят траектории с GPU-симуляцией (флаг `-b physx_cuda`), а это может требовать больше видеопамяти, чем есть у вас. Если нужно, уменьшите число параллельных сред воспроизведения, задав меньшее значение параметра `--num-procs`.

Скрипт фиксирует настройки воспроизведения траекторий, чтобы генерировать наблюдения и задавать нужное пространство действий и контроллер для всех задач бенчмарка. Все результаты обучения, подробно записанные в [проекте Wandb](https://wandb.ai/stonet2000/ManiSkill), получены на данных, воспроизведённых этим скриптом.

Если вам нужны более сложные варианты воспроизведения (например, генерация облаков точек или изменение режима контроллера), см. [документацию по воспроизведению траекторий](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html). Если вы хотите сгенерировать исходные наборы данных локально, все скрипты генерации данных хранятся в папке [data_generation](https://github.com/haosulab/ManiSkill/tree/main/scripts/data_generation).

## Оценка

Из-за разных типов сред, алгоритмов и способов оценки ниже описан единообразный и стандартизированный способ честно оценивать стратегии обучения на демонстрациях в ManiSkill. Вкратце, для честной оценки нужны такие настройки:

- Отключить частичные сбросы: среда не сбрасывается при успехе, неудаче или завершении (`ignore_terminations=True`). Вместо этого записываются несколько видов метрик успеха и неудачи.
- Все параллельные среды переконфигурируются при сбросе (`reconfiguration_freq=1`), и если в задаче есть рандомизация объектов, то рандомизируется и геометрия объектов.

Ниже приведены примеры кода, как честно оценивать политику в ManiSkill и записывать стандартные метрики. Есть варианты для CPU и GPU векторизованных сред, поскольку обычно имеет смысл оценивать политику на том же бэкенде симуляции, на котором собраны демонстрации.

### Пример кода оценки в векторизованной GPU-среде

```python
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
env_id = "PushCube-v1"
num_eval_envs = 64
env_kwargs = dict(obs_mode="state")  # Здесь можно изменить env_kwargs
eval_envs = gym.make(env_id, num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
# Другие обёртки добавляются здесь
eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=True, record_metrics=True)

# Цикл оценки: метрики записываются только для полностью завершённых эпизодов
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample()  # Замените на действие вашей политики
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # Частичных сбросов нет, поэтому truncated одновременно равен True для всех сред
    if truncated.any():
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float())
for k in eval_metrics.keys():
    print(f"{k}_mean: {torch.mean(torch.stack(eval_metrics[k])).item()}")
```

### Пример кода оценки в векторизованной CPU-среде

```python
import gymnasium as gym
from mani_skill.utils.wrappers import CPUGymWrapper
env_id = "PickCube-v1"
num_eval_envs = 8
env_kwargs = dict(obs_mode="state")  # Здесь можно изменить env_kwargs

def cpu_make_env(env_id, env_kwargs=dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        # Другие обёртки добавляются здесь
        return env
    return thunk

vector_cls = gym.vector.SyncVectorEnv if num_eval_envs == 1 else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
eval_envs = vector_cls([cpu_make_env(env_id, env_kwargs) for _ in range(num_eval_envs)])

# Цикл оценки: метрики записываются только для полностью завершённых эпизодов
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample()  # Замените на действие вашей политики
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # Частичных сбросов нет, поэтому truncated одновременно равен True для всех сред
    if truncated.any():
        for final_info in info["final_info"]:
            for k, v in final_info["episode"].items():
                eval_metrics[k].append(v)
for k in eval_metrics.keys():
    print(f"{k}_mean: {np.mean(eval_metrics[k])}")
```

Ниже описаны записываемые метрики:

- **success_once**: была ли задача успешно выполнена хотя бы в один момент эпизода.
- **success_at_end**: была ли задача успешно выполнена на последнем шаге эпизода.
- **fail_once / fail_at_end**: то же, что и выше, но для неудач. Учтите, что не у всех задач есть критерии успеха и неудачи.
- **return**: суммарная награда, накопленная за эпизод.

Для обучения на демонстрациях обычно важна только одна метрика — **success_once**; именно она обычно приводится в исследованиях и работах по ManiSkill.

## Частые замечания

Обычно, если демонстрации собраны, например, в CPU-симуляции PhysX, то и политику, обученную на этих данных, нужно оценивать на том же бэкенде симуляции. Для очень точных задач (например, PushT) даже ошибка 1e-3 может дать другой результат, поэтому это особенно важно. Поэтому для всех демонстраций, воспроизведённых инструментом воспроизведения траекторий, в имени файла траектории указан используемый бэкенд симуляции.

Источник демонстраций сильно влияет на результаты обучения. Классическое клонирование поведения хорошо имитирует демонстрации, порождённые нейросетевой политикой или политикой RL, но плохо справляется с мультимодальными демонстрациями (например, с данными телеуправления человеком или с данными, созданными планированием движения). Именно для таких случаев предназначены методы вроде Diffusion Policy (DP). Если вы не уверены, все официальные наборы данных ManiSkill в JSON-файле метаданных траектории подробно описывают, как собраны данные и какого они типа.
