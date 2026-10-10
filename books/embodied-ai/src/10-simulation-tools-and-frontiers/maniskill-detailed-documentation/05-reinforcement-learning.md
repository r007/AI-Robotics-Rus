# Обучение с подкреплением

ManiSkill поддерживает различные методы обучения с подкреплением через единый API и предоставляет несколько готовых, проверенных базовых решений (baselines) для использования и сравнения. На этой странице описано, как настроить среду обучения с подкреплением и как пользоваться базовыми решениями. Все результаты базовых решений опубликованы на нашей [открытой странице wandb](https://wandb.ai/stonet2000/ManiSkill). На ней можно фильтровать результаты по алгоритму, типу среды и т. д. Эксперименты ещё продолжаются, поэтому загружены не все результаты.

# Настройка

На этой странице собраны ключевые сведения, которые нужно знать при подготовке сред ManiSkill для обучения с подкреплением:

- как превратить среду ManiSkill в среду, совместимую с API gymnasium, в [одиночном](#gym-environment-api) и [векторизованном](#gym-vectorized-environment-api) вариантах;
- как [**правильно** и честно оценивать](#evaluation) политики RL;
- [полезные обёртки](#useful-wrappers).

Среды ManiSkill создаются функцией `make` из gymnasium. По умолчанию результат — «пакетная» среда, в которой все входы и выходы пакетированы. Это не стандартный API gymnasium. Если нужны стандартные API среды gymnasium или векторизованной среды, см. следующий раздел.

```python
import mani_skill.envs
import gymnasium as gym
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env.action_space # shape (N, D)
env.observation_space # shape (N, ...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
```

## API одиночной среды Gym

Если нужен CPU-симулятор или одиночная среда, можно применить `CPUGymWrapper`. Он по сути снимает всё пакетирование и переводит всё в numpy, поэтому среда ведёт себя как обычная среда gym. Подробности об API сред gym см. в [документации](https://gymnasium.farama.org/api/env/).

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
N = 1
env = gym.make("PickCube-v1", num_envs=N)
env = CPUGymWrapper(env)
env.action_space # shape (D, )
env.observation_space # shape (...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (...), rew (float), terminated (bool), truncated (bool)
```

## API векторизованной среды Gym

Мы также используем интерфейс `VectorEnv` (он же `AsyncVectorEnv`) из gymnasium; его можно подключить одной обёрткой, поэтому алгоритмы, которые предполагают интерфейс `VectorEnv`, работают без изменений. Подробности об API векторизованных сред gym см. в [документации](https://gymnasium.farama.org/api/vector/).

```python
import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
env.action_space # shape (N, D)
env.single_action_space # shape (D, )
env.observation_space # shape (N, ...)
env.single_observation_space # shape (...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
```

При создании векторизованной среды доступны ещё два параметра. Параметр `auto_reset` определяет, нужно ли автоматически сбрасывать параллельную среду при её завершении или обрыве; это зависит от алгоритма. Параметр `ignore_terminations` определяет, сбрасывается ли среда, когда `terminated` равен True. Как и в векторизованных средах gymnasium, возможен частичный сброс, когда одни параллельные среды сбрасываются, а другие нет.

Обратите внимание: ради эффективности всё, что возвращает среда, — это пакетные тензоры torch на GPU, а не пакетные массивы numpy на CPU. Это, пожалуй, единственное различие между векторизованными средами ManiSkill и векторизованными средами gymnasium, которое стоит учитывать.

## Оценка

Учитывая разные типы сред, алгоритмов и способов оценки, ниже описан единообразный и стандартизированный способ честно оценивать политики любого типа в ManiSkill. Вкратце, для честной оценки нужны такие настройки:

- Отключить частичные сбросы: среда не сбрасывается при успехе, неудаче или завершении (`ignore_terminations=True`). Вместо этого записываются несколько видов метрик успеха и неудачи.
- Все параллельные среды переконфигурируются при сбросе (`reconfiguration_freq=1`), и если в задаче есть рандомизация объектов, то рандомизируется и геометрия объектов.

Ниже код, который честно оценивает политику в ManiSkill и записывает стандартные метрики. Для векторизованной GPU-среды рекомендуется такой код оценки по идентификатору среды:

```python
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
env_id = "PushCube-v1"
num_eval_envs = 64
env_kwargs = dict(obs_mode="state") # Здесь можно изменить env_kwargs
eval_envs = gym.make(env_id, num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
# Другие обёртки добавляются здесь
eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=True, record_metrics=True)

# Цикл оценки: метрики записываются только для полностью завершённых эпизодов
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # Замените на действие вашей политики
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    # Частичных сбросов нет, поэтому truncated одновременно равен True для всех сред
    if truncated.any():
        for k, v in info["final_info"]["episode"].items():
            eval_metrics[k].append(v.float())
for k in eval_metrics.keys():
    print(f"{k}_mean: {torch.mean(torch.stack(eval_metrics[k])).item()}")
```

Для векторизованной CPU-среды рекомендуется такой код оценки:

```python
import gymnasium as gym
from mani_skill.utils.wrappers import CPUGymWrapper
env_id = "PickCube-v1"
num_eval_envs = 8
env_kwargs = dict(obs_mode="state") # Здесь можно изменить env_kwargs
def cpu_make_env(env_id, env_kwargs = dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        # Другие обёртки добавляются здесь
        return env
    return thunk
vector_cls = gym.vector.SyncVectorEnv if num_eval_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
eval_envs = vector_cls([cpu_make_env(env_id, env_kwargs) for _ in range(num_eval_envs)])

# Цикл оценки: метрики записываются только для полностью завершённых эпизодов
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(400):
    action = eval_envs.action_space.sample() # Замените на действие вашей политики
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

- `success_once`: была ли задача успешно выполнена хотя бы в один момент эпизода.
- `success_at_end`: была ли задача успешно выполнена на последнем шаге эпизода.
- `fail_once` / `fail_at_end`: то же, что и выше, но для неудач. Учтите, что не у всех задач есть критерии успеха и неудачи.
- `return`: суммарная награда, накопленная за эпизод.

## Полезные обёртки

Практики RL часто используют обёртки, чтобы менять и дополнять среду. Они описаны в разделе об обёртках. Некоторые распространённые:

- RecordEpisode: записывает видео и траектории rollout.
- FlattenRGBDObservations: сводит наблюдения `obs_mode="rgbd"` или `obs_mode="rgb+depth"` к простому словарю, в котором остаются только объединённый тензор `rgbd` и тензор `state`.

## Частые ошибки и замечания

В старых средах и бенчмарках часто для получения изображений для агента RL использовали `env.render(mode="rgb_array")` или `env.render()`. Это неверно: изображения наблюдений возвращаются напрямую из `env.reset()` и `env.step()`, а `env.render` в ManiSkill предназначен только для визуализации и записи видео.

Для задач с роботами наблюдения обычно состоят из информации о состоянии (например, углы суставов робота) и изображений с камер. Когда `obs_mode` отличен от `state` или `state_dict` (где есть истинные позы объектов), все задачи ManiSkill намеренно исключают часть привилегированной информации о состоянии. Кроме того, изображения, которые возвращают `env.reset()` и `env.step()`, обычно получают с камер, стоящих в определённых местах, чтобы обеспечить хороший обзор, при котором задача решаема.

# Базовые решения

Мы предоставляем несколько разных базовых решений, которые учатся по награде методом онлайн-обучения с подкреплением.

В рамках этих базовых решений мы создали стандартизированные бенчмарки RL, которые охватывают широкий диапазон сложности (лёгкие для проверки, но не насыщенные) и разнообразие типов роботизированных задач, включая, помимо прочего, классическое управление, ловкие манипуляции, манипуляции на столе, мобильные манипуляции и т. д.

## Базовые решения онлайн-RL

Список реализованных и протестированных базовых решений онлайн-RL. Ссылки на результаты ведут на соответствующие страницы wandb. Фильтры и вид можно поменять в рабочей области wandb, чтобы посмотреть результаты с другими настройками (например, обучение по состоянию или по RGB). Обратите внимание, что есть также базовые решения RL с использованием демонстраций (офлайн RL, онлайн-имитационное обучение).

| Базовое решение | Код | Результаты | Статья |
| ------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------- |
| Proximal Policy Optimization (PPO) | [ссылка](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo) | [ссылка](https://api.wandb.ai/links/stonet2000/k6lz966q) | [ссылка](http://arxiv.org/abs/1707.06347) |
| Soft Actor-Critic (SAC) | [ссылка](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/sac) | WIP | [ссылка](https://arxiv.org/abs/1801.01290) |
| Temporal Difference Learning for Model Predictive Control (TD-MPC2) | WIP | WIP | [ссылка](https://arxiv.org/abs/2310.16828) |

## Стандартные бенчмарки

Стандартные бенчмарки RL в ManiSkill состоят из двух наборов: малого набора из 8 задач и большого набора из 50 задач; оба имеют варианты на основе состояния и на основе зрения. Все стандартные задачи снабжены нормированной плотной функцией награды. Рекомендуемый малый набор создан для того, чтобы исследователи без больших вычислительных ресурсов могли разумно протестировать и сравнить свои работы. Большой набор ещё разрабатывается и тестируется.

Эти задачи охватывают чрезвычайно широкий круг проблем робототехники и RL: большие размерности наблюдений и действий, большое распределение начальных состояний, манипуляции сочленёнными объектами, обобщаемые манипуляции, мобильные манипуляции, локомоцию и т. д.

**Идентификаторы сред малого набора:**

PushCube-v1, PickCube-v1, PegInsertionSide-v1, PushT-v1, HumanoidPlaceAppleInBowl-v1, AnymalC-Reach-v1, OpenCabinetDrawer-v1

## Оценка

Чтобы правильно оценивать политики RL, см. раздел «Оценка» на [странице настройки RL](#evaluation): там описано, как настроен код. Все результаты, которые приведены в ссылках выше, соблюдают один и тот же протокол оценки.
