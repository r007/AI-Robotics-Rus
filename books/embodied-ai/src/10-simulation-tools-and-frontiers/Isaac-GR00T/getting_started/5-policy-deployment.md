## Развёртывание политики

> Для этого руководства нужны обученная контрольная точка модели и физический робот So100 (LeRobot), чтобы запустить политику.

В этом руководстве показаны примерные скрипты и фрагменты кода для развёртывания обученной политики. В качестве примера используется манипулятор So100 (LeRobot).

![alt text](../media/so100_eval_demo.gif)

### 1. Загрузка политики

Чтобы запустить сервер политики, выполните команду:

```bash
python scripts/inference_service.py --server \
    --model_path <PATH_TO_YOUR_CHECKPOINT> \
    --embodiment_tag new_embodiment \
    --data_config so100 \
    --denoising_steps 4
```

 - model_path — путь к контрольной точке, которая используется политикой; укажите путь к дообученной контрольной точке.
 - denoising_steps — число шагов шумоподавления; мы заметили, что 4 шага дают результат, сравнимый с 16.
 - embodiment_tag — метка воплощения; при дообучении на новом роботе используйте new_embodiment.
 - data_config — конфигурация данных; для so100 используйте `so100`. Если нужен другой робот, реализуйте собственные `ModalityConfig` и `TransformConfig`.

### 2. Клиентский узел

Для развёртывания дообученной модели можно использовать скрипт `scripts/inference_policy.py`. Он запускает сервер политики.

Клиентский узел можно реализовать с помощью класса `ExternalRobotInferenceClient` из `gr00t.eval.service`. Это самостоятельный класс клиент-сервер для связи с сервером политики; единственный интерфейс — конечная точка `get_action()`.

```python
from gr00t.eval.service import ExternalRobotInferenceClient
from typing import Dict, Any

raw_obs_dict: Dict[str, Any] = {} # Заполните пропуск

policy = ExternalRobotInferenceClient(host="localhost", port=5555)
raw_action_chunk: Dict[str, Any] = policy.get_action(raw_obs_dict)
```

Пользователь может скопировать этот класс и реализовать собственный клиентский узел в отдельном изолированном окружении.

### Пример с манипулятором So100 (LeRobot)

Для манипулятора So100 (LeRobot) есть пример реализации клиентского узла. Подробности — в скрипте `scripts/eval_gr00t_so100.py`.


Клиентский узел запускается такой командой:
```bash
python examples/eval_gr00t_so100.py \
 --use_policy --host <YOUR_POLICY_SERVER_HOST> \
 --port <YOUR_POLICY_SERVER_PORT> \
 --camera_index <YOUR_CAMERA_INDEX>
```

Эта команда включает робота и вызывает конечную точку сервера политики `action = get_action(obs)`, получает действие и выполняет его на роботе.
