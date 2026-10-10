#!/usr/bin/env python3
"""Create the companion notebooks for the AMD ROCm tutorial topic."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"


METADATA = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


COMMON_SETUP = r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


def find_topic_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "assets" / "metrics_snapshot.json").exists():
            return candidate
    raise RuntimeError("Запустите Jupyter из каталога темы AMD ROCm или из подкаталога notebooks.")


TOPIC_ROOT = find_topic_root()
ASSET_DIR = TOPIC_ROOT / "assets"
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/path/to/every-embodied/mujoco_pnp"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/path/to/datasets/every_embodied"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "ckpt"))

print("TOPIC_ROOT =", TOPIC_ROOT)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT =", DATA_ROOT)
print("OUTPUT_ROOT =", OUTPUT_ROOT)
print("MODEL_ROOT =", MODEL_ROOT)
'''


DISPLAY_HELPERS = r'''
try:
    from IPython.display import Image, Markdown, display
except Exception:
    class Markdown(str):
        pass

    def display(obj):
        print(obj)

    def Image(filename=None, width=None):
        return f"[image] {filename}"


def show_asset(filename, width=960):
    path = ASSET_DIR / filename
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"Нет файла материала: {path}")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    display(Markdown("\n".join(lines)))
'''


TRAINING_HELPERS = r'''
import shlex

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("В текущем окружении нет PyYAML, сначала выполните pip install pyyaml.") from exc


def require_project_layout():
    required = [
        PROJECT_ROOT / "train_model.py",
        PROJECT_ROOT / "env_config.py",
        PROJECT_ROOT / "asset" / "example_scene_y2.xml",
        PROJECT_ROOT / "mujoco_env" / "y_env2.py",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "PROJECT_ROOT не является каталогом учебника 04mujoco, отсутствует:\n"
            + "\n".join(str(path) for path in missing)
        )
    return True


def dataset_report(dataset_root):
    dataset_root = Path(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not info_path.exists():
        raise FileNotFoundError(f"Не найдены метаданные LeRobot: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks = []
    if tasks_path.exists():
        tasks = [
            json.loads(line)["task"]
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    features = info.get("features", {})
    rows = [
        ("repo_id", info.get("repo_id", "")),
        ("episodes", info.get("total_episodes", 0)),
        ("frames", info.get("total_frames", 0)),
        ("fps", info.get("fps", "")),
        ("state shape", features.get("observation.state", {}).get("shape")),
        ("action shape", features.get("action", {}).get("shape")),
        ("tasks", " / ".join(tasks)),
    ]
    md_table(["Элемент данных", "Результат чтения"], rows)
    return info, tasks


def make_training_config(
    policy_type,
    dataset_repo_id,
    dataset_root,
    output_dir,
    steps,
    batch_size,
    chunk_size,
    n_action_steps,
    save_freq,
    seed=42,
):
    return {
        "dataset": {"repo_id": dataset_repo_id, "root": str(Path(dataset_root))},
        "policy": {
            "type": policy_type,
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "device": "cuda",
        },
        "save_checkpoint": True,
        "output_dir": str(Path(output_dir)),
        "batch_size": int(batch_size),
        "job_name": Path(output_dir).name,
        "resume": False,
        "seed": int(seed),
        "num_workers": 4,
        "steps": int(steps),
        "eval_freq": 0,
        "log_freq": max(1, min(50, int(steps))),
        "save_freq": int(save_freq),
        "use_policy_training_preset": True,
        "wandb": {
            "enable": False,
            "project": f"every_embodied_{policy_type}",
            "entity": None,
            "disable_artifact": True,
        },
    }


def write_training_config(name, config):
    config_dir = OUTPUT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("Конфигурация записана:", path)
    return path


def training_command(config_path):
    return [
        sys.executable,
        str(PROJECT_ROOT / "train_model.py"),
        "--config_path",
        str(Path(config_path)),
    ]


def run_training(config_path, enabled=False):
    require_project_layout()
    command = training_command(config_path)
    print("$", shlex.join(command))
    if not enabled:
        print("Сейчас выводится только предпросмотр команды. После проверки конфигурации переключите соответствующий флаг RUN_* в True.")
        return None
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def show_rocm_resources():
    command = ["rocm-smi", "--showuse", "--showmemuse", "--showtemp"]
    print("$", shlex.join(command))
    if shutil.which(command[0]) is None:
        print("rocm-smi не найден. Проверьте, установлен ли ROCm на этой машине.")
        return
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
'''


ROLLOUT_HELPERS = r'''
import random
import time

import numpy as np
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def find_pretrained_model(run_dir):
    run_dir = Path(run_dir)
    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    candidates = sorted((run_dir / "checkpoints").glob("*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"В {run_dir} не найден pretrained_model")
    return candidates[-1]


def image_tensor(array, size=(256, 256)):
    import torch

    image = Image.fromarray(array).convert("RGB").resize(size)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


def load_policy(policy_type, policy_path, dataset_repo_id, dataset_root, device="cuda"):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if policy_type == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy as PolicyClass
    elif policy_type == "smolvla":
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyClass
    elif policy_type == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as PolicyClass
    else:
        raise ValueError(f"Неподдерживаемый policy_type: {policy_type}")

    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = PolicyClass.from_pretrained(str(policy_path), dataset_stats=metadata.stats)
    policy.to(device)
    policy.eval()
    return policy


def strict_snapshot(
    env,
    initial_target_z,
    initial_plate_pos,
    max_target_lift,
    max_lifted_run,
    stable_place_steps,
):
    target_pos = np.asarray(env.env.get_p_body(env.obj_target), dtype=np.float64)
    plate_pos = np.asarray(env.env.get_p_body("body_obj_plate_11"), dtype=np.float64)
    target_R = np.asarray(env.env.get_R_body(env.obj_target), dtype=np.float64)
    xy_dist = float(np.linalg.norm(target_pos[:2] - plate_pos[:2]))
    upright_cos = float(target_R[2, 2])
    gripper_open = bool(float(env.env.get_qpos_joint("rh_r1")[0]) < 0.1)
    tcp_high = bool(float(env.env.get_p_body("tcp_link")[2]) > 0.9)
    legacy_success = bool(env.check_success())
    plate_xy_displacement = float(np.linalg.norm(plate_pos[:2] - initial_plate_pos[:2]))
    placement_candidate = bool(
        legacy_success
        and max_target_lift >= 0.03
        and max_lifted_run >= 3
        and upright_cos >= 0.7
        and abs(float(target_pos[2] - plate_pos[2])) < 0.08
        and plate_xy_displacement < 0.05
        and gripper_open
        and tcp_high
    )
    stable_place_steps = stable_place_steps + 1 if placement_candidate else 0
    physical_success = bool(placement_candidate and stable_place_steps >= 5)
    return {
        "legacy_success": legacy_success,
        "physical_success": physical_success,
        "xy_dist": xy_dist,
        "target_z": float(target_pos[2]),
        "plate_z": float(plate_pos[2]),
        "max_target_lift": float(max_target_lift),
        "max_lifted_run": int(max_lifted_run),
        "upright_cos": upright_cos,
        "plate_xy_displacement": plate_xy_displacement,
        "placement_candidate": placement_candidate,
        "stable_place_steps": int(stable_place_steps),
        "gripper_open": gripper_open,
        "tcp_high": tcp_high,
    }, stable_place_steps


def run_closed_loop(
    policy,
    policy_type,
    instruction,
    seeds,
    output_dir,
    device="cuda",
    max_action_steps=300,
    render=True,
):
    from mujoco_env.y_env2 import SimpleEnv2
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = PROJECT_ROOT / "asset" / "example_scene_y2.xml"
    results = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        env = SimpleEnv2(str(xml_path), action_type="joint_angle", state_type="joint_angle", seed=seed)
        env.set_instruction(instruction)
        policy.reset()

        initial_target_z = float(env.env.get_p_body(env.obj_target)[2])
        initial_plate_pos = np.asarray(
            env.env.get_p_body("body_obj_plate_11"), dtype=np.float64
        ).copy()
        max_target_lift = 0.0
        lifted_run = 0
        max_lifted_run = 0
        stable_place_steps = 0
        frames = []
        started = time.perf_counter()
        final = None

        action_step = 0
        while action_step < max_action_steps and env.env.is_viewer_alive():
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            state = env.get_joint_state()[:6]
            agent_image, wrist_image = env.grab_image()
            observation = {
                "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                "observation.image": image_tensor(agent_image).to(device).unsqueeze(0),
                "observation.wrist_image": image_tensor(wrist_image).to(device).unsqueeze(0),
                "task": [instruction],
            }
            with torch.inference_mode():
                action = policy.select_action(observation)[0, :7].detach().cpu().numpy()
            action[6] = np.clip(action[6], 0.0, 1.0)
            env.step(action.astype(np.float32))

            target_z = float(env.env.get_p_body(env.obj_target)[2])
            lift = target_z - initial_target_z
            max_target_lift = max(max_target_lift, lift)
            lifted_run = lifted_run + 1 if lift >= 0.03 else 0
            max_lifted_run = max(max_lifted_run, lifted_run)
            final, stable_place_steps = strict_snapshot(
                env,
                initial_target_z,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
                stable_place_steps,
            )

            if action_step % 2 == 0:
                frames.append(np.asarray(agent_image))
            if render:
                env.render()
            action_step += 1
            if final["physical_success"]:
                break

        video_path = output_dir / f"{policy_type}_seed{seed}.mp4"
        video_saved = False
        if frames and imageio is not None:
            try:
                imageio.mimsave(video_path, frames, fps=10, quality=8)
                video_saved = True
            except Exception as exc:
                print("Не удалось сохранить видео:", repr(exc))
        elif frames:
            print("imageio не установлен, сохранение видео пропущено. Выполните pip install imageio imageio-ffmpeg и запустите снова.")
        row = {
            "policy_type": policy_type,
            "seed": int(seed),
            "instruction": instruction,
            "action_steps": int(action_step),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "video": str(video_path) if video_saved else None,
            **(final or {}),
        }
        print(json.dumps(row, ensure_ascii=False))
        results.append(row)
        env.env.close_viewer()

    result_path = output_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    physical = sum(row.get("physical_success", False) for row in results)
    print(f"physical_success = {physical}/{len(results)}")
    print("Файл результатов:", result_path)
    return results
'''


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n") + "\n",
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


NOTEBOOKS: dict[str, list[dict]] = {}


def make_training_notebook(
    filename: str,
    number: str,
    policy_type: str,
    title: str,
    summary: str,
    full_steps: int,
    batch_size: int,
    chunk_size: int,
    n_action_steps: int,
    model_note: str,
) -> None:
    gate_cells: list[dict] = []
    if policy_type == "pi0":
        gate_cells = [
            md("## Checkpoint 2: проверка доступа Hugging Face (gated)"),
            code(
                r'''
token_present = bool(os.environ.get("HF_TOKEN"))
print("HF_TOKEN передан в текущий процесс:", token_present)
if not token_present:
    print("Задайте HF_TOKEN в терминале как секрет и заново запустите ядро Jupyter. Не записывайте token в Notebook.")
'''
            ),
            md("Здесь проверяется только наличие переменной окружения, token не выводится. Для обучения pi_0 нужен доступ аккаунта к gated-весам, например PaliGemma."),
        ]

    config_cell = f'''
MODEL_TYPE = {policy_type!r}
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
TRAIN_DATA_ROOT = Path(
    os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()

RUN_SMOKE = False
RUN_FULL_TRAIN = False

SMOKE_OUTPUT = MODEL_ROOT / f"{{MODEL_TYPE}}_rocm_smoke"
FULL_OUTPUT = MODEL_ROOT / f"{{MODEL_TYPE}}_rocm_full"

smoke_config = make_training_config(
    policy_type=MODEL_TYPE,
    dataset_repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    output_dir=SMOKE_OUTPUT,
    steps=2,
    batch_size=min(4, {batch_size}),
    chunk_size={chunk_size},
    n_action_steps={n_action_steps},
    save_freq=2,
)
full_config = make_training_config(
    policy_type=MODEL_TYPE,
    dataset_repo_id=DATASET_REPO_ID,
    dataset_root=TRAIN_DATA_ROOT,
    output_dir=FULL_OUTPUT,
    steps={full_steps},
    batch_size={batch_size},
    chunk_size={chunk_size},
    n_action_steps={n_action_steps},
    save_freq=max(1, {full_steps} // 2),
)

smoke_config_path = write_training_config(f"{{MODEL_TYPE}}_smoke", smoke_config)
full_config_path = write_training_config(f"{{MODEL_TYPE}}_full", full_config)
print("smoke output =", SMOKE_OUTPUT)
print("full output =", FULL_OUTPUT)
'''

    cells = [
        md(
            f"""
            # {number} {title}

            {summary}

            Этот раздел начинается с проверки данных LeRobot и последовательно выполняет 2-step smoke, полное обучение, наблюдение за ресурсами ROCm и поиск checkpoint. Все длительные задачи по умолчанию выключены: сначала проверьте сгенерированную конфигурацию, затем явно включите флаг запуска.
            """
        ),
        code(COMMON_SETUP),
        code(DISPLAY_HELPERS),
        code(TRAINING_HELPERS),
        md("## Checkpoint 1: проверка проекта и обучающих данных"),
        code(
            r'''
require_project_layout()
print("train_model.py =", PROJECT_ROOT / "train_model.py")
print("Обучающие данные =", Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "omy_pnp_language")))
'''
        ),
        code(
            r'''
candidate = Path(os.environ.get("TRAIN_DATA_ROOT", DATA_ROOT / "omy_pnp_language"))
if (candidate / "meta" / "info.json").exists():
    dataset_report(candidate)
else:
    print("Обучающие данные ещё не готовы:", candidate)
    print("Сначала выполните 07_data_collection_and_audit.ipynb или укажите в TRAIN_DATA_ROOT существующий набор данных LeRobot.")
'''
        ),
    ]
    cells.extend(gate_cells)
    checkpoint_number = 3 if policy_type == "pi0" else 2
    cells.extend(
        [
            md(f"## Checkpoint {checkpoint_number}: генерация конфигурации smoke и полного обучения"),
            code(config_cell),
            md(
                "Файлы конфигурации записываются в `$OUTPUT_ROOT/configs`, checkpoint — в `$MODEL_ROOT`. Эти каталоги должны находиться на диске с достаточным объёмом, а не в Git-репозитории."
            ),
            md(f"## Checkpoint {checkpoint_number + 1}: сначала 2-step smoke"),
            code(
                r'''
show_rocm_resources()
run_training(smoke_config_path, enabled=RUN_SMOKE)
'''
            ),
            md(
                "2-step smoke подтверждает только, что работают загрузка данных, построение модели, forward/backward, шаг оптимизатора и запись checkpoint; он не доказывает сходимость модели и тем более не заменяет успешность closed-loop в MuJoCo."
            ),
            md(f"## Checkpoint {checkpoint_number + 2}: запуск полного обучения"),
            code(
                r'''
run_training(full_config_path, enabled=RUN_FULL_TRAIN)
'''
            ),
            md(model_note),
            md(
                """
                Типичный порядок восстановления: при нехватке видеопамяти сначала уменьшите `batch_size`; при ошибках DataLoader worker / pickle замените `num_workers` в YAML на `0`; при ошибке `401/403` проверьте права аккаунта и секретный `HF_TOKEN`; если не записывается checkpoint, сначала проверьте диск, на котором находится `$MODEL_ROOT`. Предупреждение базы MIOpen в ROCm можно записать и продолжить наблюдение, если обучение не прерывается; но если ядро завершается, появляются NaN или процесс пропадает, длительное обучение нужно остановить и вернуться к smoke.
                """
            ),
            md(f"## Checkpoint {checkpoint_number + 3}: поиск checkpoint и проверка ресурсов"),
            code(
                r'''
show_rocm_resources()
if (FULL_OUTPUT / "checkpoints").exists():
    checkpoints = sorted((FULL_OUTPUT / "checkpoints").glob("*/pretrained_model"))
    print("Доступные checkpoint:")
    for path in checkpoints:
        print(" -", path)
else:
    print("Полное обучение ещё не запускалось:", FULL_OUTPUT)
'''
            ),
            md(
                "После обучения не выбирайте только checkpoint с минимальным loss. Следующий шаг — `11_mujoco_closed_loop_deploy.ipynb`: с фиксированным seed, строгим `physical_success` и проверкой по видео сравните промежуточные checkpoint с финальным."
            ),
        ]
    )
    NOTEBOOKS[filename] = cells


NOTEBOOKS["01_device_env_check.ipynb"] = [
    md(
        """
        # 01 Проверка устройства AMD ROCm и окружения

        Цель этого раздела — сначала показать, что устройство, окружение, кэш и цепочка доступа работают. Многие сбои обучения выглядят как проблема модели, а на деле ROCm не видит GPU, кэш лежит не на том диске, сеть не даёт загрузить веса или унифицированную память занимает другой процесс.

        В Notebook в основном делаются три вещи: проверка оборудования и PyTorch, планирование каталогов для больших файлов и составление таблицы ресурсов устройства.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: видит ли ROCm и PyTorch GPU"),
    code(
        r'''
def run_cmd(cmd):
    print("$", " ".join(cmd))
    if shutil.which(cmd[0]) is None:
        print(f"Команда не найдена: {cmd[0]}")
        return
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)


run_cmd(["rocm-smi", "--showuse", "--showtemp", "--showmemuse"])

try:
    import torch
    print("torch =", torch.__version__)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Ошибка проверки PyTorch:", repr(exc))
'''
    ),
    md("Если `rocm-smi` работает, а `torch.cuda.is_available()` возвращает `True`, базовая цепочка ROCm и PyTorch работает. Если первое в порядке, а второе — `False`, сначала проверьте, что установлена сборка PyTorch для ROCm."),
    md("## Checkpoint 2: унифицированная память, видеопамять и каталоги на диске"),
    code(
        r'''
paths = [
    ("Исходный код проекта", PROJECT_ROOT),
    ("Каталог данных", DATA_ROOT),
    ("Модели и checkpoint", MODEL_ROOT),
    ("Выход Notebook", OUTPUT_ROOT),
    ("Hugging Face cache", Path(os.environ.get("HF_HOME", "/path/to/cache/huggingface"))),
]
md_table(["Элемент", "Рекомендуемый путь", "Существует"], [(name, f"`{path}`", path.exists()) for name, path in paths])
'''
    ),
    md("Кэш больших моделей, наборы данных, checkpoint и пакетные видео могут занимать очень много места. В отчёте об эксперименте достаточно описать планировку каталогов и тип диска, эти файлы не нужно класть в каталог учебника."),
    md("## Checkpoint 3: шаблон таблицы ресурсов устройства"),
    code(
        r'''
rows = [
    ("Модель устройства", "AMD Ryzen AI MAX+ / Radeon GPU"),
    ("Версия системы", "Ubuntu / WSL / другое"),
    ("Версия ROCm", "укажите версию из rocm-smi или пакета"),
    ("Версия PyTorch", "укажите torch.__version__"),
    ("Температура в простое", "например 30-45 C"),
    ("Температура при обучении", "например 75-85 C"),
    ("VRAM при обучении ACT", "укажите значение из rocm-smi"),
    ("VRAM при обучении SmolVLA", "укажите значение из rocm-smi"),
    ("Статус smoke pi0", "не запускался / пройден / причина ошибки"),
]
md_table(["Элемент", "Запись"], rows)
'''
    ),
    md("После этого раздела должно быть понятно, готово ли это устройство AMD к дальнейшему обучению и оценке."),
]


NOTEBOOKS["02_physical_success_review.ipynb"] = [
    md(
        """
        # 02 Оценка физического успеха и проверка по видео

        Этот раздел решает ключевой вопрос: действительно ли success в логе означает, что робот взял чашку и поставил её на тарелку. Здесь старое геометрическое условие успеха и `physical_success` считаются отдельно, а успех и провал проверяются по ключевым кадрам видео.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: чтение примерных метрик этой темы"),
    code(
        r'''
snapshot_path = ASSET_DIR / "metrics_snapshot.json"
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
print(json.dumps(snapshot["best_summary"], ensure_ascii=False, indent=2))
'''
    ),
    md("## Checkpoint 2: что такое `physical_success`"),
    code(
        r'''
def physical_success(row, min_lift=0.03, min_lift_steps=3, upright_threshold=0.7):
    """Пример логики оценки. В реальном проекте ориентируйтесь на поля, которые записывает скрипт eval."""
    legacy = bool(row.get("legacy_success", row.get("success", False)))
    max_lift = float(row.get("max_target_lift", row.get("max_mug_lift", 0.0)))
    lifted_steps = int(row.get("lifted_steps", 0))
    upright = float(row.get("final_target_upright_cos", row.get("final_mug_upright_cos", 1.0)))
    return legacy and max_lift >= min_lift and lifted_steps >= min_lift_steps and upright >= upright_threshold


examples = [
    {"name": "геометрически успешно, но чашка опрокинута", "success": True, "max_mug_lift": 0.08, "lifted_steps": 80, "final_mug_upright_cos": 0.2},
    {"name": "толкнули к краю тарелки, но не схватили", "success": True, "max_mug_lift": 0.005, "lifted_steps": 0, "final_mug_upright_cos": 0.99},
    {"name": "настоящий захват и установка в вертикальном положении", "success": True, "max_mug_lift": 0.09, "lifted_steps": 120, "final_mug_upright_cos": 0.96},
]
md_table(["Пример", "physical_success"], [(e["name"], physical_success(e)) for e in examples])
'''
    ),
    md("Старый success можно использовать как вспомогательный показатель, но он не заменяет проверку по видео и по физическому состоянию. Особенно в задаче захвата чашки толчки, сдавливание и опрокидывание могут сделать конечную геометрию похожей на успех."),
    md("## Checkpoint 3: проверка ключевых кадров успеха и провала"),
    code(
        r'''
show_asset("smolvla_blue_failure_sequence.jpg", width=1100)
show_asset("smolvla_blue_success_sequence.jpg", width=1100)
show_asset("act_failure_sequence.jpg", width=1100)
show_asset("act_success_sequence.jpg", width=1100)
'''
    ),
    md("При просмотре видео следите по временной оси за четырьмя вещами: касается ли целевой чашки, устойчиво ли она схвачена, перенесена ли на тарелку и вертикальна ли в конечном состоянии."),
    md("## Checkpoint 4: шаблон пакетной статистики JSONL"),
    code(
        r'''
def summarize_jsonl(path):
    path = Path(path)
    if not path.exists():
        print(f"Файл не существует: {path}")
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    legacy = sum(bool(r.get("success") or r.get("legacy_success")) for r in rows)
    physical = sum(bool(r.get("physical_success")) for r in rows)
    return {"total": total, "legacy_success": legacy, "physical_success": physical}


example_jsonl = OUTPUT_ROOT / "eval_result.jsonl"
print("Положите сюда свой JSONL и запустите:", example_jsonl)
print(summarize_jsonl(example_jsonl))
'''
    ),
]


NOTEBOOKS["03_act_dagger_diagnostics.ipynb"] = [
    md(
        """
        # 03 Перенос ACT на ROCm и диагностика DAgger

        Этот раздел на примере ACT показывает: снижение loss при обучении не означает успеха в closed-loop rollout. Сначала смотрим кривые диагностики, затем связываем воспроизведение данных, open-loop, closed-loop, видео с провалами и DAgger в одну цепочку поиска неисправностей.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: кривые диагностики ACT"),
    code(
        r'''
snapshot = json.loads((ASSET_DIR / "metrics_snapshot.json").read_text(encoding="utf-8"))
rows = []
for item in snapshot["act_progress"]:
    rows.append((item["label"], f'{item["success"]}/{item["total"]}', f'{item["rate"]:.1%}'))
md_table(["Этап", "physical_success", "Успешность"], rows)
show_asset("act_dagger_progress_curve.png", width=960)
'''
    ),
    md("Эта кривая нужна не для того, чтобы доказать идеальную генерализацию ACT, а чтобы показать направление поиска неисправностей: сначала убедиться в данных и состоянии closed-loop, и только потом решать, нужен ли DAgger."),
    md("## Checkpoint 2: шаблон команды строгой оценки ACT"),
    code(
        r'''
act_policy = MODEL_ROOT / "act_best" / "step_5000"
cmd = f"""
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
./.venv/bin/python eval_policy_success.py \\
  --policy act \\
  --act-policy-path "{act_policy}" \\
  --physical-success \\
  --seed-start 1000 \\
  --episodes 10 \\
  --max-action-steps 1200 \\
  --output-jsonl "$OUTPUT_ROOT/act_seen_1000_1009.jsonl"
"""
print(cmd)
'''
    ),
    md("Полное обучение и пакетную оценку лучше запускать в терминале или фоновом скрипте. Здесь Notebook сохраняет шаблон команды, объясняет смысл параметров и читает результаты."),
    md("## Checkpoint 3: сравнение видео успеха и провала"),
    code(
        r'''
show_asset("act_success_sequence.jpg", width=1100)
show_asset("act_failure_sequence.jpg", width=1100)
'''
    ),
    md("Провалы ACT часто случаются не из-за полной неподвижности, а из-за сбоя на одном из этапов: приближении, захвате, переносе или освобождении. При проверке видео нужно записать, на каком этапе произошёл провал."),
    md("## Checkpoint 4: запись эксперимента DAgger"),
    code(
        r'''
rows = [
    ("Длина prefix", "например 40 control tick"),
    ("Тип oracle", "scripted policy / human correction"),
    ("timestamp offset", "например, для correction episode используйте 2.0"),
    ("Вес выборки", "например, correction episode weight=0.25"),
    ("Seed оценки", "seen / mixed / heldout указывайте отдельно"),
    ("Условие замены базовой модели", "должна превзойти старый baseline по тому же physical_success"),
]
md_table(["Пункт записи", "Пример"], rows)
'''
    ),
]


NOTEBOOKS["04_smolvla_weighted_sampling.ipynb"] = [
    md(
        """
        # 04 Перенос SmolVLA на ROCm и взвешивание выборки

        Этот раздел на примере SmolVLA показывает: языковую политику нельзя оценивать только по общей успешности. Инструкции для красной и синей чашки нужно оценивать отдельно, а затем сравнить эффект копирования episode и Weighted sampler.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: результаты фиксированных инструкций для красной и синей чашки"),
    code(
        r'''
snapshot = json.loads((ASSET_DIR / "metrics_snapshot.json").read_text(encoding="utf-8"))
rows = []
for item in snapshot["smolvla_forced_instruction_n10"]:
    rows.append((item["label"], item["red_text"], item["blue_text"]))
md_table(["Стратегия", "physical_success для красной чашки", "physical_success для синей чашки"], rows)
show_asset("smolvla_red_blue_success.png", width=1100)
'''
    ),
    md("У baseline хорошо с красной чашкой и плохо с синей: модель не то чтобы совсем не умеет захватывать, а распределение, связанное с условием задачи и цветом, ещё не выучено устойчиво."),
    md("## Checkpoint 2: разница между копированием episode и Weighted sampler"),
    code(
        r'''
rows = [
    ("Копирование episode", "напрямую меняет распределение episode в наборе данных", "просто реализовать, но модель легко смещается к одному цвету"),
    ("Weighted sampler", "не меняет исходный parquet, только вероятности выборки", "проще откатить и лучше подходит для балансировки малых данных"),
    ("Оценка промежуточных checkpoint", "не смотрите только на final step", "в этом примере step500 сбалансированнее step1000"),
]
md_table(["Метод", "Как делается", "На что обратить внимание"], rows)
'''
    ),
    md("## Checkpoint 3: пересоздание графиков"),
    code(
        r'''
cmd = [
    sys.executable,
    str(TOPIC_ROOT / "code" / "generate_tutorial_assets.py"),
    "--source-root",
    str(OUTPUT_ROOT),
]
print("Если в OUTPUT_ROOT есть нужные этой главе TSV/JSONL/MP4, можно запустить:")
print(" ".join(cmd))
# subprocess.run(cmd, check=True)
'''
    ),
    md("Ячейка выше по умолчанию не выполняется, чтобы не создавать заглушки графиков, пока выходной каталог не готов. Убедитесь, что `$OUTPUT_ROOT` указан верно, и раскомментируйте последнюю строку."),
    md("## Checkpoint 4: сравнение провала и успеха после исправления для синей чашки"),
    code(
        r'''
show_asset("smolvla_blue_failure_sequence.jpg", width=1100)
show_asset("smolvla_blue_success_sequence.jpg", width=1100)
'''
    ),
]


NOTEBOOKS["05_pi0_smoke_gate.ipynb"] = [
    md(
        """
        # 05 Smoke-проверка доступа pi_0 и ограничение обучения

        Этот раздел не торопит с запуском длительного обучения, а сначала проверяет доступ к PaliGemma и Hugging Face (gated), загрузку весов, загрузку данных и 1-step smoke. Полное обучение имеет смысл только после прохождения smoke.
        """
    ),
    md(
        """
        ## Важное исправление: старая панель из 30 seed — это не генерализация по положению

        `y_env2.py` раньше всегда задавал `np.random.seed(seed=0)` независимо от переданного seed окружения. Старая панель `1010-1039` меняла только сэмплирование политики Pi0, а положения чашки и тарелки оставались фиксированными. Эти результаты всё ещё можно использовать для анализа временной структуры действий scaffold, но их нельзя считать доказательством пространственной генерализации. После исправления контроль raw/head в фиксированном окружении даёт `0/12 -> 6/12`, а gate по случайным окружениям — только `1/4`; следующим шагом нужно пересобрать данные с несколькими положениями.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: шаблон команды проверки доступа"),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"
HF_TOKEN_STDIN=1 REQUIRE_PROXY=1 ./install_hf_token_for_pi0.sh
""")
'''
    ),
    md("Эта форма команды подчёркивает две вещи: token не записывается в Notebook, а скрипт сначала проверяет `whoami`, PaliGemma и `lerobot/pi0` и только после успешной проверки всего сохраняет результат."),
    md("## Checkpoint 2: что доказывает 1-step smoke"),
    code(
        r'''
rows = [
    ("Доступ к gated-модели", "можно прочитать конфигурацию, нужную для PaliGemma / pi0"),
    ("Загрузка данных", "читаются набор данных LeRobot и ключи observation/action"),
    ("Построение policy", "policy pi0 инициализируется"),
    ("Один шаг обучения", "forward/backward/шаг оптимизатора выполняются"),
    ("Запись checkpoint", "в выходном каталоге появляется smoke checkpoint"),
]
md_table(["Проверка", "Что означает прохождение"], rows)
'''
    ),
    md("1-step smoke не доказывает сходимость политики и не означает её успешность. Это просто порог перед полным обучением."),
    md("## Checkpoint 3: шаблон команды обучения"),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"
RUN_SMOKE=1 RUN_FULL_TRAIN=1 PI0_STEPS=20000 PI0_BATCH_SIZE=4 \\
  ./run_pi0_train_eval_after_hf_ready.sh
""")
'''
    ),
    md("Полное обучение лучше запускать в терминале, tmux или фоновом скрипте. Notebook отвечает за запись параметров, объяснение ограничений и чтение summary после обучения."),
    md("## Checkpoint 4: диаграмма этапов"),
    code('show_asset("model_status_summary.png", width=900)'),
    md("## Checkpoint 5: диагностика финишера второй половины с `tcp_to_plate`"),
    md(
        """
        Сырой pi_0 пока нельзя записывать как воспроизведение успеха. Более ценная итерация диагностики — расширить state финишера второй половины с 19 до 22 измерений, добавив `tcp_to_plate = tcp_link_xyz - plate_xyz`. Это явно сообщает модели направление от центра тарелки к TCP и не даёт ей неявно угадывать цель только по изображению и phase.

        Этот эксперимент по-прежнему является диагностическим scaffold: префикс предоставляется oracle, schedule выровнен по `move_preplace`, финишер использует 22D state. Последующая проверка на 30 seed показала, что короткий schedule `move_preplace` преждевременно опускает и отпускает объект; после принудительного длинного шаблона schedule улучшенный scaffold стабильно проходит 30 seed сэмплирования политики (фактически окружение фиксировано на seed 0). Это доказывает, что узкое место найдено и устранено, но не означает, что raw pi_0 уже успешен сквозным образом.

        Следующий шаг — разбирать scaffold. Первым заменили жёсткое правило захвата: logistic-head только по phase предсказывает открытие и закрытие захвата, EEF/рука по-прежнему выводится финишером 22D. Full-state gripper head даёт точность на обучающей выборке `100%`, но в замкнутом цикле проваливается; phase-only head видит только `timestamp + phase_index_norm + phase_onehot11`, и на полной выборке seed сэмплирования политики `1010-1039` (фактически окружение фиксировано на seed 0) сохраняет strict `30/30`. Этот контроль показывает: для head по коротким событиям на малых данных входы нужно ограничивать и не подавать туда все состояния руки, которые в замкнутом цикле дрейфуют.

        Второй блок scaffold сначала заменили адаптивным gate с фиксированным порогом вместо принудительного длинного шаблона `move_preplace`: на полной выборке seed `1010-1039` (фактически окружение фиксировано на seed 0) результат всё ещё strict `30/30`, но часть прогонов опирается на страховку `max_steps=180`. Дальше обучили logistic transition head. Полная первая версия даёт точность на обучающей выборке около `99.27%`, но в замкнутом цикле seed `1010` преждевременно проваливается; вторая версия сохраняет только `tcp_to_plate_xy + local_step_norm`, точность на обучающей выборке около `95.01%`, однако на полной выборке `1010-1039` даёт strict `30/30`, legacy `30/30`. В 30 прогонах `29/30` переключений вызваны transition head, и только seed `1021` дошёл до страховки по максимальному числу шагов. Этот контроль показывает: входы phase head на малых данных тоже нужно ограничивать; стабильные геометрические признаки важнее высокой точности на обучающей выборке.

        Третий уровень — разбор примитива контакта в первой половине. Naive all-head учит все переходы по хвосту phase: метрики на обучении выглядят неплохо, но smoke `1010/1011` падает до `0/2`, потому что `descend_to_close` срабатывает слишком рано, когда TCP ещё на `0.08 m` выше пола захвата. Исправленная версия переводит `pregrasp_to_descend` в геометрическую метку, добавляет в признаки относительные величины TCP до grasp / pregrasp / floor и при развёртывании включает `descend_floor_guard`. Полная выборка `1010-1039` даёт strict `30/30`, legacy `30/30`; ключевые переключения `pregrasp->descend`, `descend->close`, `close->lift` все `30/30` срабатывают по команде head, а floor guard заблокировал `342` высоких закрытий. Это всё ещё contact scaffold, а не сквозной успех raw pi_0.

        Четвёртый уровень — убираем хвост `dataset_schedule` и переходим к финишеру `dynamic_timed`. Первый полный прогон на 30 seed дал strict только `27/30`; провальные seed `1021/1031/1036` выглядели так, будто фиксированная пауза не подходит. После разбора кода корень оказался в глобальном переключателе `--tcpplate-prefix-target-state`: префикс нуждается в `tcp_to_target`, но dynamic finisher тоже его использовал, хотя при обучении финишер должен получать `tcp_to_plate`. Когда builder состояния стал учитывать стадию, на стадии префикса используется положение относительно цели, а на стадии финишера — относительно тарелки; три провальных seed при повторной проверке дали `3/3`, а прежняя версия в том же окружении на полной выборке `1010-1039` снова дала strict `30/30`, legacy `30/30`. Но при дальнейшей трассировке выяснилось, что при непрерывной оценке остаточная динамика MuJoCo (qvel / ctrl / free-joint) от предыдущего episode может загрязнять следующий seed: у seed `1035` начальная позиция однажды оказалась вне диапазона сэмплирования. Поэтому в evaluator добавлены `--fresh-env-per-episode` и `--hard-reset-sim-data`. Итоговый чистый протокол с `--hard-reset-sim-data` на полной выборке `1010-1039` снова даёт strict `30/30`, legacy `30/30`, красная чашка `18/18`, синяя `12/12`; средний `xy=0.0219 m`, максимальный `xy=0.0450 m`, минимальный lift `0.1093 m`, минимальный upright cos `0.9504`. Прежнее значение `0.0993 m` у seed `1036` больше похоже на артефакт оценки из-за остаточного состояния reset, поэтому его больше не считают пограничным образцом.
        """
    ),
    code(
        r'''
rows = [
    ("raw full20 open-loop", "1/20 final, 3/20 ever", "raw policy всё ещё не проходит"),
    ("raw closed-loop strict", "0/20", "ближе всего к условиям развёртывания"),
    ("template-tail full20", "4/20", "фиксированный хвост спасает только часть образцов"),
    ("старая передача 19D finisher", "prefix 120/180/240/300: все 0/2", "цель второй половины всё ещё смещена к стороне чашки"),
    ("22D tcp_to_plate finisher", "strict 5/10, legacy 6/10", "state относительно тарелки — полезный сигнал, но это всё ещё scaffold"),
    ("22D + phase-scripted gripper", "strict 7/10, legacy 7/10", "тайминг захвата спасает часть провалов; оставшиеся — в основном место посадки красной чашки"),
    ("фиксированное окружение, 30 policy seeds, schedule 0..9", "strict 21/30, legacy 23/30", "все провалы связаны с коротким шаблоном move_preplace"),
    ("фиксированное окружение, 30 policy seeds, принудительный длинный schedule", "strict 30/30, legacy 30/30", "доказывает только стабильность scaffold для фиксированной сцены"),
    ("длинный schedule + phase-only learned gripper head", "strict 30/30, legacy 30/30", "первый блок scaffold заменён обучаемым head"),
    ("schedule 0..9 + adaptive move gate + learned gripper", "strict 30/30, legacy 30/30", "второй блок scaffold, инженерная версия: длинный episode0 больше не навязывается"),
    ("schedule 0..9 + xy-step transition head + learned gripper", "strict 30/30, legacy 30/30", "второй блок scaffold, обучаемая версия: 29/30 срабатываний по команде head"),
    ("policy prefix + contact transition heads + floor guard", "strict 30/30, legacy 30/30", "третий уровень без scaffold: ключевые переходы контакта срабатывают по команде head, но scaffold сохраняется"),
    ("stage-aware dynamic_timed finisher + hard-reset clean protocol", "strict 30/30, legacy 30/30", "четвёртый уровень без scaffold: prefix использует target, finisher — plate; mean xy=0.0219 m, max xy=0.0450 m"),
]
md_table(["Критерий", "Результат", "Как понимать"], rows)
'''
    ),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"

# 1. Преобразуем данные 19D phase-state в данные 22D tcp_to_plate.
python code/pi0/build_lerobot_state_phase_tcpplate.py \\
  --src-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_eef_abs_10eps_v1 \\
  --src-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_eef_abs_10eps_v1 \\
  --dst-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dst-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --overwrite

# 2. Оценка 22D finisher. Внимание: эта команда требует уже существующий 22D checkpoint.
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval/summary.json"

# 3. При проверке на большей выборке можно принудительно использовать длинный шаблон move_preplace.
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-force-schedule-episode 0 \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_long_schedule/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_long_schedule/summary.json"

# 4. Обучаем phase-only gripper head. Этот head заменяет только правило захвата и не меняет EEF/рук.
python code/pi0/train_tcpplate_gripper_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --feature-mode phase \\
  --label-source action \\
  --output "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg_summary.json"

# 5. Проверяем 30 seed сэмплирования политики с learned gripper head (окружение фактически фиксировано на seed 0).
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-force-schedule-episode 0 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_gripper_head/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_gripper_head/summary.json"

# 6. Убираем принудительный длинный episode0 и используем adaptive gate для move_preplace.
# Эта версия использует schedule 0..9 с обычным повтором; если live TCP-to-plate ещё слишком далеко,
# удерживаем move_preplace, пока xy не станет достаточно малым или не будет достигнут консервативный предел max-step.
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-adaptive-move-preplace \\
  --tcpplate-adaptive-move-preplace-xy-threshold 0.05 \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_adaptive_gate/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_adaptive_gate/summary.json"

# 7. Обучаем xy-step transition head. Full-state transition head точнее на обучающей выборке,
# но её может сбивать корреляция с высотой и направлением; стабильная версия использует только tcp_to_plate_xy + local_step_norm.
python code/pi0/train_tcpplate_transition_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --feature-mode xy_step \\
  --step-scale 180 \\
  --output "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg_summary.json"

# 8. Заменяем фиксированный пороговый gate обучаемым transition head; schedule 0..9 повторяется обычно.
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --tcpplate-transition-head-threshold 0.5 \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_transition_head/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_transition_head/summary.json"

# 9. Обучаем target-relative contact transition heads.
# naive all-head выучит «стадия скоро закончится», а не «сейчас безопасно закрыть захват»;
# здесь используются метка геометрии pregrasp и признаки геометрии grasp/floor.
python code/pi0/train_tcptarget_contact_transition_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --task-set all \\
  --early-label-mode pregrasp_geometry \\
  --feature-mode grasp_geom_count_step \\
  --step-scale 180 \\
  --output-npz "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step_summary.json"

# 10. Contact transition heads заменяют часть ручных переключений pregrasp/descend/close в первой половине.
# descend_floor_guard сохраняется, чтобы не закрыть захват раньше времени, когда TCP всё ещё заметно выше пола захвата.
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-prefix-target-state \\
  --tcpplate-contact-primitive guided_contact_lift \\
  --tcpplate-contact-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --tcpplate-contact-transition-head-threshold 0.5 \\
  --tcpplate-contact-pregrasp-hold 999 \\
  --tcpplate-contact-descend-hold 999 \\
  --tcpplate-contact-transition-head-strict \\
  --tcpplate-contact-descend-floor-guard \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --tcpplate-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --prefix-source policy \\
  --prefix-policy-path "$MODEL_ROOT/pi0_prefix_tcptarget/checkpoints/001000/pretrained_model" \\
  --prefix-dataset-repo-id datawhale_eai_pnp_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-dataset-root ./demo_data_pi0_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_contact_transition_head_unseen30/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_contact_transition_head_unseen30/summary.json"

# 11. Убираем хвост dataset_schedule и используем stage-aware dynamic_timed finisher.
# Ключевой момент: --tcpplate-prefix-target-state задаёт tcp_to_target только для prefix;
# stage-aware evaluator возвращает finisher к tcp_to_plate.
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-prefix-target-state \\
  --tcpplate-contact-primitive guided_contact_lift \\
  --tcpplate-contact-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --tcpplate-contact-transition-head-threshold 0.5 \\
  --tcpplate-contact-pregrasp-hold 999 \\
  --tcpplate-contact-descend-hold 999 \\
  --tcpplate-contact-transition-head-strict \\
  --tcpplate-contact-descend-floor-guard \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source policy \\
  --prefix-policy-path "$MODEL_ROOT/pi0_prefix_tcptarget/checkpoints/001000/pretrained_model" \\
  --prefix-dataset-repo-id datawhale_eai_pnp_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-dataset-root ./demo_data_pi0_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-steps 180 \\
  --finisher-phase-state dynamic_timed \\
  --finisher-start-phase move_preplace \\
  --timed-phase-dwell-spec move_preplace:260,lower_to_plate:40,retreat:40 \\
  --hard-reset-sim-data \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_dynamic_timed_stageaware_unseen30/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_dynamic_timed_stageaware_unseen30/summary.json"
""")
'''
    ),
    md("Чтобы запустить контроль с scripted-захватом, добавьте в команду оценки `--finisher-scripted-gripper`. Этот флаг меняет только gripper, не трогая EEF/рука, и показывает, является ли тайминг открытия и закрытия захвата главным узким местом. `--tcpplate-force-schedule-episode 0` проверяет, устраняет ли длинный шаблон `move_preplace` преждевременное освобождение. `--tcpplate-gripper-head-path` загружает learned head в хук `--finisher-scripted-gripper` и заменяет ручное правило phase обучаемой малой моделью. `--tcpplate-adaptive-move-preplace` заменяет принудительный длинный шаблон live progress gate; `--tcpplate-transition-head-path` заменяет фиксированный пороговый gate logistic transition head; `--tcpplate-contact-transition-head-path` дополнительно заменяет часть переключений phase в примитиве контакта первой половины. Для версии `dynamic_timed` особенно важен stage-aware state: prefix может использовать `tcp_to_target`, а finisher обязан использовать `tcp_to_plate`. Для пакетной оценки рекомендуется `--hard-reset-sim-data`, чтобы сначала убрать остаточную динамику MuJoCo между episode; `--fresh-env-per-episode` чище, но при частом создании окружения может возникать случайная ошибка провайдера ассетов. Текущий stage-aware dynamic finisher в чистом протоколе hard-reset на 30 seed сэмплирования политики (окружение фактически фиксировано на seed 0) даёт strict `30/30`, но всё ещё опирается на target-relative contact scaffold, floor guard, learned gripper head и финишер второй половины."),
    md("Если state normalizer старого checkpoint имеет размерность 19, а новые данные — 22, появится `mean/std` shape mismatch. В диагностическом эксперименте можно скопировать checkpoint и удалить `normalize_inputs.buffer_observation_state.mean/std`, чтобы статистики 22D набора данных инициализировались заново; такую временную обработку не нужно записывать как архитектурное новшество модели."),
]


NOTEBOOKS["06_rocm_debug_playbook.ipynb"] = [
    md(
        """
        # 06 Разбор отладки ROCm и кейсы для устранения неполадок

        Этот раздел превращает путь проб и ошибок при воспроизведении в рабочую книгу по устранению неполадок. Не нужно помнить все команды: важно разложить симптом на доказательства, корневую причину, исправление и проверку.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: шаблон записи о неполадке"),
    code(
        r'''
template = """
### Название проблемы

- Симптом:
- Команда воспроизведения:
- Ключевые доказательства:
- Что исключено:
- Корневая причина:
- Исправление:
- Проверка после исправления:
- Вывод для обучения:
"""
print(template)
'''
    ),
    md("## Checkpoint 2: индекс девяти типичных кейсов"),
    code(
        r'''
rows = [
    ("Ошибочная оценка старого success", "нужно ввести physical_success"),
    ("Проблема меток gripper в демонстрациях", "сначала проверить данные, потом подозревать модель"),
    ("ACT: loss падает, а closed-loop проваливается", "смотреть open-loop и closed-loop отдельно"),
    ("После добавления данных DAgger качество падает", "в новых данных указать, какое распределение состояний они дополняют"),
    ("Смещение SmolVLA в сторону красной или синей чашки", "оценивать отдельно по instruction"),
    ("Копирование данных вредит задаче другого типа", "сначала пробовать взвешивание на уровне sampler или loss"),
    ("Видео и статистика batch не совпадают", "видео объясняет поведение, успешность смотрим по batch JSONL"),
    ("Проблемы ресурсов при длительном обучении ROCm", "скриптовать, оценивать пакетами, проверять процессы и температуру"),
    ("Ограничение перед обучением pi0", "о длительном обучении — только после доступа и smoke"),
    ("Остатки reset у pi0", "перед непрерывной оценкой в одном окружении сначала очистить состояние динамики MuJoCo"),
]
md_table(["Кейс", "Вывод для обучения"], rows)
'''
    ),
]


NOTEBOOKS["07_data_collection_and_audit.ipynb"] = [
    md(
        """
        # 07 Сбор с нуля демонстрационных данных LeRobot в MuJoCo

        Этот раздел переводит интерактивный процесс сбора из `5.language_env.ipynb` в настраиваемую и проверяемую версию. Итог — набор данных LeRobot с изображениями двух камер, 6-мерным состоянием суставов, 7-мерным действием и языковыми инструкциями для красной и синей чашек.

        Сбор — не чисто GPU-задача: его можно выполнить на устройствах NVIDIA или AMD. Важно сохранять единообразие сцены MuJoCo, формата LeRobot, частоты управления, определений state и action и критерия успеха. Если удалённый Jupyter не может стабильно принимать события клавиатуры в окне MuJoCo, сбор можно выполнить на машине с рабочим столом, а затем синхронизировать каталог данных на обучающую машину AMD.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    code(TRAINING_HELPERS),
    md("## Checkpoint 1: настройка каталога сбора и защитных переключателей"),
    code(
        r'''
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language_local")
COLLECTION_ROOT = Path(
    os.environ.get("COLLECTION_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()
NUM_DEMOS = int(os.environ.get("NUM_DEMOS", "20"))
SEED_START = int(os.environ.get("SEED_START", "0"))
INSTRUCTIONS = [
    "Place the red mug on the plate.",
    "Place the blue mug on the plate.",
]
RUN_INTERACTIVE_COLLECTION = False
OVERWRITE_DATASET = False
RECORD_FOUR_VIEW_VIDEO = True
COLLECTION_VIDEO_DIR = Path(
    os.environ.get("COLLECTION_VIDEO_DIR", OUTPUT_ROOT / "collection_four_view")
).expanduser()

print("dataset repo_id =", DATASET_REPO_ID)
print("collection root =", COLLECTION_ROOT)
print("num demos =", NUM_DEMOS)
print("instructions =", INSTRUCTIONS)
print("four-view videos =", COLLECTION_VIDEO_DIR)
'''
    ),
    md(
        "`COLLECTION_ROOT` должен находиться на диске большой ёмкости. По умолчанию сбор не включён и существующие каталоги не удаляются; перезапись возможна только при явном `OVERWRITE_DATASET=True`. 20 episode достаточно для малого эксперимента с языковыми условиями, а для проверки генерализации по положению лучше собрать 30–50 качественных траекторий, охватив красную и синюю чашки и начальные положения."
    ),
    md("## Checkpoint 2: проверка проекта, дисплея и схемы данных"),
    code(
        r'''
require_project_layout()
print("MuJoCo scene =", PROJECT_ROOT / "asset" / "example_scene_y2.xml")
print("DISPLAY =", os.environ.get("DISPLAY"))
if not os.environ.get("DISPLAY"):
    print("DISPLAY не задан. Для интерактивного сбора нужен локальный рабочий стол, удалённый рабочий стол или доступная сессия X11.")

FEATURES = {
    "observation.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"],
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (9,),
        "names": [
            "red_x", "red_y", "red_z",
            "blue_x", "blue_y", "blue_z",
            "plate_x", "plate_y", "plate_z",
        ],
    },
}
md_table(
    ["Поле", "shape", "Назначение"],
    [
        ("observation.image", "256x256x3", "фиксированная камера"),
        ("observation.wrist_image", "256x256x3", "камера на запястье"),
        ("observation.state", "6", "текущее состояние 6 суставов"),
        ("action", "7", "цель 6 суставов на следующем шаге и захват"),
        ("obj_init", "9", "начальные xyz трёх объектов, только для воспроизведения и аудита"),
    ],
)
'''
    ),
    md("## Checkpoint 3: определение строгого критерия успеха"),
    code(
        r'''
import numpy as np


def collection_snapshot(env, initial_target_z, initial_plate_pos, max_target_lift, max_lifted_run):
    target_pos = np.asarray(env.env.get_p_body(env.obj_target), dtype=np.float64)
    plate_pos = np.asarray(env.env.get_p_body("body_obj_plate_11"), dtype=np.float64)
    target_R = np.asarray(env.env.get_R_body(env.obj_target), dtype=np.float64)
    legacy_success = bool(env.check_success())
    upright_cos = float(target_R[2, 2])
    z_gap = abs(float(target_pos[2] - plate_pos[2]))
    plate_xy_displacement = float(
        np.linalg.norm(plate_pos[:2] - np.asarray(initial_plate_pos)[:2])
    )
    placement_candidate = bool(
        legacy_success
        and max_target_lift >= 0.03
        and max_lifted_run >= 3
        and upright_cos >= 0.7
        and z_gap < 0.08
        and plate_xy_displacement < 0.05
    )
    return {
        "legacy_success": legacy_success,
        "placement_candidate": placement_candidate,
        "xy_dist": float(np.linalg.norm(target_pos[:2] - plate_pos[:2])),
        "plate_z_gap": z_gap,
        "plate_xy_displacement": plate_xy_displacement,
        "max_target_lift": float(max_target_lift),
        "max_lifted_run": int(max_lifted_run),
        "upright_cos": upright_cos,
    }
'''
    ),
    md(
        "Старый `check_success()` смотрит только на геометрию конечного состояния, поэтому срабатывает и тогда, когда чашку толкнули к тарелке. Здесь дополнительно требуется, чтобы чашка поднялась минимум на 3 см, удерживалась 3 управляющих шага подряд, в конечном состоянии стояла вертикально, а высота чашки соответствовала тарелке. Только при `physical_success=True` episode записывается."
    ),
    md("## Checkpoint 4: создание или загрузка набора данных LeRobot"),
    code(
        r'''
def create_or_load_dataset():
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    COLLECTION_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if COLLECTION_ROOT.exists() and OVERWRITE_DATASET:
        shutil.rmtree(COLLECTION_ROOT)
    if COLLECTION_ROOT.exists():
        print("Продолжаем запись в существующий набор данных:", COLLECTION_ROOT)
        return LeRobotDataset(DATASET_REPO_ID, root=COLLECTION_ROOT)
    print("Создаём новый набор данных:", COLLECTION_ROOT)
    return LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        root=COLLECTION_ROOT,
        robot_type="omy",
        fps=20,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=0,
    )
'''
    ),
    md("## Checkpoint 5: полный цикл сбора с клавиатуры"),
    code(
        r'''
def collect_demonstrations():
    import random
    import numpy as np
    from PIL import Image
    from mujoco_env.y_env2 import SimpleEnv2

    dataset = create_or_load_dataset()
    xml_path = PROJECT_ROOT / "asset" / "example_scene_y2.xml"
    COLLECTION_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError:
        cv2 = None
        if RECORD_FOUR_VIEW_VIDEO:
            print("OpenCV не установлен: данные по-прежнему собираются, но четырёхракурсные MP4 пропускаются.")

    video_writer = None
    video_tmp_path = None

    def open_episode_video(episode_id):
        nonlocal video_writer, video_tmp_path
        if not RECORD_FOUR_VIEW_VIDEO or cv2 is None:
            return
        video_tmp_path = COLLECTION_VIDEO_DIR / f"episode_{episode_id:03d}.recording.mp4"
        video_writer = cv2.VideoWriter(
            str(video_tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (640, 480),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Не удалось создать четырёхракурсное видео: {video_tmp_path}")

    def write_episode_video(env, agent_image, wrist_image):
        if video_writer is None:
            return
        views = [
            ("Agent", agent_image),
            ("Egocentric", wrist_image),
            ("Top", env.env.get_fixed_cam_rgb(cam_name="topview")),
            ("Side", env.env.get_fixed_cam_rgb(cam_name="sideview")),
        ]
        panels = []
        for label, image in views:
            panel = cv2.resize(np.asarray(image), (320, 240), interpolation=cv2.INTER_AREA)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.rectangle(panel, (0, 0), (150, 28), (20, 20, 20), thickness=-1)
            cv2.putText(
                panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            panels.append(panel)
        video_writer.write(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))

    def close_episode_video(episode_id, keep):
        nonlocal video_writer, video_tmp_path
        if video_writer is not None:
            video_writer.release()
        if video_tmp_path is not None and video_tmp_path.exists():
            if keep:
                final_path = COLLECTION_VIDEO_DIR / f"episode_{episode_id:03d}_success.mp4"
                video_tmp_path.replace(final_path)
                print("saved four-view video:", final_path)
            else:
                video_tmp_path.unlink()
        video_writer = None
        video_tmp_path = None

    def reset_episode(env, episode_id):
        seed = SEED_START + episode_id
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)
        instruction = INSTRUCTIONS[episode_id % len(INSTRUCTIONS)]
        env.set_instruction(instruction)
        initial_z = float(env.env.get_p_body(env.obj_target)[2])
        initial_plate_pos = np.asarray(
            env.env.get_p_body("body_obj_plate_11"), dtype=np.float64
        ).copy()
        print(f"episode={episode_id} seed={seed} task={instruction}")
        return initial_z, initial_plate_pos

    np.random.seed(SEED_START)
    random.seed(SEED_START)
    env = SimpleEnv2(str(xml_path), seed=SEED_START, state_type="joint_angle")
    episode_id = 0
    record_flag = False
    initial_target_z, initial_plate_pos = reset_episode(env, episode_id)
    max_target_lift = 0.0
    lifted_run = 0
    max_lifted_run = 0
    stable_place_steps = 0

    try:
        while env.env.is_viewer_alive() and episode_id < NUM_DEMOS:
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            target_z = float(env.env.get_p_body(env.obj_target)[2])
            lift = target_z - initial_target_z
            max_target_lift = max(max_target_lift, lift)
            lifted_run = lifted_run + 1 if lift >= 0.03 else 0
            max_lifted_run = max(max_lifted_run, lifted_run)
            status = collection_snapshot(
                env,
                initial_target_z,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
            )
            stable_place_steps = stable_place_steps + 1 if status["placement_candidate"] else 0
            status["stable_place_steps"] = stable_place_steps
            status["physical_success"] = bool(stable_place_steps >= 5)

            if record_flag and status["physical_success"]:
                dataset.save_episode()
                close_episode_video(episode_id, keep=True)
                print("saved:", json.dumps(status, ensure_ascii=False))
                episode_id += 1
                record_flag = False
                if episode_id >= NUM_DEMOS:
                    break
                initial_target_z, initial_plate_pos = reset_episode(env, episode_id)
                max_target_lift = 0.0
                lifted_run = 0
                max_lifted_run = 0
                stable_place_steps = 0
                continue

            teleop_delta, reset = env.teleop_robot()
            if reset:
                dataset.clear_episode_buffer()
                close_episode_video(episode_id, keep=False)
                record_flag = False
                initial_target_z, initial_plate_pos = reset_episode(env, episode_id)
                max_target_lift = 0.0
                lifted_run = 0
                max_lifted_run = 0
                stable_place_steps = 0
                continue

            if not record_flag and np.any(np.abs(teleop_delta) > 1e-8):
                record_flag = True
                open_episode_video(episode_id)
                print("Start recording")

            agent_image, wrist_image = env.grab_image()
            if record_flag:
                write_episode_video(env, agent_image, wrist_image)
            state = env.get_joint_state()[:6].astype(np.float32)
            env.step(teleop_delta)
            target_action = env.q[:7].astype(np.float32)

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": np.asarray(Image.fromarray(agent_image).resize((256, 256))),
                        "observation.wrist_image": np.asarray(Image.fromarray(wrist_image).resize((256, 256))),
                        "observation.state": state,
                        "action": target_action,
                        "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                    },
                    task=env.instruction,
                )
            env.render(teleop=True, idx=episode_id)
    finally:
        close_episode_video(episode_id, keep=False)
        env.env.close_viewer()
        shutil.rmtree(dataset.root / "images", ignore_errors=True)
    print(f"Сбор завершён: {episode_id}/{NUM_DEMOS}")
    return dataset


if RUN_INTERACTIVE_COLLECTION:
    dataset = collect_demonstrations()
else:
    print("Сбор по умолчанию выключен. После проверки DISPLAY, путей и переключателей перезаписи измените RUN_INTERACTIVE_COLLECTION на True.")
'''
    ),
    md(
        "Клавиши совпадают с исходным учебником: `W/A/S/D` — перемещение в плоскости, `R/F` — высота, `Q/E` и стрелки — ориентация, пробел — переключение захвата, `Z` — отбросить текущую неудачную попытку. После каждого успешного сохранения запись снова выключается, чтобы процесс reset не попал в следующий episode. Обучение по-прежнему использует только фиксированную камеру и камеру на запястье; виды Top/Side записываются в отдельные MP4 только для ручной проверки и не расширяют вход модели втайне."
    ),
    md("## Checkpoint 6: аудит после сбора"),
    code(
        r'''
info_path = COLLECTION_ROOT / "meta" / "info.json"
if info_path.exists():
    info, tasks = dataset_report(COLLECTION_ROOT)
    episodes_path = COLLECTION_ROOT / "meta" / "episodes.jsonl"
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_counts = {}
    for episode in episodes:
        task = episode.get("tasks", [""])[0]
        task_counts[task] = task_counts.get(task, 0) + 1
    md_table(["Инструкция", "Число episode"], sorted(task_counts.items()))
else:
    print("Данных для аудита пока нет:", COLLECTION_ROOT)
'''
    ),
    md(
        "После прохождения таблицы данных откройте upstream-файл `6.visualize_data.ipynb` и случайно воспроизведите несколько episode. Проверьте как минимум изображения, shape state и action, тайминг открытия и закрытия захвата, действительно ли объект поднимается и сбалансированы ли инструкции для красной и синей чашек. После этих проверок переходите к Notebook обучения 08–10."
    ),
]


NOTEBOOKS["07_data_collection_and_audit.ipynb"].extend(
    [
        md("## Checkpoint 7: результаты уже выполненного набора данных"),
        code(
            r'''
snapshot = json.loads(
    (ASSET_DIR / "collection_dataset_snapshot.json").read_text(encoding="utf-8")
)
rows = [
    ("repo_id", snapshot["dataset_repo_id"]),
    ("episodes", snapshot["total_episodes"]),
    ("frames", snapshot["total_frames"]),
    ("fps", snapshot["fps"]),
    ("state / action", f'{snapshot["state_shape"]} / {snapshot["action_shape"]}'),
    ("Камеры входа модели", " / ".join(snapshot["stored_cameras"])),
    ("Ракурсы проверочного видео", " / ".join(snapshot["recorder_views"])),
]
md_table(["Параметр данных", "Измерено на AMD"], rows)
print("Среднее число кадров на траекторию =", round(snapshot["total_frames"] / snapshot["total_episodes"], 1))
'''
        ),
        md("## Checkpoint 8: образцы сбора и воспроизведения с четырьмя ракурсами"),
        md(
            """
            Видео ниже — записанное на устройстве AMD воспроизведение строгой успешной политики через тот же recorder с четырьмя ракурсами, что и при сборе с клавиатуры. Оно нужно, чтобы сначала проверить, понятна ли раскладка кадра, контакт, подъём, перенос и освобождение; это не ручная демонстрация с клавиатуры. При реальном сборе установите `RUN_INTERACTIVE_COLLECTION=True`: каждая успешная демонстрация запишет MP4 с той же раскладкой в `COLLECTION_VIDEO_DIR`.

            ![Последовательность успешного захвата MuJoCo с четырьмя ракурсами](../assets/pnp_four_view_strict_success_sequence.jpg)

            <video controls width="960" src="../assets/pnp_four_view_strict_success.mp4"></video>
            """
        ),
        md("## Checkpoint 9: проверка, что seed окружения действительно меняет положения объектов"),
        code(
            r'''
source_path = PROJECT_ROOT / "mujoco_env" / "y_env2.py"
source = source_path.read_text(encoding="utf-8")
bad_pattern = "np.random.seed(seed=0)"
good_pattern = "np.random.seed(seed=seed)"
print("seed source =", source_path)
print("Найден старый вариант записи seed:", bad_pattern in source)
print("Найден исправленный вариант записи seed:", good_pattern in source)
if bad_pattern in source:
    raise RuntimeError(
        "Сейчас y_env2.py задаёт всем окружениям seed 0. Сначала исправьте это, и только потом собирайте данные с рандомизацией положения."
    )
'''
        ),
        md(
            "Эта проверка важна. Старая реализация переписывала любой переданный seed на `0`, поэтому многосидовая оценка меняла только сэмплирование политики, а положения чашки и тарелки не менялись. После исправления нужно заново собрать 30–50 траекторий с разными положениями; старые данные с фиксированной сценой ещё можно использовать для smoke-проверки цепочки, но не как обучающий набор для пространственной генерализации."
        ),
    ]
)


make_training_notebook(
    filename="08_act_training_rocm.ipynb",
    number="08",
    policy_type="act",
    title="ACT от smoke до полного обучения",
    summary="ACT — лучший базовый вариант, чтобы сначала полностью пройти замкнутый цикл. Он не опирается на языковой backbone, модель меньше, и быстрее выявляются проблемы данных, выравнивания state и action и меток захвата.",
    full_steps=5000,
    batch_size=16,
    chunk_size=10,
    n_action_steps=10,
    model_note="Пример ACT сначала использует 5 000 шагов как baseline. Одна траектория легко приводит к переобучению, поэтому для экспериментов по генерализации положения используйте несколько качественных траекторий и фиксируйте held-out seeds. Во время обучения смотрите loss, MAE действий и видеопамять, но окончательно ориентируйтесь на closed-loop physical_success.",
)


make_training_notebook(
    filename="09_smolvla_training_rocm.ipynb",
    number="09",
    policy_type="smolvla",
    title="SmolVLA от smoke до полного обучения",
    summary="SmolVLA одновременно читает изображения, язык и состояние робота. Здесь используется тот же корень данных, что и для ACT, но нужно проверить, что инструкции для красной и синей чашек присутствуют, а не смотреть только на общую успешность.",
    full_steps=20000,
    batch_size=4,
    chunk_size=5,
    n_action_steps=5,
    model_note="При первом запуске SmolVLA загружает базовые веса. Для полного обучения сохраняйте промежуточные checkpoint и отдельно считайте physical_success для красной и синей чашек; если один класс явно отстаёт, сначала проверьте распределение задач и события gripper, и только потом рассматривайте Weighted sampler.",
)


make_training_notebook(
    filename="10_pi0_training_rocm.ipynb",
    number="10",
    policy_type="pi0",
    title="pi_0 от контроля доступа до полного обучения",
    summary="Загрузка pi_0, доступ gated и нагрузка на видеопамять здесь выше. Сначала по очереди проходим доступ, данные, построение модели и 2-step обратное распространение, и только потом запускаем полное обучение.",
    full_steps=20000,
    batch_size=4,
    chunk_size=5,
    n_action_steps=5,
    model_note="Снижение loss при обучении pi_0 не гарантирует успешного захвата в замкнутом цикле. Особенно в задачах контакта на малых данных легко возникают проблемы с таймингом gripper и накоплением ошибки. Результаты raw policy, learned auxiliary head и гибрида с scaffold нужно докладывать отдельно.",
)


for notebook_name, model_name in [
    ("08_act_training_rocm.ipynb", "ACT"),
    ("09_smolvla_training_rocm.ipynb", "SmolVLA"),
    ("10_pi0_training_rocm.ipynb", "pi0"),
]:
    NOTEBOOKS[notebook_name].extend(
        [
            md("## Снимок логов завершённого обучения"),
            code(
                f'''
progress = json.loads(
    (ASSET_DIR / "training_progress_snapshot.json").read_text(encoding="utf-8")
)
records = [row for row in progress["records"] if row["model"] == {model_name!r}]
for row in records:
    ratio = row["current_step"] / max(1, row["total_steps"])
    filled = round(36 * ratio)
    bar = "#" * filled + "." * (36 - filled)
    print(f'{{row["model"]:>18}} [{{bar}}] {{row["current_step"]}}/{{row["total_steps"]}}')
    print("stage:", row["stage"])
    print("closed-loop physical_success:", row["physical_success"])
    print("note:", row["note"])
'''
            ),
            md(
                """
                ![Историческая динамика обучения AMD ROCm и результаты замкнутого цикла](../assets/training_progress_overview.png)

                Этот график получен из уже завершённых логов экспериментов на AMD, а не из повторного обучения, которое якобы выполнено при открытии Notebook. Ячейки smoke и full выше дают реальный код запуска; при длительном обучении прогресс в терминале продолжает записываться в выходной каталог, а после завершения решение о пригодности checkpoint принимается по closed-loop оценке из раздела 11.
                """
            ),
            md("## Отслеживание фоновых логов обучения в Notebook"),
            code(
                r'''
def tail_training_log(log_path, lines=25):
    log_path = Path(log_path)
    if not log_path.exists():
        print("Лог обучения ещё не существует:", log_path)
        return
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))


TRAIN_LOG = Path(os.environ.get("TRAIN_LOG", FULL_OUTPUT / "train.log"))
print("После запуска обучения эту ячейку можно выполнять повторно: tail_training_log(TRAIN_LOG)")
tail_training_log(TRAIN_LOG)
'''
            ),
        ]
    )


NOTEBOOKS["11_mujoco_closed_loop_deploy.ipynb"] = [
    md(
        """
        # 11 Развёртывание checkpoint в замкнутом цикле MuJoCo

        Этот раздел возвращает checkpoint ACT, SmolVLA или pi_0 в `SimpleEnv2`: политика с частотой 20 Гц читает изображения двух камер и 6-мерное состояние суставов и выводит 7-мерное действие. Оценка сохраняет видео и JSONL и одновременно сообщает старый геометрический успех и строгий `physical_success`.

        Это замкнутый инференс в симуляции, а не экспорт модели в сервис. После каждого действия модель заново читает наблюдение среды; если только предсказывать GT-действия на датасете, это open-loop replay, и он не заменяет здесь показатель успешности.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md(
        """
        ## Предварительный просмотр успешного выполнения без обучения: сначала поведение

        Эта ячейка не загружает модель, не запускает MuJoCo и не расходует бюджет обучения. Она показывает встроенное в репозиторий строгое успешное воспроизведение с четырьмя ракурсами, чтобы сначала проверить правильный порядок: подход, захват, подъём, перенос, освобождение и устойчивое размещение. Видео получено на фиксированном окружении с `pi0 + visual/history learned head`; это не raw pi0 и не говорит о генерализации по случайным положениям.
        """
    ),
    code(
        r'''
SUCCESS_PREVIEW = ASSET_DIR / "pnp_four_view_strict_success.mp4"
if not SUCCESS_PREVIEW.exists():
    raise FileNotFoundError(f"Не найдено видео успешного превью: {SUCCESS_PREVIEW}")

try:
    from IPython.display import Video
except ImportError:
    print("Сейчас это чистый Python-исполнитель; чтобы встроить воспроизведение видео, запустите эту ячейку в Jupyter.")
else:
    display(Video(filename=str(SUCCESS_PREVIEW), embed=True, html_attributes="controls muted"))

print("Строгое превью успеха:", SUCCESS_PREVIEW)
'''
    ),
    code(TRAINING_HELPERS),
    code(ROLLOUT_HELPERS),
    md("## Checkpoint 1: выбор модели, статистика данных и seed оценки"),
    code(
        r'''
POLICY_TYPE = os.environ.get("POLICY_TYPE", "act")  # act / smolvla / pi0
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")
EVAL_DATA_ROOT = Path(
    os.environ.get("EVAL_DATA_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()
MODEL_RUN_DIR = Path(
    os.environ.get("MODEL_RUN_DIR", MODEL_ROOT / f"{POLICY_TYPE}_rocm_full")
).expanduser()
TASK_TEXT = os.environ.get("TASK_TEXT", "Place the blue mug on the plate.")
EVAL_SEEDS = [int(value) for value in os.environ.get("EVAL_SEEDS", "1000,1001,1002,1003").split(",")]
MAX_ACTION_STEPS = int(os.environ.get("MAX_ACTION_STEPS", "300"))
RUN_CLOSED_LOOP = False

print("policy type =", POLICY_TYPE)
print("model run =", MODEL_RUN_DIR)
print("dataset =", EVAL_DATA_ROOT)
print("task =", TASK_TEXT)
print("seeds =", EVAL_SEEDS)
'''
    ),
    md(
        "Сначала сделайте smoke на 4 фиксированных seed, но 4 примера не дают статистически представительной картины. Когда модель пройдёт малую панель, расширьте её минимум до 20–30 held-out seeds и отдельно посчитайте красную и синюю чашки. При оценке не читайте координаты target и plate, фазу датасета или GT-действия как вход политики."
    ),
    md("## Checkpoint 2: проверка данных и checkpoint"),
    code(
        r'''
require_project_layout()
if (EVAL_DATA_ROOT / "meta" / "info.json").exists():
    dataset_report(EVAL_DATA_ROOT)
else:
    print("Нет статистики данных для оценки:", EVAL_DATA_ROOT)

try:
    POLICY_PATH = Path(os.environ.get("POLICY_PATH", ""))
    if not str(POLICY_PATH) or str(POLICY_PATH) == ".":
        POLICY_PATH = find_pretrained_model(MODEL_RUN_DIR)
    print("policy path =", POLICY_PATH)
except FileNotFoundError as exc:
    POLICY_PATH = None
    print(exc)
'''
    ),
    md("## Checkpoint 3: загрузка политики и запуск замкнутого цикла"),
    code(
        r'''
if RUN_CLOSED_LOOP:
    if POLICY_PATH is None:
        raise FileNotFoundError("Сначала завершите обучение или укажите pretrained_model через POLICY_PATH.")
    if POLICY_TYPE == "pi0" and not os.environ.get("HF_TOKEN"):
        print("Локальные checkpoint обычно загружаются офлайн; если всё же нужно читать gated-конфигурацию, сначала задайте HF_TOKEN как секрет.")
    show_rocm_resources()
    policy = load_policy(
        policy_type=POLICY_TYPE,
        policy_path=POLICY_PATH,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_root=EVAL_DATA_ROOT,
        device="cuda",
    )
    results = run_closed_loop(
        policy=policy,
        policy_type=POLICY_TYPE,
        instruction=TASK_TEXT,
        seeds=EVAL_SEEDS,
        output_dir=OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop",
        device="cuda",
        max_action_steps=MAX_ACTION_STEPS,
        render=True,
    )
else:
    print("Замкнутая оценка по умолчанию выключена. После проверки DISPLAY, checkpoint, статистики данных и seed переключите RUN_CLOSED_LOOP в True.")
'''
    ),
    md("## Checkpoint 4: чтение успешности и типов провалов"),
    code(
        r'''
result_path = OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop" / "results.jsonl"
if result_path.exists():
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    physical = sum(row.get("physical_success", False) for row in rows)
    legacy = sum(row.get("legacy_success", False) for row in rows)
    md_table(
        ["Критерий", "Результат"],
        [
            ("legacy_success", f"{legacy}/{len(rows)}"),
            ("physical_success", f"{physical}/{len(rows)}"),
        ],
    )
    for row in rows:
        if not row.get("physical_success"):
            print(
                "Провальные seed",
                row["seed"],
                "xy=", round(row.get("xy_dist", float("nan")), 4),
                "lift=", round(row.get("max_target_lift", float("nan")), 4),
                "upright=", round(row.get("upright_cos", float("nan")), 4),
            )
else:
    print("Результаты замкнутого цикла ещё не созданы:", result_path)
'''
    ),
    md(
        "Если `legacy_success > physical_success`, сначала смотрите видео провалов: обычно это толчки чашки, захват не был полным, объект упал во время переноса или выполнены только геометрические условия конечного состояния. Дальше вернитесь к диагностическим Notebook 02–06 и определите конкретный этап провала: approach, contact, lift, transport или release."
    ),
]


NOTEBOOKS["11_mujoco_closed_loop_deploy.ipynb"].extend(
    [
        md("## Checkpoint 5: строгое определение и результаты по четырём ракурсам"),
        code(
            r'''
strict_results = json.loads(
    (ASSET_DIR / "pi0_strict_input_results.json").read_text(encoding="utf-8")
)
predicate = strict_results["strict_predicate"]
md_table(
    ["Строгое условие", "Порог"],
    [
        ("Реальный подъём", f'>= {predicate["min_lift_m"]:.2f} м, подряд {predicate["min_lift_steps"]} шагов'),
        ("Вертикальное конечное положение", f'>= {predicate["min_upright_cos"]:.1f}'),
        ("Разница высот чашки и тарелки", f'<= {predicate["max_plate_z_gap_m"]:.2f} м'),
        ("Смещение тарелки", f'<= {predicate["max_plate_xy_displacement_m"]:.2f} м'),
        ("Устойчивое размещение", f'подряд {predicate["stable_place_steps"]} шагов'),
    ],
)
'''
        ),
        md(
            """
            ![Результаты замкнутого цикла pi0 strict-input](../assets/pi0_strict_input_progress.png)

            <video controls width="960" src="../assets/pnp_four_view_strict_success.mp4"></video>

            Видео — строгое успешное воспроизведение при env seed 0 и policy sampling seed 3. Оно относится к `pi0 + visual/history learned head`, а не к raw pi0: на вход модели подаются только изображения двух камер, язык, robot proprio и история выполненных действий, без координат чашки и тарелки, GT phase или oracle action.
            """
        ),
        md(
            "Gate случайных окружений проходит только `1/4`, поэтому в этой главе фиксированные `6/12` не записаны как успех пространственной генерализации. Следующий шаг — заново собрать многопозиционные демонстрации с исправленным seed окружения, а затем расширить оценку минимум до 20–30 действительно случайных окружений."
        ),
    ]
)


NOTEBOOKS["12_pi0_strict_input_end_to_end.ipynb"] = [
    md(
        """
        # 12 Сквозная диагностика pi0 strict-input

        Этот раздел разделяет четыре вещи, которые при отладке pi0 легче всего спутать: raw policy; learned head, использующий только допустимые для VLA входы; стабильность сэмплирования политики в фиксированной сцене и настоящая генерализация на случайные положения объектов. Все таблицы основаны на измерениях на устройстве AMD.
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1: чтение итоговых строгих результатов оценки"),
    code(
        r'''
results = json.loads(
    (ASSET_DIR / "pi0_strict_input_results.json").read_text(encoding="utf-8")
)
fixed = results["fixed_scene"]
randomized = results["randomized_environment_gate"]
rows = [
    (
        "raw pi0",
        "env seed 0; 12 policy sampling seeds",
        f'{fixed["raw_pi0"]["success"]}/{fixed["raw_pi0"]["total"]}',
    ),
    (
        "pi0 + visual/history GRU head",
        "env seed 0; та же партия из 12 policy sampling seeds",
        f'{fixed["pi0_visual_history_gru"]["success"]}/{fixed["pi0_visual_history_gru"]["total"]}',
    ),
    (
        "pi0 + visual/history GRU head",
        "случайные окружения 41–44 после исправления seed",
        f'{randomized["pi0_visual_history_gru"]["success"]}/{randomized["pi0_visual_history_gru"]["total"]}',
    ),
]
md_table(["Стратегия", "Протокол", "final strict"], rows)
'''
    ),
    md(
        """
        ![Сравнение pi0 strict-input](../assets/pi0_strict_input_progress.png)

        Рост в фиксированной сцене с `0/12` до `6/12` — существенный прогресс, но нельзя пропускать случайные окружения `1/4`. Именно поэтому учебник отдельно описывает «подгонку под фиксированную сцену» и «генерализацию по положению».
        """
    ),
    md("## Checkpoint 2: что на самом деле видит learned head"),
    code(
        r'''
rows = [
    ("Разрешено", "изображения agent/wrist, язык, 6D robot proprio, 7D команды EEF/gripper, которые предыдущая политика выполнила сама"),
    ("Запрещено", "target xyz, plate xyz, GT phase, oracle action, текущие или будущие GT-метки gripper"),
    ("Выход", "EEF keyframe / историческое действие и вероятность закрытия gripper"),
    ("Развёртывание", "порог вероятности gripper threshold=0.5, затем преобразование в бинарную команду выполнения"),
]
md_table(["Категория", "Содержание"], rows)
'''
    ),
    md(
        "У этого head входная граница совместима с VLA, но это дополнительно обученный визуальный и исторический управляющий head, поэтому в отчёте пишите `pi0 + learned head`, а не raw pi0. Он выучивает стадийные признаки из изображений и истории робота и не опирается на яркие маркеры или ручные координаты цели."
    ),
    md("## Checkpoint 3: фактическая команда обучения visual/history head"),
    code(
        r'''
import shlex

HEAD_SOURCE_REPO = os.environ.get("HEAD_SOURCE_REPO", "your_multiseed_fulltask_dataset")
HEAD_SOURCE_ROOT = Path(os.environ.get("HEAD_SOURCE_ROOT", DATA_ROOT / HEAD_SOURCE_REPO))
HEAD_SOURCE_JSONL = Path(os.environ.get("HEAD_SOURCE_JSONL", OUTPUT_ROOT / "collection_results.jsonl"))
HEAD_POLICY_PATH = Path(os.environ.get("HEAD_POLICY_PATH", MODEL_ROOT / "pi0_base" / "pretrained_model"))
HEAD_DIR = Path(os.environ.get("HEAD_DIR", MODEL_ROOT / "pi0_visual_history_head"))
FEATURE_CACHE = HEAD_DIR / "visual_features.npz"
DIRECT_HEAD = HEAD_DIR / "visual_keyframe_head.npz"
GRU_HEAD = HEAD_DIR / "visual_keyframe_gru_head.npz"
RUN_AUXILIARY_TRAIN = False

commands = [
    [
        sys.executable,
        str(TOPIC_ROOT / "code" / "pi0" / "train_pi0_visual_contact_head.py"),
        "--source", HEAD_SOURCE_REPO, str(HEAD_SOURCE_ROOT), str(HEAD_SOURCE_JSONL),
        "--policy-path", str(HEAD_POLICY_PATH),
        "--stats-repo-id", HEAD_SOURCE_REPO,
        "--stats-dataset-root", str(HEAD_SOURCE_ROOT),
        "--train-seeds", "21-32",
        "--val-seeds", "33-36",
        "--append-prev-action",
        "--feature-cache", str(FEATURE_CACHE),
        "--output", str(DIRECT_HEAD),
        "--summary-json", str(HEAD_DIR / "visual_keyframe_head_summary.json"),
    ],
    [
        sys.executable,
        str(TOPIC_ROOT / "code" / "pi0" / "train_pi0_visual_gripper_gru.py"),
        "--source", HEAD_SOURCE_REPO, str(HEAD_SOURCE_ROOT), str(HEAD_SOURCE_JSONL),
        "--feature-cache", str(FEATURE_CACHE),
        "--direct-head", str(DIRECT_HEAD),
        "--train-seeds", "21-32",
        "--val-seeds", "33-36",
        "--epochs", "300",
        "--transition-weight", "20",
        "--release-weight", "40",
        "--output", str(GRU_HEAD),
        "--summary-json", str(HEAD_DIR / "visual_keyframe_gru_summary.json"),
    ],
    [
        sys.executable,
        str(TOPIC_ROOT / "code" / "pi0" / "test_pi0_visual_gripper_gru_parity.py"),
        "--head", str(GRU_HEAD),
        "--steps", "32",
    ],
]

for command in commands:
    print("$", shlex.join(command))
if RUN_AUXILIARY_TRAIN:
    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
else:
    print("По умолчанию выводится только предпросмотр. Сначала замените пути к многопозиционным данным, JSONL коллектора и base policy, и только потом включите RUN_AUXILIARY_TRAIN.")
'''
    ),
    md("## Checkpoint 4: формулировка языка тоже относится к протоколу развёртывания"),
    code(
        r'''
rows = [
    (
        "Place the blue mug on the plate.",
        "0 active steps",
        "0/1",
        "короткий prompt выходит за пределы распределения обучения малой головы, prototype gate постоянно отклоняет передачу управления",
    ),
    (
        "Pick up the blue mug and place it on the plate.",
        "269 active steps",
        "1/1",
        "совпадает с инструкцией обучения; seed3 проходит final strict на 270 шагах",
    ),
]
md_table(["instruction", "learned head", "strict", "Явление"], rows)
'''
    ),
    md(
        "Это не значит, что VLA никогда не может принимать синонимичные инструкции; просто auxiliary head, обученная на 4 коротких наборах данных, пока не обобщает перефразировки языка. При развёртывании сначала зафиксируйте prompt обучения и оценки; в следующем раунде добавьте аугментацию синонимичными инструкциями и отдельно протестируйте language paraphrase gate."
    ),
    md("## Checkpoint 5: почему overall accuracy может обманывать"),
    code(
        r'''
rows = [
    ("Старая линейная gripper head", "99.66%", "0/4", "кадров release очень мало, поэтому общая точность скрывает полный пропуск"),
    ("GRU sequence head", "96.8% balanced", "5/5", "порог подбирается отдельно для переходов close и release"),
]
md_table(["Модель", "Метрика валидации", "held-out release", "Интерпретация"], rows)
'''
    ),
    md("## Checkpoint 6: воспроизведение итогового предиката успеха кодом"),
    code(
        r'''
def final_strict_success(row):
    p = results["strict_predicate"]
    return bool(
        row["legacy_success"]
        and row["max_target_lift"] >= p["min_lift_m"]
        and row["max_lifted_run"] >= p["min_lift_steps"]
        and row["upright_cos"] >= p["min_upright_cos"]
        and row["plate_z_gap"] <= p["max_plate_z_gap_m"]
        and row["plate_xy_displacement"] <= p["max_plate_xy_displacement_m"]
        and row["stable_place_steps"] >= p["stable_place_steps"]
    )


known_success = {
    "legacy_success": True,
    "max_target_lift": 0.11,
    "max_lifted_run": 8,
    "upright_cos": 0.95,
    "plate_z_gap": 0.0448,
    "plate_xy_displacement": 0.0105,
    "stable_place_steps": 5,
}
print("Проходит ли пример итоговую строгую проверку:", final_strict_success(known_success))
'''
    ),
    md("## Checkpoint 7: влияние ошибки seed на выводы"),
    code(
        r'''
seed_bug = results["seed_bug"]
print("Старый код:", seed_bug["old_code"])
print("После исправления:", seed_bug["fixed_code"])
print("Влияние:", seed_bug["impact"])
'''
    ),
    md(
        """
        ## Checkpoint 8: четырёхракурсное воспроизведение успеха

        ![Ключевые кадры четырёхракурсного воспроизведения успеха](../assets/pnp_four_view_strict_success_sequence.jpg)

        <video controls width="960" src="../assets/pnp_four_view_strict_success.mp4"></video>
        """
    ),
    md("## Checkpoint 9: план данных следующего раунда"),
    code(
        r'''
plan = [
    ("Сбор", "после исправления seed собрать 30–50 полных успешных траекторий с разными положениями"),
    ("Разбиение", "по группам: красная и синяя чашка, область старта, contact miss, transport tip, release fail"),
    ("Обучение", "сначала защитить fixed-scene 6/12, затем обучить visual/history head; не добавляйте шаги raw pi0 вслепую"),
    ("Оценка", "fixed scene и random env раздельно; случайных окружений минимум 20–30"),
    ("Порог замены", "новая модель по тому же протоколу превосходит защищённый baseline, а видео без ложных срабатываний толчков и зависаний"),
]
md_table(["Этап", "Действие"], plan)
'''
    ),
]


NOTEBOOKS["06_rocm_debug_playbook.ipynb"].extend([
    md("## Checkpoint 3: кейс протокола оценки pi0 с hard-reset"),
    md(
        """
        Один случай отладки финишера `dynamic_timed` очень типичен: после исправления state с учётом стадии старая непрерывная оценка уже проходила strict `30/30`, но у seed `1036` конечное `xy` однажды подошло близко к порогу. При дальнейшей трассировке выяснилось, что начальное положение чашки для следующего seed иногда выходило за пределы диапазона сэмплирования этого seed, то есть состояние qvel / ctrl / free-joint предыдущего episode загрязнило следующий episode.

        Здесь не стоит сразу считать это проблемой генерализации модели. Более чистый путь — сначала исправить протокол оценки: для малой панели можно использовать `--fresh-env-per-episode`, создавая новое окружение для каждого seed; для полной пакетной оценки рекомендуется `--hard-reset-sim-data`, который очищает нижележащие данные симуляции MuJoCo перед каждым `env.reset(seed)`. После повторного прогона seed сэмплирования политики `1010-1039` (окружение фактически фиксировано на seed 0) в чистом протоколе hard-reset результат остаётся strict `30/30`, legacy `30/30`, mean `xy=0.0219 m`, max `xy=0.0450 m`, а seed `1036` больше не находится на границе.
        """
    ),
    code(
        r'''
rows = [
    ("Симптом", "старая непрерывная оценка 30/30, но у seed 1036 однажды max xy=0.0993 m"),
    ("Аномалия", "у seed 1035 начальное положение чашки оказалось вне диапазона сэмплирования"),
    ("Корневая причина", "env.reset(seed) переустанавливает положения объектов, но не очищает остаточную динамику MuJoCo"),
    ("Исправление", "добавлены --fresh-env-per-episode и --hard-reset-sim-data"),
    ("Чистый результат", "после hard-reset 1010-1039: strict 30/30, mean xy=0.0219 m, max xy=0.0450 m"),
]
md_table(["Пункт", "Запись"], rows)
'''
    ),
    md("## Checkpoint 4: смотрим метрики и видео вместе"),
    code(
        r'''
show_asset("act_dagger_progress_curve.png", width=900)
show_asset("smolvla_red_blue_success.png", width=1000)
'''
    ),
    md("График отвечает на вопрос «каков общий результат», а видео — на вопрос «действительно ли поведение похоже на успех». Оба типа доказательств лучше одновременно включать в отчёт об эксперименте."),
    md("## Checkpoint 5: шаблон сводки результатов"),
    code(
        r'''
rows = [
    ("Устройство", "модель GPU/APU, ROCm, PyTorch, температура и видеопамять"),
    ("Данные", "число episode, тип задачи, прошёл ли аудит физического воспроизведения"),
    ("ACT", "best checkpoint, physical_success, основные типы провалов"),
    ("SmolVLA", "успешность для красной и синей чашки, стратегия сэмплирования, защищённый baseline"),
    ("pi0", "доступ gated, 1-step smoke, статус полного обучения"),
    ("Видео", "один реальный успех и один типичный провал"),
    ("Разбор", "самая важная ловушка и доказательство исправления"),
]
md_table(["Пункт", "Что нужно написать"], rows)
'''
    ),
])


def main() -> None:
    for filename, cells in NOTEBOOKS.items():
        write_nb(filename, cells)


if __name__ == "__main__":
    main()
